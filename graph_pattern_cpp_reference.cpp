// Corrected version - automorphism breaking disabled for correctness
// Keeps all other optimizations (triangle pruning, adjacency matrix, etc.)
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <set>
#include <string>
#include <algorithm>

namespace py = pybind11;

struct FastStringHash {
    inline std::size_t operator()(const std::string& s) const noexcept {
        std::size_t hash = 14695981039346656037ULL;
        for (char c : s) {
            hash ^= static_cast<std::size_t>(c);
            hash *= 1099511628211ULL;
        }
        return hash;
    }
};

template<typename T>
class SmallVector {
private:
    std::vector<T> data_;
public:
    SmallVector() { data_.reserve(16); }
    void push_back(const T& val) { data_.push_back(val); }
    const T* data() const { return data_.data(); }
    size_t size() const { return data_.size(); }
    const T& operator[](size_t idx) const { return data_[idx]; }
    typename std::vector<T>::const_iterator begin() const { return data_.begin(); }
    typename std::vector<T>::const_iterator end() const { return data_.end(); }
};

using TriangleSignature = std::set<std::pair<std::string, std::string>>;

inline TriangleSignature getTriangleSignature(
    const std::set<std::pair<int, int>>& triangle_pairs,
    const std::vector<std::string>& vertex_labels) {
    TriangleSignature labeled_triangles;
    for (const auto& pair : triangle_pairs) {
        const std::string& l1 = vertex_labels[pair.first];
        const std::string& l2 = vertex_labels[pair.second];
        if (l1 <= l2) {
            labeled_triangles.emplace(l1, l2);
        } else {
            labeled_triangles.emplace(l2, l1);
        }
    }
    return labeled_triangles;
}

class Graph {
public:
    std::vector<SmallVector<int>> adjacency;
    std::vector<std::string> labels;
    int vertex_count;
    Graph() : vertex_count(0) {}
    void reserve(int n) { adjacency.reserve(n); labels.reserve(n); }
    void addVertex(int v, const std::string& label) {
        if (v >= vertex_count) {
            adjacency.resize(v + 1);
            labels.resize(v + 1);
            vertex_count = v + 1;
        }
        labels[v] = label;
    }
    void addEdge(int u, int v) {
        adjacency[u].push_back(v);
        adjacency[v].push_back(u);
    }
    int vertexCount() const { return vertex_count; }
};

inline void countLocalTriangles(
    const std::vector<SmallVector<int>>& adjacency,
    std::vector<int>& tri_count,
    std::vector<std::set<std::pair<int, int>>>& triangle_pairs) {
    int n = adjacency.size();
    tri_count.resize(n, 0);
    triangle_pairs.resize(n);

    std::vector<std::vector<int>> sorted_adj(n);
    for (int v = 0; v < n; v++) {
        sorted_adj[v].assign(adjacency[v].begin(), adjacency[v].end());
        std::sort(sorted_adj[v].begin(), sorted_adj[v].end());
    }

    // Fixed triangle counting - only increment for current vertex
    for (int v = 0; v < n; v++) {
        const auto& neighbors = sorted_adj[v];
        for (size_t i = 0; i < neighbors.size(); i++) {
            int x = neighbors[i];
            for (size_t j = i + 1; j < neighbors.size(); j++) {
                int y = neighbors[j];
                if (std::binary_search(sorted_adj[x].begin(), sorted_adj[x].end(), y)) {
                    tri_count[v]++;
                    triangle_pairs[v].emplace(std::min(x, y), std::max(x, y));
                    triangle_pairs[x].emplace(std::min(v, y), std::max(v, y));
                    triangle_pairs[y].emplace(std::min(v, x), std::max(v, x));
                }
            }
        }
    }
}

class GraphCache {
public:
    std::vector<SmallVector<int>> adjacency;
    std::vector<std::string> labels;
    std::vector<int> tri_count;
    std::vector<std::set<std::pair<int, int>>> tri_pairs;
    std::unordered_map<std::string, std::vector<int>, FastStringHash> label_to_vertices;
    std::vector<int> degrees;
    std::vector<TriangleSignature> triangle_signatures;
    std::vector<std::vector<bool>> adjacency_matrix;
    int vertex_count;
    bool use_adjacency_matrix;

    GraphCache(const Graph& graph) {
        vertex_count = graph.vertex_count;
        adjacency = graph.adjacency;
        labels = graph.labels;
        #ifdef __ARM_ARCH
            use_adjacency_matrix = (vertex_count < 15000);
        #else
            use_adjacency_matrix = (vertex_count < 5000);
        #endif
        degrees.resize(vertex_count);
        triangle_signatures.resize(vertex_count);
        countLocalTriangles(adjacency, tri_count, tri_pairs);
        for (int v = 0; v < vertex_count; v++) degrees[v] = adjacency[v].size();
        for (int v = 0; v < vertex_count; v++) label_to_vertices[labels[v]].push_back(v);
        for (auto& kv : label_to_vertices) std::sort(kv.second.begin(), kv.second.end());
        for (int v = 0; v < vertex_count; v++) triangle_signatures[v] = getTriangleSignature(tri_pairs[v], labels);
        if (use_adjacency_matrix) {
            adjacency_matrix.resize(vertex_count, std::vector<bool>(vertex_count, false));
            for (int v = 0; v < vertex_count; v++)
                for (int u : adjacency[v]) adjacency_matrix[v][u] = true;
        }
    }
    inline bool hasEdge(int u, int v) const {
        if (use_adjacency_matrix) return adjacency_matrix[u][v];
        for (int neighbor : adjacency[u]) if (neighbor == v) return true;
        return false;
    }
};

class TriState {
public:
    std::vector<std::vector<int>> pattern_adj;
    std::vector<std::string> pattern_labels;
    const GraphCache* graph_cache;
    std::vector<int> mapping, reverse_mapping;
    int pattern_size;
    std::vector<int> pattern_tri_count;
    std::vector<std::set<std::pair<int, int>>> pattern_triangle_pairs;
    std::vector<TriangleSignature> pattern_signatures;
    std::vector<int> pattern_degrees;

    TriState(const Graph& pattern, const GraphCache* cache,
             const std::vector<int>& p_tri_count,
             const std::vector<std::set<std::pair<int, int>>>& p_tri_pairs)
        : graph_cache(cache), pattern_tri_count(p_tri_count), pattern_triangle_pairs(p_tri_pairs) {
        pattern_labels = pattern.labels;
        pattern_size = pattern.vertexCount();
        mapping.assign(pattern_size, -1);
        reverse_mapping.assign(cache->vertex_count, -1);
        pattern_adj.resize(pattern_size);
        pattern_degrees.resize(pattern_size);
        pattern_signatures.resize(pattern_size);
        for (int v = 0; v < pattern_size; v++) {
            pattern_adj[v].assign(pattern.adjacency[v].begin(), pattern.adjacency[v].end());
            std::sort(pattern_adj[v].begin(), pattern_adj[v].end());
            pattern_degrees[v] = pattern_adj[v].size();
        }
        for (int v = 0; v < pattern_size; v++)
            pattern_signatures[v] = getTriangleSignature(pattern_triangle_pairs[v], pattern_labels);
    }

    inline bool isComplete() const {
        for (int i = 0; i < pattern_size; i++)
            if (mapping[i] == -1) return false;
        return true;
    }
    inline void addPair(int p_v, int g_v) { mapping[p_v] = g_v; reverse_mapping[g_v] = p_v; }
    inline void removePair(int p_v, int g_v) { mapping[p_v] = -1; reverse_mapping[g_v] = -1; }
};

inline int selectPatternVertex(TriState& state, const py::object&) {
    // Select vertex with most mapped neighbors (best pruning)
    int best_vertex = -1, max_mapped_neighbors = -1;
    for (int v = 0; v < state.pattern_size; v++) {
        if (state.mapping[v] != -1) continue;
        int mapped_neighbors = 0;
        for (int u : state.pattern_adj[v])
            if (state.mapping[u] != -1) mapped_neighbors++;
        if (mapped_neighbors > max_mapped_neighbors) {
            max_mapped_neighbors = mapped_neighbors;
            best_vertex = v;
        }
    }
    if (best_vertex != -1 && max_mapped_neighbors > 0) return best_vertex;

    // Otherwise select vertex with highest degree
    int max_degree = -1;
    for (int v = 0; v < state.pattern_size; v++)
        if (state.mapping[v] == -1 && state.pattern_degrees[v] > max_degree) {
            max_degree = state.pattern_degrees[v];
            best_vertex = v;
        }
    return best_vertex;
}

inline bool isSubset(const TriangleSignature& sig1, const TriangleSignature& sig2) {
    if (sig1.size() > sig2.size()) return false;
    for (const auto& elem : sig1)
        if (sig2.find(elem) == sig2.end()) return false;
    return true;
}

bool findPatternTriangles(TriState& state, std::vector<std::vector<int>>& solutions,
                         bool allow_multiple, int threshold, const py::object& edge_dict) {
    if (state.isComplete()) {
        std::vector<int> sol(state.pattern_size);
        for (int p_v = 0; p_v < state.pattern_size; p_v++) sol[p_v] = state.mapping[p_v];
        solutions.push_back(sol);
        if (threshold > 0 && solutions.size() >= static_cast<size_t>(threshold)) return true;
        return !allow_multiple;
    }
    int p_v = selectPatternVertex(state, edge_dict);
    if (p_v == -1) return false;
    const std::string& p_label = state.pattern_labels[p_v];
    int p_tri_count = state.pattern_tri_count[p_v];
    int p_degree = state.pattern_degrees[p_v];
    std::vector<int> connected_g_vertices;
    connected_g_vertices.reserve(p_degree);
    for (int p_u : state.pattern_adj[p_v])
        if (state.mapping[p_u] != -1) connected_g_vertices.push_back(state.mapping[p_u]);
    std::sort(connected_g_vertices.begin(), connected_g_vertices.end());

    // No automorphism breaking - guarantee correctness
    std::vector<int> candidates;
    const auto& label_vertices_it = state.graph_cache->label_to_vertices.find(p_label);
    if (label_vertices_it == state.graph_cache->label_to_vertices.end()) return false;
    const std::vector<int>& label_vertices = label_vertices_it->second;
    candidates.reserve(label_vertices.size());

    for (int g_v : label_vertices) {
        if (g_v < 0 || g_v >= static_cast<int>(state.reverse_mapping.size())) continue;
        if (state.reverse_mapping[g_v] != -1) continue;
        if (g_v >= static_cast<int>(state.graph_cache->degrees.size())) continue;
        if (state.graph_cache->degrees[g_v] < p_degree) continue;
        if (g_v >= static_cast<int>(state.graph_cache->tri_count.size())) continue;
        if (state.graph_cache->tri_count[g_v] < p_tri_count) continue;

        // Check connectivity to already-mapped neighbors
        bool feasible = true;
        for (int g_u : connected_g_vertices) {
            if (!state.graph_cache->hasEdge(g_v, g_u)) {
                feasible = false;
                break;
            }
        }
        if (!feasible) continue;

        // Check triangle signature compatibility
        if (!isSubset(state.pattern_signatures[p_v], state.graph_cache->triangle_signatures[g_v])) continue;

        candidates.push_back(g_v);
    }

    // Sort candidates for better pruning (most constrained first)
    if (candidates.size() > 1) {
        std::sort(candidates.begin(), candidates.end(),
            [&state, &connected_g_vertices](int a, int b) {
                int conn_a = 0, conn_b = 0;
                for (int g_u : connected_g_vertices) {
                    if (state.graph_cache->hasEdge(a, g_u)) conn_a++;
                    if (state.graph_cache->hasEdge(b, g_u)) conn_b++;
                }
                if (conn_a != conn_b) return conn_a > conn_b;
                if (state.graph_cache->tri_count[a] != state.graph_cache->tri_count[b])
                    return state.graph_cache->tri_count[a] > state.graph_cache->tri_count[b];
                return a < b;
            });
    }

    // Try each candidate
    for (int g_v : candidates) {
        state.addPair(p_v, g_v);
        bool stop_early = findPatternTriangles(state, solutions, allow_multiple, threshold, edge_dict);
        state.removePair(p_v, g_v);
        if (stop_early) return true;
    }
    return false;
}

std::pair<int, std::vector<std::vector<int>>> findPatternMatches(
    const GraphCache& graph_cache, const Graph& pattern,
    bool find_all = true, int threshold = 0, py::object edge_dict = py::none()) {
    std::vector<int> p_tri_count;
    std::vector<std::set<std::pair<int, int>>> p_tri_pairs;
    countLocalTriangles(pattern.adjacency, p_tri_count, p_tri_pairs);
    TriState state(pattern, &graph_cache, p_tri_count, p_tri_pairs);
    std::vector<std::vector<int>> solutions;
    solutions.reserve(1000);
    findPatternTriangles(state, solutions, find_all, threshold, edge_dict);

    // Deduplicate solutions
    std::set<std::set<int>> unique_solutions;
    for (const auto& sol : solutions) unique_solutions.insert(std::set<int>(sol.begin(), sol.end()));

    std::vector<std::vector<int>> result;
    result.reserve(unique_solutions.size());
    for (const auto& sol_set : unique_solutions) {
        std::vector<int> sol_vec(sol_set.begin(), sol_set.end());
        std::sort(sol_vec.begin(), sol_vec.end());
        result.push_back(sol_vec);
    }
    std::sort(result.begin(), result.end());
    return {result.size(), result};
}

PYBIND11_MODULE(graph_pattern_cpp, m) {
    m.doc() = "Optimized C++ graph pattern matching";
    py::class_<Graph>(m, "Graph")
        .def(py::init<>())
        .def("addVertex", &Graph::addVertex)
        .def("addEdge", &Graph::addEdge)
        .def("vertexCount", &Graph::vertexCount)
        .def("reserve", &Graph::reserve);
    py::class_<GraphCache>(m, "GraphCache")
        .def(py::init<const Graph&>())
        .def_readonly("vertex_count", &GraphCache::vertex_count)
        .def_readonly("use_adjacency_matrix", &GraphCache::use_adjacency_matrix)
        .def_readonly("tri_count", &GraphCache::tri_count)
        .def_readonly("degrees", &GraphCache::degrees)
        .def_readonly("labels", &GraphCache::labels);
    m.def("find_pattern_matches", &findPatternMatches,
          py::arg("graph_cache"), py::arg("pattern"),
          py::arg("find_all") = true, py::arg("threshold") = 0,
          py::arg("edge_dict") = py::none(),
          "Find pattern matches in graph");
}
