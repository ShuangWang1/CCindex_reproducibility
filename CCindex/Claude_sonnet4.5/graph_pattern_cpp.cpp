// graph_pattern_cpp.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <string>
#include <algorithm>
#include <random>
#include <ctime>
#include <cstdint>

namespace py = pybind11;

struct SearchStats {
    std::uint64_t states_visited = 0;
    std::uint64_t vertices_examined = 0;
    std::uint64_t candidates_accepted = 0;
    std::uint64_t rejected_label = 0;
    std::uint64_t rejected_injective = 0;
    std::uint64_t rejected_degree = 0;
    std::uint64_t rejected_triangle_count = 0;
    std::uint64_t rejected_signature = 0;
    std::uint64_t rejected_adjacency = 0;
};

struct AblationOptions {
    bool use_label_index = true;
    bool use_degree_filter = true;
    bool use_triangle_filter = true;
    bool use_reordering = true;
};

AblationOptions optionsFor(const std::string& mode) {
    AblationOptions options;
    if (mode == "full" || mode == "no_cpp") return options;
    if (mode == "no_triangle_filter") {
        options.use_triangle_filter = false;
        return options;
    }
    if (mode == "no_candidate_prefilter") {
        options.use_label_index = false;
        options.use_degree_filter = false;
        return options;
    }
    if (mode == "no_reorder") {
        options.use_reordering = false;
        return options;
    }
    throw std::invalid_argument("Unknown ablation mode: " + mode);
}

// Triangle signature type
using TriangleSignature = std::set<std::pair<std::string, std::string>>;

// Graph structure
class Graph {
public:
    std::unordered_map<int, std::unordered_set<int>> adjacency;
    std::unordered_map<int, std::string> labels;

    void addVertex(int v, const std::string& label) {
        labels[v] = label;
        if (adjacency.find(v) == adjacency.end()) {
            adjacency[v] = std::unordered_set<int>();
        }
    }

    void addEdge(int u, int v) {
        adjacency[u].insert(v);
        adjacency[v].insert(u);
    }

    int vertexCount() const {
        return labels.size();
    }
};

// Function to compute triangle signature
TriangleSignature getTriangleSignature(
    const std::set<std::pair<int, int>>& triangle_pairs,
    const std::unordered_map<int, std::string>& vertex_labels) {

    TriangleSignature labeled_triangles;
    for (const auto& pair : triangle_pairs) {
        std::string l1 = vertex_labels.at(pair.first);
        std::string l2 = vertex_labels.at(pair.second);
        if (l1 > l2) std::swap(l1, l2);
        labeled_triangles.insert({l1, l2});
    }
    return labeled_triangles;
}

// Count local triangles for each vertex
void countLocalTriangles(
    const std::unordered_map<int, std::unordered_set<int>>& adjacency,
    std::unordered_map<int, int>& tri_count,
    std::unordered_map<int, std::set<std::pair<int, int>>>& triangle_pairs) {

    for (const auto& [v, neighbors] : adjacency) {
        tri_count[v] = 0;
        triangle_pairs[v] = std::set<std::pair<int, int>>();

        std::vector<int> neighbor_list(neighbors.begin(), neighbors.end());

        for (size_t i = 0; i < neighbor_list.size(); i++) {
            int x = neighbor_list[i];
            for (size_t j = i + 1; j < neighbor_list.size(); j++) {
                int y = neighbor_list[j];
                if (adjacency.at(x).count(y) > 0) {
                    tri_count[v]++;
                    int minv = std::min(x, y);
                    int maxv = std::max(x, y);
                    triangle_pairs[v].insert({minv, maxv});
                }
            }
        }
    }
}

// Graph cache for preprocessed data
class GraphCache {
public:
    std::unordered_map<int, std::unordered_set<int>> adjacency;
    std::unordered_map<int, std::string> labels;
    std::unordered_map<int, int> tri_count;
    std::unordered_map<int, std::set<std::pair<int, int>>> tri_pairs;
    std::unordered_map<std::string, std::vector<int>> label_to_vertices;
    std::unordered_map<int, int> degrees;
    std::unordered_map<int, TriangleSignature> triangle_signatures;

    GraphCache(const Graph& graph) {
        adjacency = graph.adjacency;
        labels = graph.labels;

        // Compute triangle counts
        countLocalTriangles(adjacency, tri_count, tri_pairs);

        // Compute degrees
        for (const auto& [v, adj] : adjacency) {
            degrees[v] = adj.size();
        }

        // Build label to vertices mapping
        for (const auto& [v, label] : labels) {
            label_to_vertices[label].push_back(v);
        }

        // Compute triangle signatures
        for (const auto& [v, _] : adjacency) {
            triangle_signatures[v] = getTriangleSignature(tri_pairs[v], labels);
        }
    }
};

// State for the isomorphism search
class TriState {
public:
    std::unordered_map<int, std::vector<int>> pattern_adj;
    std::unordered_map<int, std::string> pattern_labels;
    const GraphCache* graph_cache;

    std::unordered_map<int, int> mapping;
    std::unordered_map<int, int> reverse_mapping;
    std::unordered_set<int> p_next, p_cur, g_next, g_cur;

    int pattern_size;
    std::unordered_map<int, int> pattern_tri_count;
    std::unordered_map<int, std::set<std::pair<int, int>>> pattern_triangle_pairs;
    std::unordered_map<int, TriangleSignature> pattern_signatures;

    TriState(const Graph& pattern, const GraphCache* cache,
             const std::unordered_map<int, int>& p_tri_count,
             const std::unordered_map<int, std::set<std::pair<int, int>>>& p_tri_pairs)
        : graph_cache(cache), pattern_tri_count(p_tri_count), pattern_triangle_pairs(p_tri_pairs) {

        pattern_labels = pattern.labels;
        pattern_size = pattern.vertexCount();

        // Convert adjacency to vector format and sort in reverse order
        for (const auto& [v, neighbors] : pattern.adjacency) {
            pattern_adj[v] = std::vector<int>(neighbors.begin(), neighbors.end());
            std::sort(pattern_adj[v].rbegin(), pattern_adj[v].rend());
        }

        // Compute pattern signatures
        for (const auto& [v, _] : pattern_adj) {
            pattern_signatures[v] = getTriangleSignature(pattern_triangle_pairs[v], pattern_labels);
        }
    }

    bool isComplete() const {
        return mapping.size() == static_cast<size_t>(pattern_size);
    }

    void addPair(int p_v, int g_v) {
        mapping[p_v] = g_v;
        reverse_mapping[g_v] = p_v;

        for (int p_u : pattern_adj[p_v]) {
            if (mapping.find(p_u) == mapping.end()) {
                if (p_next.find(p_u) == p_next.end() && p_cur.find(p_u) == p_cur.end()) {
                    p_next.insert(p_u);
                }
            }
        }

        p_next.erase(p_v);
        p_cur.erase(p_v);

        for (int g_u : graph_cache->adjacency.at(g_v)) {
            if (reverse_mapping.find(g_u) == reverse_mapping.end()) {
                if (g_next.find(g_u) == g_next.end() && g_cur.find(g_u) == g_cur.end()) {
                    g_next.insert(g_u);
                }
            }
        }

        g_next.erase(g_v);
        g_cur.erase(g_v);
    }

    void removePair(int p_v, int g_v) {
        std::vector<int> invalid_p, invalid_g;

        for (int p_u : pattern_adj[p_v]) {
            if (mapping.find(p_u) == mapping.end()) {
                bool keep = false;
                for (int p_x : pattern_adj[p_u]) {
                    if (mapping.find(p_x) != mapping.end() && p_x != p_v) {
                        keep = true;
                        break;
                    }
                }
                if (!keep) invalid_p.push_back(p_u);
            }
        }

        for (int g_u : graph_cache->adjacency.at(g_v)) {
            if (reverse_mapping.find(g_u) == reverse_mapping.end()) {
                bool keep = false;
                for (int t_x : graph_cache->adjacency.at(g_u)) {
                    if (reverse_mapping.find(t_x) != reverse_mapping.end() && t_x != g_v) {
                        keep = true;
                        break;
                    }
                }
                if (!keep) invalid_g.push_back(g_u);
            }
        }

        mapping.erase(p_v);
        reverse_mapping.erase(g_v);

        for (int p_u : invalid_p) {
            p_next.erase(p_u);
            p_cur.erase(p_u);
        }

        for (int g_u : invalid_g) {
            g_next.erase(g_u);
            g_cur.erase(g_u);
        }
    }
};

// Select next pattern vertex with edge_dict support
int selectPatternVertex(TriState& state, const py::object& edge_dict_obj,
                        const AblationOptions& options) {
    if (!options.use_reordering) {
        int selected = state.pattern_size;
        for (const auto& [v, _] : state.pattern_adj) {
            if (!state.mapping.count(v)) selected = std::min(selected, v);
        }
        return selected;
    }
    if (!state.p_next.empty()) {
        std::vector<int> candidates(state.p_next.begin(), state.p_next.end());
        if (candidates.size() > 1) {
            return *std::max_element(candidates.begin(), candidates.end(),
                [&state](int a, int b) {
                    int count_a = 0, count_b = 0;
                    for (int u : state.pattern_adj[a]) {
                        if (state.mapping.find(u) != state.mapping.end()) count_a++;
                    }
                    for (int u : state.pattern_adj[b]) {
                        if (state.mapping.find(u) != state.mapping.end()) count_b++;
                    }
                    if (count_a != count_b) return count_a < count_b;
                    return a > b;
                });
        }
        return candidates[0];
    } else if (!state.p_cur.empty()) {
        return *state.p_cur.begin();
    } else {
        std::vector<int> candidates;
        for (const auto& [v, _] : state.pattern_adj) {
            if (state.mapping.find(v) == state.mapping.end()) {
                candidates.push_back(v);
            }
        }

        // If edge_dict is None, use degree-based selection
        if (edge_dict_obj.is_none()) {
            return *std::max_element(candidates.begin(), candidates.end(),
                [&state](int a, int b) {
                    if (state.pattern_adj[a].size() != state.pattern_adj[b].size()) {
                        return state.pattern_adj[a].size() < state.pattern_adj[b].size();
                    }
                    return a > b;
                });
        }

        // Use edge_dict for selection
        py::dict edge_dict = edge_dict_obj.cast<py::dict>();
        std::unordered_map<int, double> pattern_vertex_value;

        for (int k : candidates) {
            pattern_vertex_value[k] = 0.0;
            for (int a : state.pattern_adj[k]) {
                std::string l1 = state.pattern_labels[k];
                std::string l2 = state.pattern_labels[a];
                if (l1 > l2) std::swap(l1, l2);
                std::string mtf = l1 + "_" + l2;

                if (!edge_dict.contains(mtf.c_str())) {
                    pattern_vertex_value[k] += 1.0;
                } else {
                    double val = edge_dict[mtf.c_str()].cast<double>();
                    pattern_vertex_value[k] += 1.0 / val;
                }
            }
        }

        return *std::max_element(candidates.begin(), candidates.end(),
            [&pattern_vertex_value](int a, int b) {
                if (pattern_vertex_value[a] != pattern_vertex_value[b]) {
                    return pattern_vertex_value[a] < pattern_vertex_value[b];
                }
                return a > b;
            });
    }
}

// Check if signature1 is subset of signature2
bool isSubset(const TriangleSignature& sig1, const TriangleSignature& sig2) {
    for (const auto& elem : sig1) {
        if (sig2.find(elem) == sig2.end()) {
            return false;
        }
    }
    return true;
}

// Main recursive search function
bool findPatternTriangles(TriState& state,
                         std::vector<std::vector<int>>& solutions,
                         bool allow_multiple,
                         int threshold,
                         const py::object& edge_dict,
                         const AblationOptions& options,
                         SearchStats& stats) {
    stats.states_visited++;
    if (state.isComplete()) {
        std::vector<int> sol(state.pattern_size);
        for (const auto& [p_v, g_v] : state.mapping) {
            sol[p_v] = g_v;
        }
        solutions.push_back(sol);

        if (threshold > 0 && solutions.size() >= static_cast<size_t>(threshold)) {
            std::set<std::set<int>> unique;
            for (const auto& candidate : solutions) {
                unique.insert(std::set<int>(candidate.begin(), candidate.end()));
            }
            if (unique.size() >= static_cast<size_t>(threshold)) return true;
        }
        return !allow_multiple;
    }

    int p_v = selectPatternVertex(state, edge_dict, options);
    const std::string& p_label = state.pattern_labels[p_v];
    int p_tri_count = state.pattern_tri_count[p_v];
    int p_degree = state.pattern_adj[p_v].size();

    std::vector<int> connected_g_vertices;
    for (int p_u : state.pattern_adj[p_v]) {
        if (state.mapping.find(p_u) != state.mapping.end()) {
            connected_g_vertices.push_back(state.mapping[p_u]);
        }
    }

    std::vector<int> candidate_pool;
    if (options.use_label_index) {
        auto found = state.graph_cache->label_to_vertices.find(p_label);
        if (found != state.graph_cache->label_to_vertices.end()) candidate_pool = found->second;
    } else {
        for (const auto& [g_v, _] : state.graph_cache->labels) candidate_pool.push_back(g_v);
    }
    if (!options.use_reordering) std::sort(candidate_pool.begin(), candidate_pool.end());

    std::vector<int> candidates;
    for (int g_v : candidate_pool) {
            stats.vertices_examined++;
            if (state.graph_cache->labels.at(g_v) != p_label) {
                stats.rejected_label++;
                continue;
            }
            if (state.reverse_mapping.find(g_v) != state.reverse_mapping.end()) {
                stats.rejected_injective++;
                continue;
            }

            if (options.use_degree_filter && state.graph_cache->degrees.at(g_v) < p_degree) {
                stats.rejected_degree++;
                continue;
            }

            if (options.use_triangle_filter && state.graph_cache->tri_count.at(g_v) < p_tri_count) {
                stats.rejected_triangle_count++;
                continue;
            }

            bool feasible = true;
            for (int g_u : connected_g_vertices) {
                if (state.graph_cache->adjacency.at(g_v).find(g_u) ==
                    state.graph_cache->adjacency.at(g_v).end()) {
                    feasible = false;
                    break;
                }
            }

            if (!feasible) {
                stats.rejected_adjacency++;
                continue;
            }

            const TriangleSignature& pattern_sig = state.pattern_signatures[p_v];
            const TriangleSignature& graph_sig = state.graph_cache->triangle_signatures.at(g_v);

            if (options.use_triangle_filter && !isSubset(pattern_sig, graph_sig)) {
                stats.rejected_signature++;
                continue;
            }

            candidates.push_back(g_v);
            stats.candidates_accepted++;
    }

    if (options.use_reordering && candidates.size() > 1) {
        std::sort(candidates.begin(), candidates.end(),
            [&state, &connected_g_vertices](int a, int b) {
                int conn_a = 0, conn_b = 0;
                for (int g_u : connected_g_vertices) {
                    if (state.graph_cache->adjacency.at(a).find(g_u) !=
                        state.graph_cache->adjacency.at(a).end()) conn_a++;
                    if (state.graph_cache->adjacency.at(b).find(g_u) !=
                        state.graph_cache->adjacency.at(b).end()) conn_b++;
                }
                if (conn_a != conn_b) return conn_a > conn_b;
                if (state.graph_cache->tri_count.at(a) != state.graph_cache->tri_count.at(b)) {
                    return state.graph_cache->tri_count.at(a) > state.graph_cache->tri_count.at(b);
                }
                return a < b;
            });
    }

    for (int g_v : candidates) {
        state.addPair(p_v, g_v);
        bool stop_early = findPatternTriangles(state, solutions, allow_multiple, threshold,
                                               edge_dict, options, stats);
        state.removePair(p_v, g_v);

        if (stop_early) {
            return true;
        }
    }

    return false;
}

// Main pattern matching function
std::pair<int, std::vector<std::vector<int>>> findPatternMatches(
    const GraphCache& graph_cache,
    const Graph& pattern,
    bool find_all = true,
    int threshold = 0,
    py::object edge_dict = py::none()) {

    std::unordered_map<int, int> p_tri_count;
    std::unordered_map<int, std::set<std::pair<int, int>>> p_tri_pairs;
    countLocalTriangles(pattern.adjacency, p_tri_count, p_tri_pairs);

    TriState state(pattern, &graph_cache, p_tri_count, p_tri_pairs);

    std::vector<std::vector<int>> solutions;
    SearchStats stats;
    findPatternTriangles(state, solutions, find_all, threshold, edge_dict,
                         optionsFor("full"), stats);

    // Deduplicate solutions
    std::set<std::set<int>> unique_solutions;
    for (const auto& sol : solutions) {
        unique_solutions.insert(std::set<int>(sol.begin(), sol.end()));
    }

    std::vector<std::vector<int>> result;
    for (const auto& sol_set : unique_solutions) {
        result.push_back(std::vector<int>(sol_set.begin(), sol_set.end()));
    }

    return {result.size(), result};
}

py::tuple findPatternMatchesAblation(
    const GraphCache& graph_cache,
    const Graph& pattern,
    bool find_all = true,
    int threshold = 0,
    py::object edge_dict = py::none(),
    const std::string& mode = "full") {
    std::unordered_map<int, int> p_tri_count;
    std::unordered_map<int, std::set<std::pair<int, int>>> p_tri_pairs;
    countLocalTriangles(pattern.adjacency, p_tri_count, p_tri_pairs);
    TriState state(pattern, &graph_cache, p_tri_count, p_tri_pairs);
    std::vector<std::vector<int>> solutions;
    SearchStats stats;
    findPatternTriangles(state, solutions, find_all, threshold, edge_dict,
                         optionsFor(mode), stats);

    std::set<std::set<int>> unique_solutions;
    for (const auto& sol : solutions) {
        unique_solutions.insert(std::set<int>(sol.begin(), sol.end()));
    }
    std::vector<std::vector<int>> result;
    for (const auto& sol : unique_solutions) result.emplace_back(sol.begin(), sol.end());

    py::dict out;
    out["states_visited"] = stats.states_visited;
    out["vertices_examined"] = stats.vertices_examined;
    out["candidates_accepted"] = stats.candidates_accepted;
    out["rejected_label"] = stats.rejected_label;
    out["rejected_injective"] = stats.rejected_injective;
    out["rejected_degree"] = stats.rejected_degree;
    out["rejected_triangle_count"] = stats.rejected_triangle_count;
    out["rejected_signature"] = stats.rejected_signature;
    out["rejected_adjacency"] = stats.rejected_adjacency;
    return py::make_tuple(result.size(), result, out);
}

// Python bindings
#ifndef GRAPH_PATTERN_MODULE_NAME
#define GRAPH_PATTERN_MODULE_NAME graph_pattern_cpp
#endif

PYBIND11_MODULE(GRAPH_PATTERN_MODULE_NAME, m) {
    m.doc() = "C++ implementation of graph pattern matching";

    py::class_<Graph>(m, "Graph", py::module_local())
        .def(py::init<>())
        .def("addVertex", &Graph::addVertex)
        .def("addEdge", &Graph::addEdge)
        .def("vertexCount", &Graph::vertexCount);

    py::class_<GraphCache>(m, "GraphCache", py::module_local())
        .def(py::init<const Graph&>());

    m.def("find_pattern_matches", &findPatternMatches,
          py::arg("graph_cache"),
          py::arg("pattern"),
          py::arg("find_all") = true,
          py::arg("threshold") = 0,
          py::arg("edge_dict") = py::none(),
          "Find pattern matches in graph");
    m.def("find_pattern_matches_ablation", &findPatternMatchesAblation,
          py::arg("graph_cache"), py::arg("pattern"),
          py::arg("find_all") = true, py::arg("threshold") = 0,
          py::arg("edge_dict") = py::none(), py::arg("mode") = "full",
          "Find pattern matches and return ablation search counters");
}
