// ccindex, 2026-10-01 version.  Same search and same Python interface/output as ../graph_pattern_cpp.cpp (the 2026-09-30
// version with the neighbour candidate pool): triangle-pruned backtracking that maps the pattern one vertex at a time,
// a candidate must have the vertex's label, at least its degree, at least its triangle count, a superset of its triangle
// label signature, and an edge to the image of every mapped pattern neighbour.  The answer is the set of distinct
// target node sets, sorted.  What changed, for speed only (each is exact, none drops a feasible candidate):
//   1. Labels are integer ids (string compares and string hashing gone from the search).
//   2. Triangle signatures are sorted arrays of label-pair codes plus a 64-bit summary mask, so the subset test is one
//      AND for most rejections and a short merge otherwise (was std::set<std::pair<std::string,std::string>> lookups).
//   3. The matching order is computed once per query (the old per-step choice depends only on which pattern vertices
//      are mapped, which is the same at every node of a given depth): root = a vertex of the target's rarest label
//      among the pattern's labels (ties: more triangles, then higher degree), then the vertex with most mapped
//      neighbours (ties: rarer label, then higher degree).
//   4. No per-step sort of the candidates (it changed the visiting order only; every candidate is still tried).
//   5. Used-target-vertex test by scanning the <= pattern-size current images (no target-size array per query).
//   6. Distinct node sets kept in a hash set as mappings are found (was: every mapping kept, then
//      std::set<std::set<int>>).
//   7. Target adjacency in one sorted array (CSR); edge test by a flat bit matrix below 5000 vertices (as before),
//      otherwise a scan of the shorter of the two neighbour lists.
//   8. A pattern label absent from the target answers 0 at once.
//   9. Symmetry breaking: of mappings that differ only by an automorphism of the pattern (same node set), one is
//      searched (conditions image(a) < image(b), Grochow & Kellis 2007; see symmetry_conditions).
//  10. count_pattern_matches: the number of distinct node sets without building them for Python.
// Patterns and targets are taken as simple graphs (repeated edges count once).  threshold > 0 now counts distinct node
// sets (was mappings); find_all=False still stops at the first mapping.  search_stats is for measurement.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

class Graph {
public:
    std::vector<std::vector<int>> adjacency;
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

static inline uint64_t mask_bit(uint64_t code) { return 1ULL << ((code * 0x9E3779B97F4A7C15ULL) >> 58); }

// Sorted, duplicate-free CSR adjacency of a Graph.
static void to_csr(const Graph& g, std::vector<int>& off, std::vector<int>& adj) {
    int n = g.vertex_count;
    off.assign(n + 1, 0);
    adj.clear();
    for (int v = 0; v < n; v++) {
        std::vector<int> nb(g.adjacency[v].begin(), g.adjacency[v].end());
        std::sort(nb.begin(), nb.end());
        nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
        adj.insert(adj.end(), nb.begin(), nb.end());
        off[v + 1] = static_cast<int>(adj.size());
    }
}

// For every vertex v: tri[v] = triangles through v; sig[sig_off[v] .. sig_off[v+1]) = sorted distinct codes of the
// (smaller, larger) label-id pairs of the edges opposite v in those triangles; mask[v] = their summary bits.
static void triangles(int n, const std::vector<int>& off, const std::vector<int>& adj, const std::vector<int>& lab,
                      uint64_t L, std::vector<int>& tri, std::vector<int>& sig_off, std::vector<uint64_t>& sig,
                      std::vector<uint64_t>& mask) {
    tri.assign(n, 0);
    sig_off.assign(n + 1, 0);
    sig.clear();
    mask.assign(n, 0);
    std::vector<uint64_t> buf;
    for (int v = 0; v < n; v++) {
        buf.clear();
        for (int i = off[v]; i < off[v + 1]; i++) {
            int x = adj[i];
            const int* xb = adj.data() + off[x];
            const int* xe = adj.data() + off[x + 1];
            for (int j = i + 1; j < off[v + 1]; j++) {
                int y = adj[j];
                if (std::binary_search(xb, xe, y)) {
                    tri[v]++;
                    uint64_t a = lab[x], b = lab[y];
                    if (a > b) std::swap(a, b);
                    buf.push_back(a * L + b);
                }
            }
        }
        std::sort(buf.begin(), buf.end());
        buf.erase(std::unique(buf.begin(), buf.end()), buf.end());
        for (uint64_t c : buf) mask[v] |= mask_bit(c);
        sig.insert(sig.end(), buf.begin(), buf.end());
        sig_off[v + 1] = static_cast<int>(sig.size());
    }
}

class GraphCache {
public:
    int vertex_count;
    bool use_adjacency_matrix;
    std::vector<std::string> labels;
    std::vector<int> tri_count, degrees;
    std::unordered_map<std::string, int> label_id;
    std::vector<std::vector<int>> label_vertices;     // by label id, sorted
    std::vector<int> lab, off, adj, sig_off;
    std::vector<uint64_t> sig, mask, bits;
    size_t words = 0;

    GraphCache(const Graph& graph) {
        vertex_count = graph.vertex_count;
        labels = graph.labels;
        use_adjacency_matrix = (vertex_count < 5000);
        lab.resize(vertex_count);
        for (int v = 0; v < vertex_count; v++) {
            auto it = label_id.emplace(labels[v], static_cast<int>(label_id.size())).first;
            lab[v] = it->second;
        }
        label_vertices.resize(label_id.size());
        for (int v = 0; v < vertex_count; v++) label_vertices[lab[v]].push_back(v);
        to_csr(graph, off, adj);
        degrees.resize(vertex_count);
        for (int v = 0; v < vertex_count; v++) degrees[v] = off[v + 1] - off[v];
        triangles(vertex_count, off, adj, lab, label_id.size(), tri_count, sig_off, sig, mask);
        if (use_adjacency_matrix) {
            words = (static_cast<size_t>(vertex_count) + 63) / 64;
            bits.assign(words * vertex_count, 0);
            for (int v = 0; v < vertex_count; v++)
                for (int i = off[v]; i < off[v + 1]; i++) bits[v * words + (adj[i] >> 6)] |= 1ULL << (adj[i] & 63);
        }
    }
    inline bool hasEdge(int u, int v) const {
        if (use_adjacency_matrix) return (bits[u * words + (v >> 6)] >> (v & 63)) & 1ULL;
        if (off[u + 1] - off[u] > off[v + 1] - off[v]) std::swap(u, v);
        for (int i = off[u]; i < off[u + 1]; i++) if (adj[i] == v) return true;
        return false;
    }
};

struct VecHash {
    size_t operator()(const std::vector<int>& s) const noexcept {
        uint64_t h = 1469598103934665603ULL;
        for (int x : s) { h ^= static_cast<uint64_t>(x); h *= 1099511628211ULL; h ^= h >> 29; }
        return static_cast<size_t>(h);
    }
};

struct Search {
    const GraphCache& g;
    int P;
    std::vector<int> order;                  // pattern vertex at each depth
    std::vector<int> plab, pdeg, ptri;       // per depth
    std::vector<std::vector<uint64_t>> psig; // per depth
    std::vector<uint64_t> pmask;             // per depth
    std::vector<std::vector<int>> back;      // per depth: depths of its earlier-mapped pattern neighbours
    std::vector<std::vector<int>> below;     // per depth: earlier depths whose image must be smaller (symmetry breaking)
    std::vector<std::vector<int>> above;     // per depth: earlier depths whose image must be larger
    std::vector<int> m;                      // image at each depth
    std::unordered_set<std::vector<int>, VecHash> found;
    std::vector<int> scratch;
    bool find_all;
    size_t threshold;
    uint64_t mappings = 0, states = 0;
    std::vector<uint64_t> by_depth;          // search states per depth (measurement only)
    std::vector<std::vector<int>>* record = nullptr;   // automorphism search: every mapping, by pattern vertex
    size_t cap = 0;
    bool overflow = false;

    Search(const GraphCache& cache, bool fa, int th) : g(cache), find_all(fa), threshold(th > 0 ? th : 0) {}

    inline bool subset(int d, int v) const {
        const auto& a = psig[d];
        const uint64_t* b = g.sig.data() + g.sig_off[v];
        const uint64_t* be = g.sig.data() + g.sig_off[v + 1];
        for (uint64_t c : a) {
            while (b != be && *b < c) ++b;
            if (b == be || *b != c) return false;
            ++b;
        }
        return true;
    }

    // true = stop
    bool go(int d) {
        states++;
        by_depth[d]++;
        if (d == P) {
            mappings++;
            if (record) {
                std::vector<int> phi(P);
                for (int k = 0; k < P; k++) phi[order[k]] = m[k];
                record->push_back(std::move(phi));
                if (record->size() > cap) { overflow = true; return true; }
                return false;
            }
            scratch.assign(m.begin(), m.end());
            std::sort(scratch.begin(), scratch.end());
            found.insert(scratch);
            if (!find_all) return true;
            return threshold && found.size() >= threshold;
        }
        const int L = plab[d], D = pdeg[d], T = ptri[d];
        const uint64_t M = pmask[d];
        const auto& bk = back[d];
        const int* pool;
        const int* pend;
        int skip = -1;
        if (bk.empty()) {
            pool = g.label_vertices[L].data();
            pend = pool + g.label_vertices[L].size();
        } else {
            int best = bk[0];
            for (int b : bk) if (g.degrees[m[b]] < g.degrees[m[best]]) best = b;
            skip = best;
            pool = g.adj.data() + g.off[m[best]];
            pend = g.adj.data() + g.off[m[best] + 1];
        }
        for (; pool != pend; ++pool) {
            int v = *pool;
            if (g.lab[v] != L || g.degrees[v] < D || g.tri_count[v] < T || (g.mask[v] & M) != M) continue;
            bool ok = true;
            for (int k = 0; k < d; k++) if (m[k] == v) { ok = false; break; }
            if (!ok) continue;
            for (int b : below[d]) if (m[b] > v) { ok = false; break; }
            if (!ok) continue;
            for (int b : above[d]) if (m[b] < v) { ok = false; break; }
            if (!ok) continue;
            for (int b : bk) if (b != skip && !g.hasEdge(v, m[b])) { ok = false; break; }
            if (!ok || !subset(d, v)) continue;
            m[d] = v;
            if (go(d + 1)) return true;
        }
        return false;
    }
};

struct Result {
    std::vector<std::vector<int>> sets;
    size_t count = 0;
    uint64_t mappings = 0, states = 0, root_pool = 0;
    std::vector<uint64_t> by_depth;
};

// Pattern side of a query: label ids, triangles, signatures, matching order and per-depth lists.  False if a pattern
// label is absent from the target.
static bool setup(Search& s, const GraphCache& cache, const Graph& pattern) {
    int P = pattern.vertex_count;
    s.P = P;
    std::vector<int> plab(P);
    for (int v = 0; v < P; v++) {
        auto it = cache.label_id.find(pattern.labels[v]);
        if (it == cache.label_id.end()) return false;              // a label the target does not have
        plab[v] = it->second;
    }
    std::vector<int> off, adj, ptri, sig_off;
    std::vector<uint64_t> sig, mask;
    to_csr(pattern, off, adj);
    triangles(P, off, adj, plab, cache.label_id.size(), ptri, sig_off, sig, mask);
    auto freq = [&](int v) { return cache.label_vertices[plab[v]].size(); };
    auto deg = [&](int v) { return off[v + 1] - off[v]; };

    // matching order
    std::vector<int> pos(P, -1), nmapped(P, 0);
    s.order.reserve(P);
    for (int d = 0; d < P; d++) {
        int best = -1;
        for (int v = 0; v < P; v++) {
            if (pos[v] != -1) continue;
            if (best == -1) { best = v; continue; }
            bool better;
            if (nmapped[v] != nmapped[best]) better = nmapped[v] > nmapped[best];
            else if (nmapped[v] == 0 && freq(v) != freq(best)) better = freq(v) < freq(best);
            else if (nmapped[v] == 0 && ptri[v] != ptri[best]) better = ptri[v] > ptri[best];
            else if (nmapped[v] > 0 && freq(v) != freq(best)) better = freq(v) < freq(best);
            else better = deg(v) > deg(best);
            if (better) best = v;
        }
        pos[best] = d;
        s.order.push_back(best);
        for (int i = off[best]; i < off[best + 1]; i++) nmapped[adj[i]]++;
    }
    s.plab.resize(P); s.pdeg.resize(P); s.ptri.resize(P); s.psig.resize(P); s.pmask.resize(P);
    s.back.resize(P); s.below.assign(P, {}); s.above.assign(P, {});
    for (int d = 0; d < P; d++) {
        int v = s.order[d];
        s.plab[d] = plab[v];
        s.pdeg[d] = deg(v);
        s.ptri[d] = ptri[v];
        s.psig[d].assign(sig.begin() + sig_off[v], sig.begin() + sig_off[v + 1]);
        s.pmask[d] = mask[v];
        for (int i = off[v]; i < off[v + 1]; i++) if (pos[adj[i]] < d) s.back[d].push_back(pos[adj[i]]);
    }
    s.m.assign(P, -1);
    s.by_depth.assign(P + 1, 0);
    return true;
}

// Symmetry breaking (Grochow & Kellis 2007): conditions image(a) < image(b) such that of every class of mappings that
// differ by a label-preserving automorphism of the pattern exactly one satisfies them.  Such mappings have the same
// node set, so the distinct node sets are unchanged; only repeated mappings are skipped.  The automorphisms are the
// pattern's own matches onto itself (a monomorphism of a graph onto itself is an automorphism).  More than
// AUT_CAP automorphisms: no conditions (still exact, just no saving).
static const size_t AUT_CAP = 20000;

static std::vector<std::pair<int, int>> symmetry_conditions(const Graph& pattern, const std::vector<int>& rank) {
    std::vector<std::pair<int, int>> conds;
    int P = pattern.vertex_count;
    GraphCache self(pattern);
    Search a(self, true, 0);
    std::vector<std::vector<int>> aut;
    a.record = &aut;
    a.cap = AUT_CAP;
    if (!setup(a, self, pattern)) return conds;
    a.go(0);
    if (a.overflow) return conds;
    while (aut.size() > 1) {
        std::vector<std::vector<char>> orbit(P, std::vector<char>(P, 0));
        for (const auto& phi : aut) for (int v = 0; v < P; v++) orbit[v][phi[v]] = 1;
        int best = -1;                     // the earliest vertex in the matching order whose orbit is not itself
        for (int v = 0; v < P; v++) {
            int size = 0;
            for (int u = 0; u < P; u++) size += orbit[v][u];
            if (size > 1 && (best == -1 || rank[v] < rank[best])) best = v;
        }
        if (best == -1) break;
        for (int u = 0; u < P; u++) if (u != best && orbit[best][u]) conds.emplace_back(best, u);
        std::vector<std::vector<int>> stab;
        for (auto& phi : aut) if (phi[best] == best) stab.push_back(std::move(phi));
        aut.swap(stab);
    }
    return conds;
}

// Searches; with want_sets the sorted node sets, otherwise only their number (in result.count).
static Result run(const GraphCache& cache, const Graph& pattern, bool find_all, int threshold, bool want_sets = true) {
    Result result;
    if (pattern.vertex_count == 0) return result;
    Search s(cache, find_all, threshold);
    if (!setup(s, cache, pattern)) return result;
    std::vector<int> pos(pattern.vertex_count);
    for (int d = 0; d < pattern.vertex_count; d++) pos[s.order[d]] = d;
    for (auto [a, b] : symmetry_conditions(pattern, pos)) {          // checked when the later of the two is mapped
        if (pos[a] < pos[b]) s.below[pos[b]].push_back(pos[a]);
        else s.above[pos[a]].push_back(pos[b]);
    }
    s.go(0);
    result.count = s.found.size();
    if (want_sets) {
        result.sets.assign(s.found.begin(), s.found.end());
        std::sort(result.sets.begin(), result.sets.end());
    }
    result.mappings = s.mappings;
    result.states = s.states;
    result.by_depth = s.by_depth;
    result.root_pool = cache.label_vertices[s.plab[0]].size();
    return result;
}

std::pair<int, std::vector<std::vector<int>>> findPatternMatches(
    const GraphCache& graph_cache, const Graph& pattern,
    bool find_all = true, int threshold = 0, py::object edge_dict = py::none()) {
    Result r = run(graph_cache, pattern, find_all, threshold);
    return {static_cast<int>(r.sets.size()), std::move(r.sets)};
}

// Number of distinct node sets only (no node sets built or passed to Python).
int countPatternMatches(const GraphCache& graph_cache, const Graph& pattern) {
    return static_cast<int>(run(graph_cache, pattern, true, 0, false).count);
}

// measuring the search: distinct node sets, mappings found, search states (all, and per depth), root pool size
py::dict searchStats(const GraphCache& graph_cache, const Graph& pattern) {
    Result r = run(graph_cache, pattern, true, 0, false);
    py::dict d;
    d["sets"] = r.count;
    d["mappings"] = r.mappings;
    d["states"] = r.states;
    d["by_depth"] = r.by_depth;
    d["root_pool"] = r.root_pool;
    return d;
}

PYBIND11_MODULE(graph_pattern_cpp, m) {
    m.doc() = "Optimized C++ graph pattern matching (2026-10-01 version)";
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
    m.def("count_pattern_matches", &countPatternMatches, py::arg("graph_cache"), py::arg("pattern"),
          "Number of distinct node sets (find_all, no threshold), without returning them");
    m.def("search_stats", &searchStats, py::arg("graph_cache"), py::arg("pattern"),
          "dict: sets, mappings, states, by_depth, root_pool of one query");
}
