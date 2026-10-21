import pandas as pd
import igraph as ig
from itertools import combinations, combinations_with_replacement
from collections import deque, Counter
import random
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from multiprocessing import Manager
from concurrent.futures import as_completed
import networkx as nx
import os
import resource
from datetime import datetime
import argparse
from vf2 import run_vf2
from ast import literal_eval
from run_ccindex_benchmark import run_cc_index

parser = argparse.ArgumentParser(description="tree search")
parser.add_argument("--MaxDepth", type=int, default=6)
parser.add_argument("--SM_method", type=str, default='ccindex', help='subgraph matching method: ccindex or vf2')
parser.add_argument("--threshold", type=int, default=10, help='pattern occurrence number threshold')
parser.add_argument("--SM_stop_early", type=bool, default=False, help='if subgaph matching stops early when meeting the threshold')
parser.add_argument("--LevelStartFrom", type=int, default=0, help='Tree search construct from which level')
parser.add_argument("--GraphQueuePath", type=str, default='', help='if tree search not construct from scratch, set the path to read graph queue')
parser.add_argument(
    "--CppModule",
    choices=("graph_pattern_cpp_reference", "graph_pattern_cpp"),
    default="graph_pattern_cpp",
    help="isolated C++ matcher module to benchmark",
)
parser.add_argument(
    "--ResultDir",
    type=str,
    default="ad_matcher_results_current",
    help="fresh directory for per-level matching-time CSV files",
)


def convert_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours} : {minutes} : {seconds:.2f}"

#@profile
def write_doc(g_queue,depth,directory_name):

    if not os.path.exists(directory_name+str(depth)):
        os.makedirs(directory_name+str(depth))
    data = []

    for i,g in enumerate(g_queue):
        if depth == g['trinum']:
            g.vs['id'] = range(g.vcount())
            # Create a dictionary for each graph with the necessary information
            g['extendableedge'] = str(g['extendableedge'])
            g.write_gml(directory_name+str(depth)+f"/{i}.gml")
            g['extendableedge'] = literal_eval(g['extendableedge'])


def tree_search(graph, next_level_graph_dict):
    edge_pool = edge_dict
    edge_list = graph['extendableedge']
    node_labels = graph.vs["label"]
    edge_mtfs = [f'{min(int(node_labels[u]), int(node_labels[v]))}&{max(int(node_labels[u]), int(node_labels[v]))}' for u, v in edge_list]

    ##########check with no node adding, there are still new graph
    done_list = []
    for sp_e in edge_list:
        if sp_e in done_list:
            continue

        node_i, node_j = sp_e
        neigh = [tuple(sorted([i,node_i])) for i in graph.neighbors(node_i, mode="all") if i != node_j and graph.are_adjacent(i, node_j) == False]

        for tmp in [tuple(sorted([i,node_j])) for i in graph.neighbors(node_j, mode = 'all') if i != node_i and graph.are_adjacent(i, node_i) == False]:
            neigh.append(tmp)
        neigh = set(neigh).intersection(edge_list)
        for n in neigh:
            e_graph_2 = graph.copy()

            done_list.append(n)
            edge = list((set(n) - set(sp_e)).union(set(sp_e) - set(n)))
            e_graph_2.add_edges([(edge[0], edge[1])])
            e_graph_2['trinum'] += 1


            #e2_edge_list = e_graph_2['extendableedge']
            e2_edge_list = edge_list.copy() #super important

            tmp_node = list(set(n).intersection(sp_e))[0]
            e2_edge_list.append(tuple(sorted((edge[0], edge[1]))))
            e2_edge_list.remove(tuple(sorted((tmp_node, edge[0]))))
            e2_edge_list.remove(tuple(sorted((tmp_node, edge[1]))))
            e_graph_2['extendableedge'] = e2_edge_list

            if len(e_graph_2.cliques(min=3, max=3)) > len(graph.cliques(min=3, max=3)) + 1:
                continue

            g_nx = convert_to_nx(e_graph_2)

            e_graph_2['hashvalue'] = nx.weisfeiler_lehman_graph_hash(g_nx,node_attr="label")

            if e_graph_2['hashvalue'] not in next_level_graph_dict.keys():
                next_level_graph_dict[e_graph_2['hashvalue']] = e_graph_2

    for e_m, e_i in zip(edge_mtfs,edge_list):

        for new_tri in edge_pool[e_m]:
            nodes = new_tri.split('&')
            new_edges = [f'{nodes[i]}&{nodes[j]}' for i,j in [(0,1),(0,2),(1,2)]]
            # construct an initial graph
            if e_m not in new_edges:
                continue
            e_graph = graph.copy()

            mtf_nodes = e_m.split('&')
            counter1 = Counter(nodes)
            counter2 = Counter(mtf_nodes)

            insert_node = list((counter1 - counter2).elements())[0]

            new_index = e_graph.vcount()
            e_graph.add_vertex()
            e_graph.vs[new_index]["label"] = insert_node
            e_graph.add_edges([(e_i[0], new_index), (e_i[1], new_index)])
            e_graph['trinum'] += 1

            new_edge_list = edge_list.copy()
            new_edge_list.remove(e_i)
            new_edge_list.append((e_i[0], new_index))
            new_edge_list.append((e_i[1], new_index))
            e_graph['extendableedge'] = new_edge_list

            g_nx = convert_to_nx(e_graph)
            e_graph['hashvalue'] = nx.weisfeiler_lehman_graph_hash(g_nx,node_attr="label")

            if e_graph['hashvalue'] not in next_level_graph_dict:
                next_level_graph_dict[e_graph['hashvalue']] = e_graph

    #gc.collect()
    return next_level_graph_dict

import hashlib
from hashlib import blake2b

def _hash_label(label, digest_size):
    return blake2b(label.encode("ascii"), digest_size=digest_size).hexdigest()

def _init_node_labels(G, edge_attr, node_attr):
    if node_attr:
        return {u: str(dd[node_attr]) for u, dd in G.nodes(data=True)}
    elif edge_attr:
        return {u: "" for u in G}
    else:
        return {u: str(deg) for u, deg in G.degree()}

def _neighborhood_aggregate(G, node, node_labels, edge_attr=None):
    """
    Compute new labels for given node by aggregating
    the labels of each node's neighbors.
    """
    label_list = []
    for nbr in G.neighbors(node):
        prefix = "" if edge_attr is None else str(G[node][nbr][edge_attr])
        label_list.append(prefix + node_labels[nbr])
    return node_labels[node] + "".join(sorted(label_list))

def weisfeiler_lehman_graph_hash(
    G, edge_attr=None, node_attr=None, iterations=3, digest_size=16
):
    def weisfeiler_lehman_step(G, labels, edge_attr=None):
        """
        Apply neighborhood aggregation to each node
        in the graph.
        Computes a dictionary with labels for each node.
        """
        new_labels = {}
        for node in G.nodes():
            label = _neighborhood_aggregate(G, node, labels, edge_attr=edge_attr)
            new_labels[node] = _hash_label(label, digest_size)
        return new_labels

    # set initial node labels
    node_labels = _init_node_labels(G, edge_attr, node_attr)

    subgraph_hash_counts = []
    for _ in range(iterations):
        node_labels = weisfeiler_lehman_step(G, node_labels, edge_attr=edge_attr)
        counter = Counter(node_labels.values())
        # sort the counter, extend total counts
        subgraph_hash_counts.extend(sorted(counter.items(), key=lambda x: x[0]))

    # hash the final counter
    return _hash_label(str(tuple(subgraph_hash_counts)), digest_size)

def convert_to_nx(g):
    edge_list = g.get_edgelist()
    vertex_names = g.vs["label"]
    vertex_idx = range(len(vertex_names))

    # Create a new networkx graph
    g_nx_graph = nx.Graph()

    # Add edges to the networkx graph
    for idx, name in zip(vertex_idx, vertex_names):
        g_nx_graph.add_node(idx, label=name)
    for e in edge_list:
        g_nx_graph.add_edge(e[0], e[1])
    return g_nx_graph

def process_g_list(i, g1_list,g2_list):
    flag_list = []
    for g1 in g1_list:
        flag = True
        for g2 in g2_list:
            c1 = [node_dict[i] for i in g1.vs["label"]]
            c2 = [node_dict[i] for i in g2.vs["label"]]

            g1_nx = convert_to_nx(g1)
            g2_nx = convert_to_nx(g2)
            if nx.weisfeiler_lehman_graph_hash(g1_nx,node_attr="label") == nx.weisfeiler_lehman_graph_hash(g2_nx,node_attr="label"):
                flag = False
                break
        flag_list.append(flag)
    combine_2_g = g2_list
    for g,f in zip(g1_list,flag_list):
        if f:
            combine_2_g.append(g)
    return combine_2_g

def construct_queue(filter_tri_dict):
    g_queue = deque()
    for mtf in filter_tri_dict.keys():
        tri_dict = filter_tri_dict
        tri_dict = {k:int(v[0]) for k,v in tri_dict.items()}
        tri_dict[mtf] -= 1

        nodes = mtf.split('&')
        g = ig.Graph()
        g.add_vertices(3)
        g.vs["label"] = nodes
        g.add_edges([(0,1),(0,2),(1,2)])
        g['trinum'] = 1
        g['extendableedge'] = g.get_edgelist()

        g_queue.append(g)
    return g_queue

def read_queue(graph_path, level):
    g_queue = []

    path = f'{graph_path}/{level}'
    for f in os.listdir(f'{graph_path}/{level}'):
        gir_dir = f'{path}/{f}'
        graph = ig.read(gir_dir)
        graph['extendableedge'] = literal_eval(graph['extendableedge'])
        g_queue.append(graph)

    return g_queue

args = parser.parse_args()

df = pd.read_csv("ad_data.csv")

nodes = [mtf.split('&') for mtf in df.columns.to_list()]
nodes = sorted(set([i for sublist in nodes for i in sublist]))
node_dict = {k:v for v,k in enumerate(nodes)}



##########Setting threshold for motif occurrence numbers

threshold = args.threshold
ori_tri_dict = df.to_dict(orient="list")
filter_tri_dict = ori_tri_dict.copy()

directory_name = os.path.join(args.ResultDir, 'graphs') + os.sep
if not os.path.exists(directory_name):
    os.makedirs(directory_name)
for k,v in ori_tri_dict.items():
    if v[0] <= threshold:
        del filter_tri_dict[k]
cts = []
for k in ori_tri_dict.keys():
    for i in k.split('&'):
        cts.append(i)

cts = set(cts)

ct_idx_mapping = {k:v for v,k in enumerate(sorted(cts))}
edge_dict = {'&'.join(map(str, sorted(pair))): [] for pair in combinations_with_replacement(ct_idx_mapping.values(), 2)}
ct_idx_mapping = {k:str(v) for k,v in ct_idx_mapping.items()}
#edge_dict = {'&'.join(sorted(pair)):[] for pair in combinations_with_replacement(cts,2)}
filter_tri_idx_dict = {}
for k,v in filter_tri_dict.items():
    cells = k.split('&')
    idx_list = [ct_idx_mapping[part] for part in cells]
    new_key = '&'.join(idx_list)
    filter_tri_idx_dict[new_key] = v

filter_tri_dict = filter_tri_idx_dict

for e_nodes in edge_dict.keys():
    e_i, e_j = e_nodes.split('&')
    for tri_nodes in filter_tri_dict.keys():
        if e_i != e_j:
            if e_i in tri_nodes.split('&') and e_j in tri_nodes.split('&'):
                edge_dict[e_nodes].append(tri_nodes)
        else:
            if e_nodes in tri_nodes:
                edge_dict[e_nodes].append(tri_nodes)

edge_dict_copy = edge_dict.copy()
for k,v in edge_dict_copy.items():
    if len(v)==0:
        del edge_dict[k]

initial_time = time.time()
def main():

    max_depth = args.MaxDepth
    level_start = args.LevelStartFrom
    sm_method = args.SM_method
    stop_early = args.SM_stop_early
    graph_path = args.GraphQueuePath

    if level_start == 0:
        g_queue = construct_queue(filter_tri_dict)
    else:
        g_queue = read_queue(graph_path,level_start)

    gir_dir = 'spatial_13months-disease-replicate_1.gml'
    graph = ig.read(gir_dir)

    now = datetime.now()
    print(now)

    print(f"Target node size {len(graph.vs['label'])}    Node Type size {len(set(graph.vs['label']))}")

    graph.vs['label'] = [str(int(x)) for x in graph.vs['label']]
    for i in range(level_start, max_depth-1):

        start_time = time.time()

        #g_next_level_list = []
        next_level_graph_dict = {}
        for g in g_queue:
            next_level_graph_dict = tree_search(g,next_level_graph_dict)
            #g_next_level_list.append(g_next_level)
        time_point = time.time()
        time_elapse = convert_time(time_point - initial_time)
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f'{time_elapse}  trinum-{i+2} tree searching: {convert_time(time_point - start_time)}')
        print(f"Memory consumption : {mem:.2f} MB")

        g_queue = list(next_level_graph_dict.values())
        #print("length of the graph before pruning: ",len(g_queue))

        ###################using cc-index method to compute how many times it occur in real data

        start_time = time.time()

        result_path = os.path.join(args.ResultDir, f'{sm_method}_tri_{i+2}.csv')
        print(result_path)
        result_dir = os.path.dirname(result_path)
        os.makedirs(result_dir, exist_ok=True)
        if os.path.exists(result_path):
            raise FileExistsError(
                f"Refusing to overwrite benchmark result: {result_path}"
            )

        use_multiprocessing = False
        cpu_num = 1

        if sm_method == 'ccindex':
            graph_name = f'ad_graph_{i+2}'

            result = run_cc_index(
                graph, g_queue, result_path, graph_name,
                use_multiprocessing=False, perform_cpp=True,
                cpp_module_name=args.CppModule)
            # match_count = []
            # for r in result:
            #     match_count.append(r['matchCounts'])
            # for g,c in zip(g_queue, match_count):
            #     g['count'] = c
        else:
            g_queue = run_vf2(graph, g_queue, result_path, use_multiprocessing, cpu_num)


        #g_queue = process_sequential(graph_cache, g_queue)
        time_point = time.time()
        time_elapse = convert_time(time_point - initial_time)
        print(f'{time_elapse}  searching in all candidates: {convert_time(time_point - start_time)}')

        update_g_queue = []
        for g in g_queue:
            if g['count'] >= threshold or g['count'] == -1:
                update_g_queue.append(g)
        g_queue = update_g_queue

        print("length of the graph after pruning: ",len(g_queue))
        print('#'*50)
        print('#'*50)
        #write_doc(g_queue,i+2,directory_name)



if __name__ == "__main__":
    main()
