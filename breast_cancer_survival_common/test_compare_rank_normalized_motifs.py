import unittest
import networkx as nx
import numpy as np
import pandas as pd
from compare_all_observed_motifs import attach_original_screen
from compare_rank_normalized_motifs import decode_graph, build_groups


class CommonMotifTests(unittest.TestCase):
    def records(self, a, b):
        return [{'triangle_level': 1, 'cohort': 'bz', 'graph': a},
                {'triangle_level': 1, 'cohort': 'metabric', 'graph': b}]

    def test_shared_names_match_despite_different_numeric_codes(self):
        bz=decode_graph('6&6&3','0-1;0-2;1-2',{'6':'tumor cells','3':'endothelial'})
        meta=decode_graph('7&1&7','0-1;0-2;1-2',{'7':'Tumor cells','1':'Endothelial'})
        self.assertEqual(len(build_groups(self.records(bz,meta))),1)

    def test_identical_numeric_codes_do_not_override_different_cell_types(self):
        bz=decode_graph('6&6&3','0-1;0-2;1-2',{'6':'tumor cells','3':'endothelial'})
        meta=decode_graph('6&6&3','0-1;0-2;1-2',{'6':'T cells','3':'Macrophages'})
        self.assertEqual(build_groups(self.records(bz,meta)),[])

    def test_stromal_types_require_explicit_aliases(self):
        bz=decode_graph('5&5&6','0-1;0-2;1-2',{'5':'stromal cells','6':'tumor cells'})
        mapping={'2':'Fibroblasts','7':'Tumor cells'}
        meta=decode_graph('2&2&7','0-1;0-2;1-2',mapping)
        self.assertEqual(build_groups(self.records(bz,meta)),[])
        meta=decode_graph('2&2&7','0-1;0-2;1-2',mapping,aliases={'fibroblasts':'stromal cells'})
        self.assertEqual(len(build_groups(self.records(bz,meta))),1)

    def test_exact_isomorphism_rejects_wl_collision(self):
        cycle=nx.cycle_graph(6)
        triangles=nx.disjoint_union(nx.complete_graph(3),nx.complete_graph(3))
        for graph in [cycle,triangles]: nx.set_node_attributes(graph,'same','label')
        self.assertEqual(nx.weisfeiler_lehman_graph_hash(cycle,node_attr='label'),nx.weisfeiler_lehman_graph_hash(triangles,node_attr='label'))
        self.assertEqual(build_groups(self.records(cycle,triangles)),[])

    def test_primary_screen_membership_includes_undefined_p_values(self):
        frame=pd.DataFrame({'triangle_level':[1,1,1],'wl_hash':['estimated','undefined','outside']})
        primary=pd.DataFrame({'triangle_level':[1,1],'wl_hash':['estimated','undefined'],'p_value_rank_binary':[.2,np.nan]})
        result=attach_original_screen(frame,primary)
        self.assertEqual(result.in_original_positive_median_rank_screen.tolist(),[True,True,False])
        self.assertTrue(pd.isna(result.loc[1,'original_p_value_rank_binary']))

    def test_topology_only_ignores_cell_types(self):
        a=decode_graph('0&0&0','0-1;0-2;1-2',{'0':'B cells'},mode='topology-only')
        b=decode_graph('7&7&7','0-1;0-2;1-2',{'7':'Tumor cells'},mode='topology-only')
        self.assertEqual(len(build_groups(self.records(a,b))),1)

if __name__=='__main__': unittest.main()
