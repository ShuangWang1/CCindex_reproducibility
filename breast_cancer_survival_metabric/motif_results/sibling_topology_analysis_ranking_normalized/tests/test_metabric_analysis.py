import importlib.util
from pathlib import Path
import unittest
import os
import pytest
import numpy as np
import pandas as pd
import networkx as nx

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('metabric_analysis_test',ROOT/'run_analysis.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class MetabricAnalysisTests(unittest.TestCase):
    def test_nine_type_features_and_deterministic_null(self):
        graph=nx.Graph([(0,1),(1,2),(0,2)])
        nx.set_node_attributes(graph,{0:'6',1:'7',2:'8'},'label')
        first,_=a.engine.extra_features(graph,'test',16,20260916)
        second,_=a.engine.extra_features(graph,'test',16,20260916)
        self.assertEqual(len(first),747)
        self.assertEqual(len(a.engine.EDGE_TYPES),45)
        self.assertEqual(len(a.engine.TRIANGLE_TYPES),165)
        self.assertEqual(first['type_6_t_cells_count'],1)
        self.assertEqual(first['type_7_tumor_cells_count'],1)
        self.assertEqual(first['type_8_vascular_sma+_count'],1)
        self.assertEqual(first['typed_triangle_6_7_8_count'],1)
        for k in first: np.testing.assert_equal(first[k],second[k])

    @pytest.mark.integration
    @unittest.skipUnless(os.environ.get('CCINDEX_RUN_INTEGRATION') == '1', 'Requires saved METABRIC traces')
    def test_selection_counts_and_zero_medians_retained(self):
        source=pd.read_csv(a.TRACE_DIR/'trace_nodes.csv')
        selected=a.select_estimable_motifs(source,5,.05)
        self.assertEqual(len(selected),2082)
        self.assertEqual(int(selected.significant_rank_p.sum()),613)
        self.assertTrue(selected.median_patient_rank_score.eq(0).any())
        selected.loc[selected.index[0],'p_value_rank_binary']=.05
        self.assertFalse(bool(a.select_estimable_motifs(selected,5,.05).set_index('wl_hash').loc[selected.iloc[0].wl_hash,'significant_rank_p']))

    def test_bz_defaults_isolated(self):
        bz=a.load_module('isolated_bz_analysis',a.REFERENCE/'run_analysis.py')
        self.assertEqual(len(bz.engine.LABELS),8)
        self.assertEqual(bz.LABEL_NAMES['6'],'tumor cells')
        self.assertEqual(a.LABEL_NAMES['6'],'T cells')

if __name__=='__main__':unittest.main()
