from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from statsmodels.duration.hazard_regression import PHReg

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from breast_cancer_survival_metabric import rank_methods as methods
from breast_cancer_survival_metabric import analyze_motif_survival as raw
from breast_cancer_survival_metabric.run_rank_normalized_analysis import Pipeline, RelationshipIndex, parent_records, graph_from_encoding
from breast_cancer_survival import generate_tri1_motif_frequencies as tri1


class RankPipelineTests(unittest.TestCase):
    def test_ties_absence_and_full_vocabulary(self):
        counts = sparse.csr_matrix([[0,7,7,2,0,1],[0,0,0,0,0,0]])
        expected = [[0,.9,.9,.6,0,.4],[0,0,0,0,0,0]]
        np.testing.assert_allclose(methods.rank_score_sparse(counts,6).toarray(),expected)
        with self.assertRaises(ValueError): methods.rank_score_sparse(counts,1)

    def test_rank_each_level_before_equal_image_average(self):
        counts=sparse.csr_matrix([[100,10,1,4],[1,2,3,3],[0,8,10,1]])
        samples=pd.DataFrame({'row_index':[0,1,2],'patient_id':['A','A','B']})
        clinical=pd.DataFrame({'METABRIC.ID':['A','B'],'T':[10,20],'Death':[1,0]})
        patients, sums, _, _=raw.prepare_cohort(samples,clinical)
        mean=sparse.diags(1/patients.n_images.to_numpy())@sums
        scores=methods.rank_score_sparse(counts[:,:2],2)
        np.testing.assert_allclose((mean@scores).toarray(),[[.5,.5],[0,1]])
        self.assertFalse(np.allclose((mean@scores).toarray(),methods.rank_score_sparse(sums@counts[:,:2],2).toarray()))

    def test_strict_median_binary_and_continuous_cox(self):
        time=np.array([2,2,5,6,8,8,12,15,18,18],float)
        event=np.array([1,1,0,1,0,1,0,1,1,0])
        scores=np.array([0,.8,.1,.7,.4,.4,.4,.9,.2,.4])
        result, high, z=methods.analyze_rank_survival(scores,time,event)
        np.testing.assert_array_equal(high,scores>np.median(scores))
        self.assertFalse(high[scores==np.median(scores)].any())
        fit=PHReg(time,z[:,None],status=event,ties='efron').fit(disp=0)
        self.assertAlmostEqual(result['p_value_rank_continuous_per_sd'],fit.pvalues[0],places=10)
        vectorized=raw.vectorized_binary_cox(high,time,event)
        self.assertAlmostEqual(result['p_value_rank_binary'],vectorized['p_value'][0],places=8)

    def test_tri1_counts_and_hashes(self):
        frequencies,n=tri1.delaunay_triangle_frequencies(np.array([[0,0],[1,0],[1,1],[0,1]]),np.array(['8','8','8','8']))
        rows=tri1.result_rows(frequencies)
        self.assertEqual(sum(r['matches'] for r in rows),n)
        self.assertEqual(n,2)
        for row in rows:
            self.assertEqual(methods.motif_hash(graph_from_encoding(row['node_labels'],row['edge_list'])),row['wl_hash'])

    def test_reverse_index_matches_forward_growth_including_label_eight(self):
        parent=nx.complete_graph(3)
        nx.set_node_attributes(parent,{0:'2',1:'7',2:'8'},'label')
        ph=methods.motif_hash(parent)
        child=parent.copy(); child.add_node(3,label='8'); child.add_edges_from([(0,3),(1,3)])
        ch=methods.motif_hash(child)
        labels,edges=methods.graph_encoding(child)
        records=parent_records((ch,labels,edges))
        self.assertIn(ph,[r[0] for r in records])
        forward=methods.potential_direct_children(parent,{ch},[str(i) for i in range(9)])
        self.assertEqual([methods.motif_hash(x[2]) for x in forward],[ch])
        with self.assertRaises(ValueError): parent_records(('wrong',labels,edges))

    def test_target_traversal_does_not_include_unrelated_siblings(self):
        obj=object.__new__(Pipeline); obj.a=SimpleNamespace(levels=[1,2,3])
        edges=pd.DataFrame([(1,'a',2,'b'),(1,'a',2,'sibling'),(2,'b',3,'c'),(2,'sibling',3,'other')],
                           columns=['parent_level','parent_hash','child_level','child_hash'])
        expected={2:{'b'},1:{'a'},3:{'c'}}
        self.assertEqual(obj.selected_nodes(edges,2,'b'),expected)
        index=RelationshipIndex(edges)
        self.assertEqual(index.select(2,'b',[1,2,3]),expected)
        observed=index.subset(expected)
        self.assertEqual(set(zip(observed.parent_hash,observed.child_hash)),{('a','b'),('b','c')})

    def test_zero_median_relative_retained_and_undefined_fit_audited(self):
        obj=object.__new__(Pipeline)
        obj.a=SimpleNamespace(skip_plots=False,p_threshold=.05)
        obj.signature='fixture'
        nodes=pd.DataFrame({'triangle_level':[1,2,2],'wl_hash':['target','zero','constant'],
            'lineage_role':['target','descendant','descendant'],'p_value_rank_binary':[.01,.1,np.nan],
            'median_patient_rank_score':[.5,0.,0.],
            'rank_binary_fit_status':['estimated','estimated','not_estimable_no_group_variation']})
        edges=pd.DataFrame([(1,'target',2,'zero'),(1,'target',2,'constant')],
                           columns=['parent_level','parent_hash','child_level','child_hash'])
        target=SimpleNamespace(triangle_level=1,wl_hash='target',p_value_rank_binary=.01,median_patient_rank_score=.5)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            obj.write_trace(folder,nodes,edges,target,expanded=True)
            result=pd.read_csv(folder/'trace_nodes.csv')
            self.assertEqual(len(result),3)
            self.assertTrue(result.rank_result_available.all())
            self.assertTrue(result.loc[result.wl_hash.eq('constant'),'p_value_rank_binary'].isna().all())
            self.assertGreater((folder/'rank_normalized_cox_p_value_by_level.png').stat().st_size,1000)


if __name__=='__main__': unittest.main()
