#!/usr/bin/env python3
"""Audit source membership, statistics, graph mappings and generated artifacts."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import networkx as nx
from PIL import Image
from scipy.stats import mannwhitneyu
import run_same_parent_comparison as sibling

a=sibling.analysis
OUT=a.PROJECT_DIR/'results'

def main():
    parent_out=OUT/'same_parent_comparsion'
    summary=json.loads((parent_out/'analysis_summary.json').read_text())
    assert summary['status']=='complete' and summary['eligible_parents']==86
    assert summary['input_checksums']==a.provenance(a.TRACE_DIR)
    features=pd.read_csv(OUT/'tables/tri5_motif_criteria.csv.gz')
    nodes=pd.read_csv(a.TRACE_DIR/'trace_nodes.csv')
    selected=a.select_estimable_motifs(nodes,5,.05)
    assert set(features.wl_hash)==set(selected.wl_hash) and len(features)==2082
    assert int(features.significant_rank_p.sum())==613
    assert features.significance_group.eq('nonsignificant').sum()==1469
    metrics=sibling.engine.engine.intrinsic_metrics(features)
    assert len(metrics)==747
    assert not any(k.startswith('dispersion_') or k in a.METADATA for k in metrics)
    index=pd.read_csv(parent_out/'parent_group_index.csv')
    assert len(index)==1443 and int(index.eligible.sum())==86
    edges=pd.read_csv(a.TRACE_DIR/'trace_edges.csv')
    relations=sibling.engine.engine.prepare_relationships(edges)
    assert len(relations)==7610 and relations.parent_hash.nunique()==1078
    expected,_=sibling.engine.engine.build_parent_index(relations,features,3)
    actual=index.set_index('parent_hash')
    for row in expected.itertuples():
        for key in ['trace_tri5_children','analyzable_nonsignificant_children','analyzable_significant_children','eligible']:
            assert getattr(row,key)==actual.loc[row.parent_hash,key]
    sibling.audit_criterion_eligibility(features,index,argparse.Namespace(outdir=parent_out,min_per_class=3))
    graph_rows=nodes[nodes.triangle_level.isin([4,5])].set_index('wl_hash')
    graphs={}
    for h,row in graph_rows.iterrows():
        g=a.base.graph_from_encoding(row.node_labels,row.edge_list,h)
        assert a.base.triangle_count(g)==row.triangle_level
        graphs[h]=g
    for row in relations.itertuples():
        reversed_child=graphs[row.child_hash].copy()
        if row.growth_operation=='add_vertex':
            reversed_child.remove_node(int(row.changed_node_or_edge))
        else:
            assert row.growth_operation=='add_edge'
            reversed_child.remove_edge(*map(int,row.changed_node_or_edge.split('-')))
        assert nx.is_isomorphic(reversed_child,graphs[row.parent_hash],node_match=lambda x,y:x['label']==y['label'])
    gallery=pd.read_csv(parent_out/'motif_gallery_index.csv')
    operations={}
    rel_lookup=relations.set_index(['parent_hash','child_hash'])
    for row in gallery.itertuples():
        mapping={int(k):int(v) for k,v in json.loads(row.parent_to_child_mapping).items()}
        parent=graphs[row.parent_hash];child=graphs[row.child_hash]
        assert set(mapping)==set(parent.nodes) and len(set(mapping.values()))==len(parent)
        assert all(parent.nodes[p]['label']==child.nodes[c]['label'] for p,c in mapping.items())
        embedded={tuple(sorted((mapping[u],mapping[v]))) for u,v in parent.edges}
        child_edges={tuple(sorted(e)) for e in child.edges}
        assert embedded<=child_edges
        added_nodes=set(json.loads(row.added_nodes))
        added_edges={tuple(x) for x in json.loads(row.added_edges)}
        assert set(child.nodes)-set(mapping.values())==added_nodes
        assert child_edges-embedded==added_edges
        rel=rel_lookup.loc[(row.parent_hash,row.child_hash)]
        op=rel.growth_operation;operations[op]=operations.get(op,0)+1
        if op=='add_vertex':
            assert added_nodes=={int(rel.changed_node_or_edge)} and len(added_edges)==2
            assert all(u in added_nodes or v in added_nodes for u,v in added_edges)
        else:
            assert op=='add_edge' and not added_nodes
            assert added_edges=={tuple(sorted(map(int,rel.changed_node_or_edge.split('-'))))}
    combined=pd.read_csv(parent_out/'all_parent_criteria_comparisons.csv.gz')
    indexed=features.set_index('wl_hash')
    for parent in index.loc[index.eligible].itertuples():
        folder=parent_out/parent.parent_hash
        children=pd.read_csv(folder/'child_motifs.csv')
        c=pd.read_csv(folder/'criteria_comparisons.csv')
        eligible=pd.read_csv(folder/'criterion_eligibility.csv')
        assert set(c.metric)==set(eligible.loc[eligible.testability.eq('testable'),'metric'])
        np.testing.assert_allclose(c.location_q_bh,a.base.benjamini_hochberg(c.location_p_value),rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(c.dispersion_q_bh,a.base.benjamini_hochberg(c.dispersion_p_value),equal_nan=True,rtol=1e-10,atol=1e-12)
        order=c.sort_values(['location_q_bh','absolute_cliffs_delta','metric'],ascending=[True,False,True]).metric.tolist()
        assert c.metric.tolist()==order
        assert pd.read_csv(folder/'top_criteria.csv').metric.tolist()==order[:24]
        assert set(children.child_hash)<=set(selected.wl_hash)
        assert len(children)==parent.analyzable_total_children
        assert set(c.metric)<=set(metrics)
        # Independently recompute the leading location test and effect size.
        first=c.iloc[0];metric=first.metric
        group=indexed.loc[children.child_hash]
        high=group.loc[group.significant_rank_p,metric].replace([np.inf,-np.inf],np.nan).dropna()
        low=group.loc[~group.significant_rank_p,metric].replace([np.inf,-np.inf],np.nan).dropna()
        result=mannwhitneyu(high,low,alternative='two-sided',method='asymptotic')
        np.testing.assert_allclose(result.pvalue,first.location_p_value)
        np.testing.assert_allclose(2*result.statistic/(len(high)*len(low))-1,first.cliffs_delta)
        assert len(combined.loc[combined.parent_hash.eq(parent.parent_hash)])==len(c)
    matrix=pd.read_csv(parent_out/'clustered_location_bh_q_matrix.csv.gz',index_col=0)
    assert set(matrix.columns)==set(index.loc[index.eligible,'parent_hash'])
    for row in combined.itertuples():
        np.testing.assert_allclose(matrix.loc[row.metric,row.parent_hash],row.location_q_bh)
    images=list(OUT.rglob('*.png'))
    for path in images:
        with Image.open(path) as im:im.verify()
    assert len(list(parent_out.glob('*/tri5_child_motif_gallery.png')))==86
    result={'status':'passed','motifs_analyzed':2082,'significant_motifs':613,'nonsignificant_motifs':1469,'undefined_excluded':1866,'intrinsic_criteria':747,'all_tri4_parents':1443,'eligible_parents':86,'parent_criterion_tests':len(combined),'validated_graph_encodings':len(graphs),'validated_source_transitions':len(relations),'validated_gallery_embeddings':len(gallery),'growth_operation_counts':operations,'verified_pngs':len(images),'input_checksums':a.provenance(a.TRACE_DIR),'checks':['exact source membership and class counts','complete parent audit and multi-parent memberships','criterion testability and no outcome metadata','within-parent BH correction and sorting','independent leading Mann-Whitney and effect sizes','cell-preserving graph embeddings and recorded growth operations','exported q-matrix agrees with all criterion tests','all PNG images verified']}
    (OUT/'validation_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    recurrence=pd.read_csv(parent_out/'top_criteria_recurrence.csv')
    findings=recurrence.head(10)[['display_name','parents_in_top_criteria','parents_bh_q_lt_0_05','median_absolute_cliffs_delta_when_top']]
    findings.to_csv(parent_out/'leading_criteria_summary.csv',index=False)
    significant=combined.loc[combined.location_q_bh.lt(.05)]
    lines=[
        '# METABRIC TRI-5 sibling comparison results','',
        'TRI-1 lineage: `'+a.TARGET_HASH+'`. All counts below use the existing rank-normalized binary Cox results; survival models were not refitted.','',
        '- TRI-5 motifs: 3,948; analyzed: 2,082 (613 significant and 1,469 nonsignificant); undefined fits excluded: 1,866.',
        '- TRI-4 parents: 1,443, including 1,078 with TRI-5 children; 270 have both significance classes; 86 meet the minimum of three motifs per class.',
        f'- Intrinsic criteria: 747; testable parent–criterion comparisons: {len(combined):,}.',
        f'- Criteria with within-parent location BH q < 0.05: {len(significant):,} tests across {significant.parent_hash.nunique()} parents and {significant.metric.nunique()} distinct criteria.','',
        '## Recurring leading criteria','',
        'Recurrence counts appearances among each parent’s top 24 criteria. Rankings use within-parent location BH q, then absolute Cliff’s delta. A top-24 appearance does not itself imply q < 0.05.','',
        '| Criterion | Parents in top 24 | Parents with q < 0.05 | Median absolute delta when top |',
        '|---|---:|---:|---:|',
    ]
    for row in findings.itertuples():
        lines.append(f'| {row.display_name} | {row.parents_in_top_criteria} | {row.parents_bh_q_lt_0_05} | {row.median_absolute_cliffs_delta_when_top:.3f} |')
    lines += ['', '## Files and interpretation','',
        '- [Parent comparisons and figures](same_parent_comparsion/README.md)',
        '- [All parent–criterion results](same_parent_comparsion/all_parent_criteria_comparisons.csv.gz)',
        '- [Criterion recurrence](same_parent_comparsion/top_criteria_recurrence.csv)',
        '- [Parent eligibility audit](same_parent_comparsion/parent_group_index.csv)',
        '- [Validation](validation_summary.json)','',
        'Positive Cliff’s delta means higher criterion values among significant siblings; negative means lower. These are descriptive, within-lineage comparisons, not held-out predictive performance. BH correction is separate within each parent, and children may share parents and patients. Geometry requiring tissue coordinates is unavailable.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
