#!/usr/bin/env python3
"""METABRIC significant versus nonsignificant siblings within TRI-4 parents."""
from pathlib import Path
import json
import sys
import tempfile
import argparse
import pandas as pd
import numpy as np
import run_analysis as analysis

engine=analysis.load_module('metabric_same_parent_reference',analysis.REFERENCE/'run_same_parent_comparison.py')
engine.analysis=analysis
engine.engine.analysis=analysis
# Reuse the rank-native export adapter, with METABRIC defaults.
engine.PROJECT_DIR=analysis.PROJECT_DIR

def audit_criterion_eligibility(frame,index,args):
    metrics=engine.engine.intrinsic_metrics(frame)
    for parent in index.loc[index.eligible].itertuples():
        folder=args.outdir/parent.parent_hash
        children=pd.read_csv(folder/'child_motifs.csv')
        group=children[['child_hash','significance_group']].merge(frame.drop(columns='significance_group'),left_on='child_hash',right_on='wl_hash',validate='one_to_one')
        values=group[metrics].to_numpy(float); finite=np.isfinite(values)
        high=group.significance_group.eq('significant').to_numpy()
        n_high=finite[high].sum(axis=0);n_low=finite[~high].sum(axis=0)
        minimum=np.where(finite,values,np.inf).min(axis=0)
        maximum=np.where(finite,values,-np.inf).max(axis=0)
        reason=np.where((n_high<args.min_per_class)|(n_low<args.min_per_class),'insufficient_finite_values',np.where(minimum==maximum,'constant','testable'))
        pd.DataFrame({'metric':metrics,'n_significant':n_high,'n_nonsignificant':n_low,'testability':reason}).to_csv(folder/'criterion_eligibility.csv',index=False)

def main():
    args=engine.parse_args()
    if args.motif_galleries_only or args.clustered_heatmap_only:
        engine.main();return
    frame=pd.read_csv(args.features,dtype={'wl_hash':str})
    with tempfile.TemporaryDirectory(prefix='metabric-sibling-') as tmp:
        compatibility=Path(tmp)/'features.csv.gz'
        analysis.add_compatibility_columns(frame).to_csv(compatibility,index=False)
        settings=argparse.Namespace(**vars(args));settings.features=compatibility
        engine.engine.run(settings)
    engine.rewrite_rank_native_csvs(args.outdir)
    summary=engine.rewrite_documentation(args.outdir,args)
    summary.update(cohort='METABRIC',target_hash=analysis.TARGET_HASH,input_checksums=analysis.provenance(args.trace_dir),reproduction_command=f'python {Path(__file__).relative_to(analysis.REPO)}')
    source=pd.read_csv(args.trace_dir/'trace_nodes.csv');edges=pd.read_csv(args.trace_dir/'trace_edges.csv')
    relations=engine.engine.prepare_relationships(edges)
    parents=source.loc[source.triangle_level.eq(4)].copy()
    index=pd.read_csv(args.outdir/'parent_group_index.csv')
    # Include parents without any TRI-5 child in the complete audit.
    missing=parents.loc[~parents.wl_hash.isin(index.parent_hash)]
    extra=[]
    for row in missing.itertuples():
        extra.append(dict(parent_hash=row.wl_hash,trace_tri5_children=0,analyzable_nonsignificant_children=0,analyzable_significant_children=0,analyzable_total_children=0,mixed_significance_group=False,eligible=False,exclusion_reason='no_tri5_children',parent_node_labels=row.node_labels,parent_cell_type_composition=row.cell_type_composition))
    if extra:index=pd.concat([index,pd.DataFrame(extra)],ignore_index=True).sort_values('parent_hash')
    index['excluded_tri5_children']=index.trace_tri5_children-index.analyzable_total_children
    index.to_csv(args.outdir/'parent_group_index.csv',index=False)
    audit_criterion_eligibility(frame,index,args)
    memberships=relations.merge(source.loc[source.triangle_level.eq(5),['wl_hash','p_value_rank_binary','rank_binary_fit_status']],left_on='child_hash',right_on='wl_hash',validate='many_to_one')
    memberships['included_in_criterion_comparison']=memberships.child_hash.isin(frame.wl_hash)
    memberships.to_csv(args.outdir/'all_parent_child_memberships.csv.gz',index=False)
    summary.update(trace_parents_with_tri5_children=summary['trace_parents'],trace_parents=len(parents),excluded_no_tri5_children=len(missing))
    (args.outdir/'analysis_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    guide=args.outdir/'README.md'
    text=guide.read_text().replace(f'TRI-4 parents in the lineage: {summary["trace_parents_with_tri5_children"]}',f'TRI-4 parents in the lineage: {len(parents)} ({summary["trace_parents_with_tri5_children"]} have TRI-5 children)')
    text+='\nThe complete parent audit includes parents with no TRI-5 children. `all_parent_child_memberships.csv.gz` retains undefined-fit memberships for auditing. METABRIC transitions use child-graph coordinates for the recorded removed vertex/edge; every gallery embedding is checked against that record.\n'
    guide.write_text(text)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
