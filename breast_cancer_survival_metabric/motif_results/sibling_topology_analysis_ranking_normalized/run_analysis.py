#!/usr/bin/env python3
"""METABRIC rank-normalized TRI-5 topology criteria for the 2c9dad37 lineage."""
from pathlib import Path
import importlib.util
import json
import hashlib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
REPO = PROJECT_DIR.parents[2]
REFERENCE = REPO/'breast_cancer_survival/motif_results/sibling_topology_analysis_ranking_normalized'
TARGET_HASH = '2c9dad37a530a40bfd159b028f2d8b4f'
TRACE_DIR = PROJECT_DIR.parent/'significant_mtf_trace_ranking_normalized/tri1'/f'{TARGET_HASH}_new'
TRANSITION_COORDINATES = 'child'

def load_module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ImportError(str(path))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

rank = load_module('metabric_rank_topology_reference',REFERENCE/'run_analysis.py')
# Native annotation vocabulary is source configuration. Study files are read
# when running the analysis, so feature helpers can be imported without data.
LABEL_NAMES = dict(enumerate([
    'B cells', 'Endothelial', 'Fibroblasts', 'Macrophages', 'Myoepithelial',
    'Myofibroblasts', 'T cells', 'Tumor cells', 'Vascular SMA+',
]))
LABEL_NAMES = {str(code): name for code, name in LABEL_NAMES.items()}
LABEL_COLORS=dict(zip(map(str,range(9)),['#4C78A8','#ECA82C','#B279A2','#E45756','#72B7B2','#FF9DA6','#54A24B','#7A5195','#9D755D']))
LABEL_ABBREVIATIONS=dict(zip(map(str,range(9)),['B','End','Fib','Mac','MyoE','MyoF','T','Tum','SMA']))
rank.engine.configure_labels(LABEL_NAMES,LABEL_COLORS,LABEL_ABBREVIATIONS)
rank.LABEL_NAMES=rank.engine.LABEL_NAMES;rank.LABELS=rank.engine.LABELS
rank.PROJECT_DIR=PROJECT_DIR;rank.MOTIF_RESULTS=PROJECT_DIR.parent
rank.TARGET_HASH=TARGET_HASH;rank.TRACE_DIR=TRACE_DIR

def __getattr__(name): return getattr(rank,name)

def validate_label_mapping(path=None):
    path = path or REPO/'breast_cancer/metabric_samples_intra/node_label_cell_type_mapping.csv'
    labels = pd.read_csv(path, dtype=str)
    observed = dict(labels[['node_label', 'cell_type']].itertuples(index=False, name=None))
    if observed != LABEL_NAMES or len(labels) != len(LABEL_NAMES):
        raise ValueError('METABRIC label mapping does not match the native nine-type vocabulary')

def provenance(trace):
    return {name:hashlib.sha256((trace/name).read_bytes()).hexdigest() for name in ['trace_nodes.csv','trace_edges.csv']}

def update_cohort_guide(outdir):
    path=outdir/'tables/README.md'
    text=path.read_text()
    start=text.index('| Code | Cell type |')
    end=text.index('\n### 1. Size',start)
    table='| Code | Cell type |\n|---:|---|\n'+'\n'.join(f'| {code} | {name} |' for code,name in LABEL_NAMES.items())+'\n'
    text=text[:start]+table+text[end:]
    text=text.replace('the 36 unordered cell-type pairs','the 45 unordered cell-type pairs')
    text=text.replace('120 possible unordered triples from eight cell types','165 possible unordered triples from nine cell types')
    text=text.replace('three criteria, producing 360 columns','three criteria, producing 495 columns')
    text=text.replace('each of the eight cell types','each of the nine cell types')
    path.write_text(text)

def main():
    args=rank.parse_args()
    validate_label_mapping()
    rank.main()
    summary_path=args.outdir/'analysis_summary.json'
    summary=json.loads(summary_path.read_text())
    summary.update(cohort='METABRIC',target_hash=TARGET_HASH,input_checksums=provenance(args.trace_dir),cell_type_mapping=LABEL_NAMES,intrinsic_criteria=747,reproduction_command=f'python {Path(__file__).relative_to(REPO)}')
    summary_path.write_text(json.dumps(summary,indent=2)+'\n')
    update_cohort_guide(args.outdir)
    source=pd.read_csv(args.trace_dir/'trace_nodes.csv')
    audit=source.loc[source.triangle_level.eq(5)].copy()
    selected=set(rank.select_estimable_motifs(source,5,args.p_threshold).wl_hash)
    audit['included_in_criterion_comparison']=audit.wl_hash.isin(selected)
    audit['exclusion_reason']=audit.rank_binary_fit_status.where(~audit.included_in_criterion_comparison,'')
    audit.to_csv(args.outdir/'tables/tri5_selection_audit.csv.gz',index=False)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
