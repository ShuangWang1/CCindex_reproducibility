#!/usr/bin/env python3
"""Publish shared motifs with nominal rank-normalized Cox p < .05 in both cohorts."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('MPLCONFIGDIR','/tmp/common-significant-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import networkx as nx
import numpy as np
import pandas as pd

from compare_all_observed_motifs import label_maps, decode_graph, COLORS, format_value

HERE=Path(__file__).resolve().parent


def select_common_significant(frame, threshold=.05):
    """No median, original-screen-membership, effect-direction or BH filter."""
    bz=pd.to_numeric(frame.bz_p_value_rank_binary, errors='coerce')
    meta=pd.to_numeric(frame.metabric_p_value_rank_binary, errors='coerce')
    result=frame.loc[np.isfinite(bz)&np.isfinite(meta)&bz.lt(threshold)&meta.lt(threshold)].copy()
    if {'bz_hazard_ratio_rank_binary','metabric_hazard_ratio_rank_binary'}.issubset(result.columns):
        result['same_hazard_direction']=result.bz_hazard_ratio_rank_binary.gt(1).eq(result.metabric_hazard_ratio_rank_binary.gt(1))
    return result


def safe(value):
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    if isinstance(value,np.generic):value=value.item()
    return None if pd.isna(value) else value


def write_json(data,path):
    path.write_text(json.dumps(safe(data),indent=2,allow_nan=False)+'\n')


def draw_pair(row,maps,folder,pdf):
    a=decode_graph(row.bz_node_labels,row.bz_edge_list,maps['bz'])
    b=decode_graph(row.metabric_node_labels,row.metabric_edge_list,maps['metabric'])
    matcher=nx.algorithms.isomorphism.GraphMatcher(b,a,node_match=lambda x,y:x['label']==y['label'])
    if not matcher.is_isomorphic():raise ValueError('Nonmatching labeled graphs in selected pairs')
    ref=nx.planar_layout(a) if nx.check_planarity(a)[0] else nx.spring_layout(a,seed=0)
    fig,axes=plt.subplots(1,2,figsize=(14,7.4))
    for ax,(cohort,graph) in zip(axes,[('bz',a),('metabric',b)]):
        positions={n:ref[n if cohort=='bz' else matcher.mapping[n]] for n in graph}
        nx.draw_networkx(graph,pos=positions,ax=ax,with_labels=True,font_size=9,font_color='white',node_size=700,
                         node_color=[COLORS[d['label']] for _,d in graph.nodes(data=True)],
                         edgecolors='white',linewidths=1.8,edge_color='#666666',width=2)
        p=getattr(row,cohort+'_p_value_rank_binary');hr=getattr(row,cohort+'_hazard_ratio_rank_binary')
        lo=getattr(row,cohort+'_ci95_low_rank_binary');hi=getattr(row,cohort+'_ci95_high_rank_binary')
        med=getattr(row,cohort+'_median_patient_rank_score')
        rank=getattr(row,cohort+'_rank_within_common_significant_level')
        heading=(f'{cohort.upper()} | {getattr(row,cohort+"_wl_hash")}\n'
                 f'Cox p = {p:.5g}; HR = {hr:.3g} (95% CI {lo:.3g}–{hi:.3g})\n'
                 f'Median score = {med:.5g}; rank among selected TRI motifs = {int(rank)}')
        ax.set_title(heading,fontsize=10,pad=16)
        ax.text(.5,-.12,f'n = {int(getattr(row,cohort+"_n_survival"))}; deaths = {int(getattr(row,cohort+"_events_survival"))}\n'
                f'High rank: {int(getattr(row,cohort+"_n_high_rank"))}; low rank: {int(getattr(row,cohort+"_n_low_rank"))}',
                transform=ax.transAxes,ha='center',fontsize=9)
        ax.set_axis_off()
    names=sorted({d['label'] for _,d in a.nodes(data=True)})
    handles=[plt.Line2D([0],[0],marker='o',linestyle='none',color=COLORS[name],label=name,markersize=9) for name in names]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.09),ncol=len(names),frameon=False)
    direction='Same hazard direction' if row.same_hazard_direction else 'OPPOSITE hazard directions'
    fig.suptitle(f'TRI-{row.triangle_level} | common significant motif {int(row.pair_number):03d}\n'
                 'Nominal rank-normalized Cox p < 0.05 in both datasets; no median restriction',fontsize=13,y=.98)
    fig.text(.5,.025,direction,ha='center',fontsize=10,color='#3568A8' if row.same_hazard_direction else '#A33A35')
    fig.tight_layout(rect=(0,.19,1,.89))
    fig.savefig(folder/'paired_topology.png',dpi=150)
    fig.savefig(folder/'paired_topology.svg')
    pdf.savefig(fig)
    plt.close(fig)
    return a,b


def graph_metadata(graph):
    return {'nodes':[{'node_id':n,**d} for n,d in graph.nodes(data=True)],
            'edges':[list(e) for e in graph.edges],
            'cell_type_counts':dict(sorted(Counter(d['label'] for _,d in graph.nodes(data=True)).items()))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=HERE/'all_motifs/common_motif_pairs.csv.gz')
    parser.add_argument('--outdir',type=Path,default=HERE/'common_significant')
    args=parser.parse_args();out=args.outdir.resolve();out.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(args.input,dtype={'bz_wl_hash':str,'metabric_wl_hash':str})
    selected=select_common_significant(data)
    if selected.duplicated(['triangle_level','bz_wl_hash','metabric_wl_hash']).any():raise ValueError('Duplicate motif pairs')
    selected['max_cohort_p_value']=selected[['bz_p_value_rank_binary','metabric_p_value_rank_binary']].max(axis=1)
    selected=selected.sort_values(['triangle_level','max_cohort_p_value','bz_wl_hash'],kind='stable').reset_index(drop=True)
    selected['pair_number']=np.arange(1,len(selected)+1)
    for cohort in ['bz','metabric']:
        selected[cohort+'_rank_within_common_significant_global']=selected[cohort+'_p_value_rank_binary'].rank(method='min').astype('Int64')
        selected[cohort+'_rank_within_common_significant_level']=selected.groupby('triangle_level')[cohort+'_p_value_rank_binary'].rank(method='min').astype('Int64')
        selected[cohort+'_hazard_direction']=np.where(selected[cohort+'_hazard_ratio_rank_binary'].gt(1),'higher hazard','lower hazard')
    selected['source_all_motifs_visualization']=[str((args.input.parent/p).resolve()) for p in selected.visualization]
    selected['motif_directory']=[f'tri{int(r.triangle_level)}/{r.common_motif_id}' for r in selected.itertuples()]
    for field,name in [('visualization','paired_topology.png'),('vector_visualization','paired_topology.svg'),('metadata_file','metadata.json')]:
        selected[field]=selected.motif_directory+'/'+name
    maps=label_maps();members=[]
    with PdfPages(out/'all_common_significant_motifs.pdf') as pdf:
        for i,row in enumerate(selected.itertuples(index=False),1):
            folder=out/row.motif_directory;folder.mkdir(parents=True,exist_ok=True)
            a,b=draw_pair(row,maps,folder,pdf)
            flat=row._asdict()
            nested={'common_motif_id':row.common_motif_id,'triangle_level':int(row.triangle_level),
                    'pair_number':int(row.pair_number),'selection_rule':'finite rank-normalized binary Cox Wald p < 0.05 in BOTH cohorts; no median restriction',
                    'same_hazard_direction':bool(row.same_hazard_direction),'cohorts':{},'all_table_fields':flat}
            for cohort,graph in [('bz',a),('metabric',b)]:
                item={key[len(cohort)+1:]:value for key,value in flat.items() if key.startswith(cohort+'_')}
                item['survival_time_unit']='months' if cohort=='bz' else 'days'
                item['graph']=graph_metadata(graph)
                nested['cohorts'][cohort]=item
                members.append({'common_motif_id':row.common_motif_id,'triangle_level':int(row.triangle_level),
                                'pair_number':int(row.pair_number),'cohort':cohort,**{k:v for k,v in item.items() if k!='graph'},
                                'cell_type_counts':json.dumps(item['graph']['cell_type_counts'],sort_keys=True),
                                'visualization':row.visualization,'metadata_file':row.metadata_file})
            write_json(nested,folder/'metadata.json')
            for cohort in ['bz','metabric']:
                write_json(nested['cohorts'][cohort],folder/f'{cohort}_graph_and_statistics.json')
            if i%25==0 or i==len(selected):print(f'Visualized {i}/{len(selected)} common significant pairs',flush=True)
    selected.to_csv(out/'common_significant_motifs_metadata.csv',index=False)
    selected.to_csv(out/'common_significant_motifs_metadata.csv.gz',index=False,compression='gzip')
    member_frame=pd.DataFrame(members)
    member_frame.to_csv(out/'common_significant_motif_members.csv',index=False)
    counts=[]
    for level in range(1,8):
        group=selected.loc[selected.triangle_level.eq(level)]
        counts.append({'triangle_level':level,'common_significant_motifs':len(group),
                       'same_hazard_direction':int(group.same_hazard_direction.sum()),
                       'opposite_hazard_directions':int((~group.same_hazard_direction).sum())})
        directory=out/f'tri{level}';directory.mkdir(exist_ok=True)
        group.to_csv(directory/'metadata.csv',index=False)
    counts=pd.DataFrame(counts);counts.to_csv(out/'overlap_by_level.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    x=counts.triangle_level.to_numpy()
    axes[0].bar(x,counts.common_significant_motifs,color='#54A24B')
    for level,n in zip(x,counts.common_significant_motifs):axes[0].text(level,n+3,str(n),ha='center')
    axes[0].set(title=f'{len(selected)} common significant motifs',xlabel='Triangle level',ylabel='Motif pairs',xticks=x,ylim=(0,max(1,counts.common_significant_motifs.max()*1.17)))
    axes[1].bar(x,counts.same_hazard_direction,label='Same direction',color='#3568A8')
    axes[1].bar(x,counts.opposite_hazard_directions,bottom=counts.same_hazard_direction,label='Opposite directions',color='#D1495B')
    axes[1].set(title='Hazard-ratio direction across cohorts',xlabel='Triangle level',ylabel='Motif pairs',xticks=x)
    axes[1].legend(frameon=False)
    fig.suptitle('Rank-normalized Cox p < 0.05 in BOTH datasets | no positive-median requirement',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/'overlap_by_level.png',dpi=180);fig.savefig(out/'overlap_by_level.pdf');plt.close(fig)
    summary={'active_definition':'finite nominal rank-normalized binary Cox Wald p < 0.05 in both BZ and METABRIC',
             'positive_median_required':False,'bh_adjustment_used_for_selection':False,
             'matching':'same triangle level and exact labeled graph isomorphism after aligning shared cell-type names; no cell-type merging',
             'common_significant_motifs':len(selected),'by_level':counts.to_dict('records'),
             'same_hazard_direction':int(selected.same_hazard_direction.sum()),
             'opposite_hazard_directions':int((~selected.same_hazard_direction).sum()),
             'zero_median_in_both':int((selected.bz_median_patient_rank_score.eq(0)&selected.metabric_median_patient_rank_score.eq(0)).sum()),
             'positive_median_in_both':int((selected.bz_median_patient_rank_score.gt(0)&selected.metabric_median_patient_rank_score.gt(0)).sum()),
             'source_table':str(args.input.resolve()),'source_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
             'pdf_pages':len(selected),'png_topology_figures':len(selected),'svg_topology_figures':len(selected)}
    write_json(summary,out/'comparison_summary.json')
    lines=['# Common significant motifs: active 424-motif collection','',
           f'**{len(selected)} pairs** satisfy nominal **rank-normalized binary Cox p < 0.05 in BOTH datasets**, without requiring a positive median score. BH q-values and hazard direction do not determine inclusion.','',
           'Graph matching still requires the same triangle level and exact cell-type-labeled topology, after aligning the five shared cell-type names. Cell types are not merged. Original graph IDs differ because numeric cell-label codes differ.','',
           '## Visualizations and metadata','',
           '- `all_common_significant_motifs.pdf`: one paired BZ/METABRIC graph figure per motif, covering every selected pair.',
           '- `common_significant_motifs_metadata.csv` and `.csv.gz`: one row per pair, containing both graph IDs and feature indices, node labels and edge lists, p-values, hazard ratios, confidence intervals, medians, cohort and group sizes, fit statuses, ranking fields, effect directions and artifact paths.',
           '- `common_significant_motif_members.csv`: one row per cohort per pair (848 rows for 424 pairs), with decoded cell-type compositions and statistics.',
           '- `triN/<common_motif_id>/`: paired topology `.png` and `.svg`, nested `metadata.json`, and separate `bz_graph_and_statistics.json` and `metabric_graph_and_statistics.json` files. Graph JSONs include original numeric cell labels, decoded cell-type names and adjacency.',
           '- `triN/metadata.csv`: level-specific metadata, including header-only tables for TRI-1 and TRI-2, which have no selected pairs.',
           '- `overlap_by_level.csv`, `.png`, `.pdf`: selected counts and same/opposite hazard-direction counts.','',
           '## Ranks and interpretation','',
           '`*_common_rank_*_rank_binary` preserves ranks among all 64,635 common motifs. `*_rank_within_common_significant_*` adds ranks among the selected 424 only. `*_original_rank_*_rank_binary` preserves the original positive-median-screen ranks where available; blank values mean the motif was outside that original ranking family or had an undefined fit. Raw-count screen ranks remain separately named `*_rank_*_unadjusted`. All ranks use ascending p-values with minimum ranks for exact ties.',
           '`max_cohort_p_value` is only a display-order key (the larger of the two p-values), not a combined p-value or meta-analysis.',
           f'{summary["same_hazard_direction"]} motifs have concordant hazard directions; {summary["opposite_hazard_directions"]} have opposite directions. Significance in both cohorts does not require the same direction. Graph colors encode shared cell types, not hazard direction.',
           'For a zero median, the strict high-score rule (`score > median`) becomes an absence-versus-presence split. Such motifs are deliberately retained here. The underlying unadjusted Efron Cox models and cohort data are unchanged.','',
           '## Reproduce','', '```bash','python breast_cancer_survival_common/export_common_significant_motifs.py','```','']
    (out/'README.md').write_text('\n'.join(lines))
    if out.resolve()==(HERE/'common_significant').resolve():
        # Preserve the former zero-overlap report before promoting the new definition.
        archive=HERE/'strict_positive_median'
        if not (archive/'comparison_summary.json').exists():
            archive.mkdir(exist_ok=True)
            for path in HERE.iterdir():
                if path.is_file() and path.suffix in {'.csv','.json','.png','.pdf','.md'}:
                    shutil.copy2(path,archive/path.name)
        for name in ['overlap_by_level.csv','overlap_by_level.png','overlap_by_level.pdf','comparison_summary.json']:
            shutil.copy2(out/name,HERE/name)
        root_table=selected.copy()
        for column in ['visualization','vector_visualization','metadata_file','motif_directory']:
            root_table[column]=[str((out/p).relative_to(HERE)) for p in root_table[column]]
        root_table.to_csv(HERE/'common_motif_pairs.csv',index=False)
        root_members=member_frame.copy()
        for column in ['visualization','metadata_file']:
            root_members[column]=[str((out/p).relative_to(HERE)) for p in root_members[column]]
        root_members.to_csv(HERE/'common_motif_members.csv',index=False)
        root_readme=['# BZ–METABRIC common motifs','',
                     f'## Active definition: {len(selected)} common significant motifs','',
                     '**Rank-normalized Cox p < 0.05 in both cohorts; no positive-median requirement.** The root overlap plots and pair/member tables now use this definition.','',
                     '- [Complete visualizations and metadata](common_significant/README.md)',
                     '- [All paired figures in one PDF](common_significant/all_common_significant_motifs.pdf)',
                     '- [Metadata for all selected pairs](common_significant/common_significant_motifs_metadata.csv)',
                     '- [All 64,635 observed common motifs](all_motifs/README.md)',
                     '- [Archived strict positive-median comparison](strict_positive_median/README.md)','',
                     'The former strict comparison required positive medians in both cohorts and had zero overlap. It is archived and is no longer the active definition.','']
        (HERE/'README.md').write_text('\n'.join(root_readme))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
