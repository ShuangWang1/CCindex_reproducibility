#!/usr/bin/env python3
"""Compare all observed TRI-1--7 motifs across cohorts, without outcome filters."""
from __future__ import annotations
import argparse
from collections import Counter
from html import escape
import json
import os
from pathlib import Path
import sqlite3
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
os.environ.setdefault('MPLCONFIGDIR','/tmp/common-all-breast-motifs')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from compare_rank_normalized_motifs import decode_graph, same_graph, normalize_name
from breast_cancer_survival_metabric import rank_methods
from breast_cancer_survival_metabric.analyze_motif_survival import vectorized_binary_cox, STATUS_LABELS

BZ_ART=REPO/'breast_cancer_classification_workstation/classification_by_level/mtf_occurrence/mrmr_selection/parameter_tuning_bz/artifacts'
BZ_RES=REPO/'breast_cancer_survival/motif_results'
META_RES=REPO/'breast_cancer_survival_metabric/motif_results'
BZ_MOTIFS=REPO/'breast_cancer_meta/BZ_res_threshold_10_cpp_0811'
KEYS=['triangle_level','wl_hash']
COLORS={'b cells':'#4C78A8','t cells':'#54A24B','endothelial':'#F2B134','macrophages':'#B279A2','tumor cells':'#E45756'}


def log(text): print(text,flush=True)


def csv(frame,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    frame.to_csv(temp,index=False,compression='gzip' if path.suffix=='.gz' else None)
    temp.replace(path)


def read(path,**kw): return pd.read_csv(path,dtype={'wl_hash':str},**kw)


def label_maps():
    result={}
    for cohort,dataset in [('bz','BZ'),('metabric','metabric')]:
        f=pd.read_csv(REPO/f'breast_cancer/{dataset}_samples_intra/node_label_cell_type_mapping.csv',dtype=str)
        result[cohort]=dict(f[['node_label','cell_type']].itertuples(index=False,name=None))
    return result


def encoding(graph):
    nodes=sorted(graph.nodes)
    labels='&'.join(graph.nodes[n]['numeric_label'] for n in nodes)
    return labels,';'.join(f'{u}-{v}' for u,v in sorted(graph.edges))


def recover_bz(level, selected, features, directory):
    if level==1:
        return selected[['wl_hash','representative_node_labels']].rename(columns={'representative_node_labels':'bz_node_labels'}).assign(bz_edge_list='0-1;0-2;1-2')
    root=BZ_ART/f'tri_{level}'
    counts=sparse.load_npz(root/'motif_counts.npz').tocsc()
    samples=pd.read_csv(root/'samples.csv')
    feature_index=features.set_index('wl_hash').feature_index
    columns=feature_index.loc[selected.wl_hash].to_numpy(int)
    if (np.diff(counts.indptr)[columns]==0).any(): raise ValueError('Observed BZ motif has no occurrence')
    first=counts.indices[counts.indptr[columns]]
    wanted=selected[['wl_hash']].copy().reset_index(drop=True)
    wanted['source_row']=first
    parts=[]
    for source_row,group in wanted.groupby('source_row'):
        sample=samples.iloc[int(source_row)]['sample']
        frame=pd.read_csv(BZ_MOTIFS/sample/f'ccindex_tri_{level}.csv',usecols=['wl_hash','node_labels','edge_list'],dtype=str)
        parts.append(frame.loc[frame.wl_hash.isin(group.wl_hash)].drop_duplicates('wl_hash'))
    found=pd.concat(parts,ignore_index=True).drop_duplicates('wl_hash') if parts else pd.DataFrame(columns=['wl_hash','node_labels','edge_list'])
    if set(found.wl_hash)!=set(selected.wl_hash): raise ValueError('Incomplete BZ topology recovery')
    return found.rename(columns={'node_labels':'bz_node_labels','edge_list':'bz_edge_list'})


def match_level(level,out,maps):
    destination=out/'cache'/f'tri{level}_common_topologies.csv.gz'
    audit_path=destination.with_name(f'tri{level}_matching_audit.json')
    if destination.exists() and audit_path.exists():
        return read(destination),json.loads(audit_path.read_text())
    if level==1:
        bz=read(BZ_RES/'tri_1_cox_results_sorted.csv',usecols=['feature_index','wl_hash','representative_node_labels','nodes','edges'])
    else:
        bz=read(BZ_ART/f'tri_{level}/feature_index.csv.gz',usecols=['feature_index','wl_hash','representative_node_labels','nodes','edges'])
    meta=read(META_RES/f'cache/tri{level}/features.csv.gz',usecols=['feature_index','wl_hash','representative_node_labels','nodes','edges'])
    bz_names={normalize_name(name):code for code,name in maps['bz'].items()}
    shared=set(bz_names)&{normalize_name(v) for v in maps['metabric'].values()}
    allowed={code for code,name in maps['metabric'].items() if normalize_name(name) in shared}
    keep=meta.representative_node_labels.map(lambda s:set(s.split('&'))<=allowed)
    eligible=meta.loc[keep].copy()
    top=read(META_RES/f'cache/tri{level}/topologies.csv.gz',usecols=['wl_hash','node_labels','edge_list'])
    top=top.loc[top.wl_hash.isin(eligible.wl_hash)]
    eligible=eligible.merge(top,on='wl_hash',validate='one_to_one')
    bz_ids=set(bz.wl_hash)
    rows=[]
    for r in eligible.itertuples(index=False):
        graph=decode_graph(r.node_labels,r.edge_list,maps['metabric'])
        recoded=graph.copy()
        nx.set_node_attributes(recoded,{n:bz_names[normalize_name(d['original_cell_type'])] for n,d in graph.nodes(data=True)},'label')
        target=nx.weisfeiler_lehman_graph_hash(recoded,node_attr='label')
        if target in bz_ids:
            rows.append({'triangle_level':level,'bz_wl_hash':target,'metabric_wl_hash':r.wl_hash,
                         'metabric_node_labels':r.node_labels,'metabric_edge_list':r.edge_list,
                         'metabric_feature_index':int(r.feature_index)})
    pairs=pd.DataFrame(rows,columns=['triangle_level','bz_wl_hash','metabric_wl_hash','metabric_node_labels','metabric_edge_list','metabric_feature_index'])
    selected=bz.loc[bz.wl_hash.isin(pairs.bz_wl_hash)]
    if len(pairs):
        topology=recover_bz(level,selected,bz,out)
        pairs=pairs.merge(topology,left_on='bz_wl_hash',right_on='wl_hash',validate='many_to_one').drop(columns='wl_hash')
        pairs=pairs.merge(selected[['wl_hash','feature_index']],left_on='bz_wl_hash',right_on='wl_hash',validate='many_to_one').drop(columns='wl_hash').rename(columns={'feature_index':'bz_feature_index'})
        accepted=[]
        for r in pairs.itertuples(index=False):
            a=decode_graph(r.bz_node_labels,r.bz_edge_list,maps['bz'])
            b=decode_graph(r.metabric_node_labels,r.metabric_edge_list,maps['metabric'])
            if same_graph(a,b): accepted.append(True)
            else: accepted.append(False)
        pairs=pairs.loc[accepted].reset_index(drop=True)
    else:
        for col in ['bz_node_labels','bz_edge_list','bz_feature_index']: pairs[col]=pd.Series(dtype=str)
    pairs['common_motif_id']=[f'tri{level}_{h}' for h in pairs.bz_wl_hash]
    audit={'triangle_level':level,'bz_all_observed':len(bz),'metabric_all_observed':len(meta),
           'metabric_shared_type_candidates':len(eligible),'common_pairs':len(pairs),
           'bz_common_motifs':int(pairs.bz_wl_hash.nunique()),'metabric_common_motifs':int(pairs.metabric_wl_hash.nunique())}
    csv(pairs,destination)
    audit_path.write_text(json.dumps(audit,indent=2)+'\n')
    log(f'TRI-{level}: {len(bz):,} BZ / {len(meta):,} METABRIC observed; {len(pairs):,} common pairs')
    return pairs,audit


def cohort_bz():
    root=BZ_RES/'normalized_by_ranking_motif_analysis'
    cores=pd.read_csv(root/'core_rank_scores.csv.gz',usecols=['PID','source_sample']).drop_duplicates()
    patients=pd.read_csv(root/'patient_rank_scores.csv.gz',usecols=['PID','time_months','event']).drop_duplicates().sort_values('PID').reset_index(drop=True)
    if patients.PID.duplicated().any() or cores.source_sample.duplicated().any(): raise ValueError('Inconsistent cached BZ cohort')
    return cores,patients


def bz_scores(level,selected,out):
    cores,patients=cohort_bz()
    if level==1:
        raw=read(BZ_RES/'tri_1_cox_results_sorted.csv',usecols=['wl_hash'])
        vocab=sorted(raw.wl_hash)
        lookup={h:i for i,h in enumerate(vocab)}
        samples=cores[['source_sample','PID']].copy()
        inventory=pd.read_csv(BZ_ART/'tri_2/samples.csv')
        samples=samples.merge(inventory[['sample','source_sample']],on='source_sample',validate='one_to_one')
        rows=[];cols=[];values=[]
        for i,sample in enumerate(samples.itertuples(index=False)):
            f=read(BZ_MOTIFS/sample.sample/'ccindex_tri_1.csv',usecols=['wl_hash','matches'])
            f=f.loc[f.matches.gt(0)]
            rows.extend([i]*len(f));cols.extend([lookup[h] for h in f.wl_hash]);values.extend(f.matches)
        counts=sparse.csr_matrix((values,(rows,cols)),shape=(len(samples),len(vocab)))
        columns=np.array([lookup[h] for h in selected],int)
    else:
        directory=BZ_ART/f'tri_{level}'
        inventory=pd.read_csv(directory/'samples.csv').reset_index(names='artifact_row')
        samples=cores.merge(inventory[['sample','source_sample','artifact_row']],on='source_sample',validate='one_to_one')
        f=read(directory/'feature_index.csv.gz',usecols=['wl_hash','feature_index'])
        lookup=f.set_index('wl_hash').feature_index
        counts=sparse.load_npz(directory/'motif_counts.npz').tocsr()[samples.artifact_row.to_numpy(int)]
        columns=lookup.loc[selected].to_numpy(int)
    scores=rank_methods.rank_score_sparse(counts,counts.shape[1])[:,columns]
    position={pid:i for i,pid in enumerate(patients.PID)}
    row=[];col=[];weights=[]
    groups=samples.groupby('PID').indices
    for pid,i in position.items():
        indices=groups[pid]
        row.extend([i]*len(indices));col.extend(indices);weights.extend([1/len(indices)]*len(indices))
    mean=sparse.csr_matrix((weights,(row,col)),shape=(len(patients),len(samples)))
    result=(mean@scores).tocsc()
    return result,patients.time_months.to_numpy(float),patients.event.to_numpy(int)


def normalized_stats(cohort,level,hashes,out):
    dest=out/'cache'/f'{cohort}_tri{level}_rank_statistics.csv.gz'
    if dest.exists():
        saved=read(dest)
        if set(saved.wl_hash)!=set(hashes): raise ValueError('Cached common motif identities changed; use a new output directory')
        return saved
    if cohort=='bz':
        scores,time,event=bz_scores(level,hashes,out)
    else:
        f=read(META_RES/f'cache/tri{level}/features.csv.gz',usecols=['wl_hash','feature_index'])
        columns=f.set_index('wl_hash').feature_index.loc[hashes].to_numpy(int)
        scores=sparse.load_npz(META_RES/f'cache/tri{level}/patient_rank_scores.npz').tocsc()[:,columns]
        patients=pd.read_csv(META_RES/'patient_level_survival.csv')
        time,event=patients.time_days.to_numpy(float),patients.event.to_numpy(int)
    parts=[]
    for start in range(0,len(hashes),5000):
        selected=hashes[start:start+5000]
        values=scores[:,start:start+5000].toarray()
        median=np.median(values,axis=0)
        fit=vectorized_binary_cox(values>median,time,event)
        part=pd.DataFrame({'triangle_level':level,'wl_hash':selected,'median_patient_rank_score':median,
                           'p_value_rank_binary':fit['p_value'],'hazard_ratio_rank_binary':fit['hazard_ratio'],
                           'ci95_low_rank_binary':fit['ci95_low'],'ci95_high_rank_binary':fit['ci95_high'],
                           'rank_binary_fit_status':[STATUS_LABELS[int(c)] for c in fit['status_code']],
                           'n_survival':len(time),'events_survival':int(event.sum()),
                           'n_high_rank':fit['n_enriched'],'n_low_rank':fit['n_non_enriched']})
        parts.append(part)
    result=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=KEYS+['p_value_rank_binary','median_patient_rank_score'])
    csv(result,dest)
    log(f'TRI-{level} {cohort}: rank-normalized statistics for {len(result):,} common motifs')
    return result


def raw_statistics(cohort,level,hashes):
    if cohort=='bz':
        database=BZ_RES/'significant_mtf_trace_ranking_normalized/cache/cox_results.sqlite'
        connection=sqlite3.connect(f'file:{database}?mode=ro',uri=True)
        pieces=[]
        for start in range(0,len(hashes),400):
            values=hashes[start:start+400]
            placeholders=','.join('?' for _ in values)
            pieces.append(pd.read_sql_query(f'SELECT * FROM cox WHERE triangle_level=? AND wl_hash IN ({placeholders})',connection,params=[level,*values]))
        connection.close()
        f=pd.concat(pieces,ignore_index=True)
    elif level==1:
        f=read(META_RES/'cache/tri1/raw_cox_results.csv.gz')
        f['rank_within_level_unadjusted']=f.p_value_unadjusted.rank(method='min').astype('Int64')
        f['rank_global_unadjusted']=pd.NA
    else:
        f=read(REPO/f'breast_cancer_survival_metabric/results/tri_{level}_cox_results_sorted.csv.gz')
        rename={c:c+'_unadjusted' for c in ['p_value','hazard_ratio','ci95_low','ci95_high','rank_global','rank_within_level']}
        f=f.rename(columns={**rename,'median_patient_count':'median_patient_raw_count'})
    cols=KEYS+[c for c in ['p_value_unadjusted','hazard_ratio_unadjusted','ci95_low_unadjusted','ci95_high_unadjusted',
                          'rank_global_unadjusted','rank_within_level_unadjusted','median_patient_raw_count'] if c in f]
    result=f.loc[f.wl_hash.isin(hashes),cols].copy()
    if set(result.wl_hash)!=set(hashes): raise ValueError(f'Missing raw results: {cohort} TRI-{level}')
    return result


def format_value(value):
    if pd.isna(value): return 'NA'
    number=float(value)
    return f'{int(number):,}' if number.is_integer() else f'{number:.4g}'


def save_svg(record,folder,maps):
    a=decode_graph(record.bz_node_labels,record.bz_edge_list,maps['bz'])
    b=decode_graph(record.metabric_node_labels,record.metabric_edge_list,maps['metabric'])
    matcher=nx.algorithms.isomorphism.GraphMatcher(b,a,node_match=lambda x,y:x['label']==y['label'])
    if not matcher.is_isomorphic(): raise ValueError('Saved visualization pair is not isomorphic')
    positions=nx.planar_layout(a) if nx.check_planarity(a)[0] else nx.spring_layout(a,seed=0)
    body=['<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="600" viewBox="0 0 1240 600">',
          '<rect width="1240" height="600" fill="white"/>',
          f'<text x="620" y="25" text-anchor="middle" font-family="sans-serif" font-size="16">TRI-{record.triangle_level}: same topology and cell-type names</text>']
    for panel,(cohort,graph) in enumerate([('bz',a),('metabric',b)]):
        offset=panel*620
        h=getattr(record,f'{cohort}_wl_hash')
        lines=[f'{cohort.upper()} | {h}',
               f'Rank-normalized p={format_value(getattr(record,cohort+"_p_value_rank_binary"))}; HR={format_value(getattr(record,cohort+"_hazard_ratio_rank_binary"))}',
               f'Common-set TRI rank={format_value(getattr(record,cohort+"_common_rank_within_level_rank_binary"))}; original raw TRI rank={format_value(getattr(record,cohort+"_rank_within_level_unadjusted"))}']
        for i,line in enumerate(lines): body.append(f'<text x="{offset+310}" y="{55+i*23}" text-anchor="middle" font-family="sans-serif" font-size="13">{escape(line)}</text>')
        xy={n:(offset+310+float(positions[n if cohort=='bz' else matcher.mapping[n]][0])*210,
                    300+float(positions[n if cohort=='bz' else matcher.mapping[n]][1])*155) for n in graph}
        for u,v in graph.edges:
            x1,y1=xy[u];x2,y2=xy[v]
            body.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#666" stroke-width="2"/>')
        for n,d in graph.nodes(data=True):
            x,y=xy[n];color=COLORS[d['label']]
            body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="15" fill="{color}" stroke="white" stroke-width="2"><title>{escape(d["original_cell_type"])}; original label {escape(d["numeric_label"])}</title></circle>')
    names=sorted({d['label'] for _,d in a.nodes(data=True)})
    for i,name in enumerate(names):
        x=90+i*225
        body.extend([f'<circle cx="{x}" cy="550" r="8" fill="{COLORS[name]}"/>',f'<text x="{x+15}" y="555" font-family="sans-serif" font-size="13">{escape(name)}</text>'])
    body.append('</svg>')
    folder.mkdir(parents=True,exist_ok=True)
    (folder/f'{record.bz_wl_hash}__{record.metabric_wl_hash}.svg').write_text('\n'.join(body))


def attach_original_screen(frame,primary):
    """Screen membership does not require an estimable Cox p-value."""
    renamed={c:'original_'+c for c in primary if c not in KEYS}
    primary=primary.rename(columns=renamed).copy()
    primary['in_original_positive_median_rank_screen']=True
    result=frame.merge(primary,on=KEYS,how='left',validate='one_to_one')
    result['in_original_positive_median_rank_screen']=result.in_original_positive_median_rank_screen.eq(True)
    return result


def save_example_panels(pairs,out,maps):
    """Small PNG previews; complete paired visualizations remain in the SVG collection."""
    for level,frame in pairs.groupby('triangle_level'):
        ranked=frame.assign(example_rank_sum=frame.bz_common_rank_global_rank_binary+frame.metabric_common_rank_global_rank_binary)
        examples=ranked.sort_values(['example_rank_sum','bz_wl_hash'],na_position='last').head(6)
        fig,axes=plt.subplots(len(examples),2,figsize=(13,3.3*len(examples)),squeeze=False)
        for i,r in enumerate(examples.itertuples(index=False)):
            a=decode_graph(r.bz_node_labels,r.bz_edge_list,maps['bz'])
            b=decode_graph(r.metabric_node_labels,r.metabric_edge_list,maps['metabric'])
            matcher=nx.algorithms.isomorphism.GraphMatcher(b,a,node_match=lambda x,y:x['label']==y['label'])
            if not matcher.is_isomorphic(): raise ValueError('Preview graphs do not match')
            reference=nx.planar_layout(a) if nx.check_planarity(a)[0] else nx.spring_layout(a,seed=0)
            for j,(cohort,graph) in enumerate([('bz',a),('metabric',b)]):
                ax=axes[i,j]
                positions={n:reference[n if cohort=='bz' else matcher.mapping[n]] for n in graph}
                nx.draw_networkx(graph,pos=positions,ax=ax,with_labels=False,node_size=210,
                    node_color=[COLORS[d['label']] for _,d in graph.nodes(data=True)],edgecolors='white',width=1.4)
                h=getattr(r,cohort+'_wl_hash')
                ax.set_title(f'{cohort.upper()} {h}\nCommon-set TRI rank {format_value(getattr(r,cohort+"_common_rank_within_level_rank_binary"))}; '
                             f'p={format_value(getattr(r,cohort+"_p_value_rank_binary"))}',fontsize=8)
                ax.set_axis_off()
        handles=[plt.Line2D([0],[0],marker='o',linestyle='none',color=color,label=name) for name,color in COLORS.items()]
        fig.legend(handles=handles,loc='lower center',ncol=5,frameon=False,fontsize=9)
        fig.suptitle(f'TRI-{level}: examples of common motifs (all {len(frame):,} pairs have individual SVGs)\nExamples ordered by summed common-set global Cox p-value rank',fontsize=11)
        fig.tight_layout(rect=(0,.035,1,.955))
        fig.savefig(out/f'tri{level}/common_motif_examples.png',dpi=160)
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=Path(__file__).resolve().parent/'all_motifs')
    parser.add_argument('--skip-visualizations',action='store_true')
    args=parser.parse_args();out=args.outdir;out.mkdir(parents=True,exist_ok=True)
    maps=label_maps()
    parts=[];audits=[];statistics={'bz':[],'metabric':[]}
    for level in range(1,8):
        pairs,audit=match_level(level,out,maps)
        parts.append(pairs);audits.append(audit)
        if len(pairs):
            for cohort in statistics:
                hashes=sorted(set(pairs[f'{cohort}_wl_hash']))
                normalized=normalized_stats(cohort,level,hashes,out)
                raw=raw_statistics(cohort,level,hashes)
                statistics[cohort].append(normalized.merge(raw,on=KEYS,validate='one_to_one'))
    pairs=pd.concat(parts,ignore_index=True)
    for cohort in statistics:
        frame=pd.concat(statistics[cohort],ignore_index=True)
        frame['common_rank_global_rank_binary']=frame.p_value_rank_binary.rank(method='min').astype('Int64')
        frame['common_rank_within_level_rank_binary']=frame.groupby('triangle_level').p_value_rank_binary.rank(method='min').astype('Int64')
        primary=read((BZ_RES if cohort=='bz' else META_RES)/'normalized_by_ranking_motif_analysis/significant/all_nonzero_threshold_rank_results.csv.gz',
                     usecols=KEYS+['p_value_rank_binary','rank_global_rank_binary','rank_within_level_rank_binary','q_value_bh_global_rank_binary','q_value_bh_within_level_rank_binary'])
        frame=attach_original_screen(frame,primary)
        overlap=frame.original_p_value_rank_binary.notna()&frame.p_value_rank_binary.notna()
        np.testing.assert_allclose(frame.loc[overlap,'p_value_rank_binary'],frame.loc[overlap,'original_p_value_rank_binary'],rtol=1e-6,atol=1e-10)
        csv(frame,out/f'{cohort}_common_motif_statistics.csv.gz')
        rename={c:f'{cohort}_{c}' for c in frame if c!='triangle_level'}
        pairs=pairs.merge(frame.rename(columns=rename),on=['triangle_level',f'{cohort}_wl_hash'],validate='many_to_one')
    pairs['same_hazard_direction']=pd.array([pd.NA]*len(pairs),dtype='boolean')
    estimated=pairs.bz_hazard_ratio_rank_binary.notna()&pairs.metabric_hazard_ratio_rank_binary.notna()
    pairs.loc[estimated,'same_hazard_direction']=(pairs.loc[estimated,'bz_hazard_ratio_rank_binary']>1)==(pairs.loc[estimated,'metabric_hazard_ratio_rank_binary']>1)
    pairs['visualization']=[f'tri{r.triangle_level}/visualizations/{r.bz_wl_hash}__{r.metabric_wl_hash}.svg' for r in pairs.itertuples()]
    csv(pairs,out/'common_motif_pairs.csv.gz')
    csv(pd.DataFrame(audits),out/'overlap_by_level.csv')
    for level,group in pairs.groupby('triangle_level'):
        csv(group,out/f'tri{level}/common_motif_pairs.csv')
        if not args.skip_visualizations:
            for i,r in enumerate(group.itertuples(index=False),1):
                save_svg(r,out/f'tri{level}/visualizations',maps)
                if i%5000==0: log(f'TRI-{level}: saved {i:,}/{len(group):,} paired vector visualizations')
    if not args.skip_visualizations:
        save_example_panels(pairs,out,maps)
    audit=pd.DataFrame(audits)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    x=np.arange(1,8)
    axes[0].bar(x-.18,audit.bz_all_observed,width=.36,label='BZ',color='#3568A8')
    axes[0].bar(x+.18,audit.metabric_all_observed,width=.36,label='METABRIC',color='#D1495B')
    axes[0].set(yscale='log',xticks=x,xlabel='Triangle level',ylabel='Observed motifs (log scale)',title='Complete observed vocabularies');axes[0].legend(frameon=False)
    axes[1].bar(x,audit.common_pairs,color='#54A24B')
    axes[1].set(xticks=x,xlabel='Triangle level',ylabel='Common motif pairs',title='Same topology and shared cell-type names')
    for level,n in zip(x,audit.common_pairs):axes[1].text(level,n+max(audit.common_pairs)*.015,f'{n:,}',ha='center',fontsize=9)
    axes[1].set_ylim(0,max(audit.common_pairs)*1.15)
    fig.tight_layout();fig.savefig(out/'overlap_by_level.png',dpi=180);fig.savefig(out/'overlap_by_level.pdf');plt.close(fig)
    summary={'scope':'all observed motifs; no p-value, median, or survival estimability filter',
             'matching':'same triangle level and exact graph isomorphism preserving shared cell-type names; no cell-type merging',
             'bz_all_observed':int(audit.bz_all_observed.sum()),'metabric_all_observed':int(audit.metabric_all_observed.sum()),
             'common_pairs':len(pairs),'bz_common_motifs':int(pairs[['triangle_level','bz_wl_hash']].drop_duplicates().shape[0]),
             'metabric_common_motifs':int(pairs[['triangle_level','metabric_wl_hash']].drop_duplicates().shape[0]),
             'by_level':audits,'visualizations_written':not args.skip_visualizations,
             'both_nominal_p_below_0_05':int((pairs.bz_p_value_rank_binary.lt(.05)&pairs.metabric_p_value_rank_binary.lt(.05)).sum()),
             'normalized_ranking_scope':'common-set ranks use every estimable common source motif; original positive-median-screen ranks are separately retained',
             'shared_cell_types':sorted(COLORS),'networkx_version':nx.__version__}
    (out/'comparison_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# All observed common motifs: BZ and METABRIC','',
           f'Compared **{summary["bz_all_observed"]:,} BZ** and **{summary["metabric_all_observed"]:,} METABRIC** observed motifs across TRI-1–7. Found **{len(pairs):,} common pairs**. No significance, positive-median, or Cox-estimability filter was applied.','',
           'Matching preserves graph adjacency and cell-type identity after translating numeric codes into shared cell-type names. The five shared cell types are B cells, T cells, endothelial, macrophages, and tumor cells. Stromal categories are not merged. A translated WL hash is a lookup accelerator; exact labeled graph isomorphism verifies every match. The separate original WL hash from each dataset is retained.','',
           '## Files','',
           '- `common_motif_pairs.csv.gz`: every match, both original graph IDs, graph encodings, raw-count ranks/statistics, normalized statistics, original normalized-screen ranks when available, and the paired SVG path.',
           '- `triN/common_motif_pairs.csv`: the same information separated by triangle level.',
           '- `triN/visualizations/<BZ_hash>__<METABRIC_hash>.svg`: a paired topology visualization for **every** match, with aligned node positions, consistent cell-type colors, graph IDs, ranks, p-values and hazard ratios. SVGs open in a browser and remain sharp when zoomed.',
           '- `bz_common_motif_statistics.csv.gz`, `metabric_common_motif_statistics.csv.gz`: one row per common source motif, retaining undefined Cox fits with explicit statuses.',
           '- `overlap_by_level.csv`, `.png`, `.pdf`: all-level overlap summary. `triN/common_motif_examples.png` previews six pairs per level, ordered by summed common-set global rank.','',
           '## Ranking definitions','',
           '- `rank_*_unadjusted`: existing raw-count Cox ranks from the complete original raw-count screen. METABRIC TRI-1 has a newly derived within-level raw rank but no original global raw rank because the original METABRIC raw screen covered TRI-2–7.',
           '- `original_rank_*_rank_binary`: the pre-existing rank-normalized ranks from each cohort’s complete positive-median screen. Blank for motifs outside that screening family; these ranks are not fabricated.',
           '- `common_rank_*_rank_binary`: normalized Cox p-value ranks recalculated over **all common motifs** in each cohort (global across levels and separately within TRI level). Undefined p-values have no rank. These ranks have a different denominator from the original screen ranks.',
           'Normalized scores rank positive counts within image and TRI level against the full observed cohort vocabulary, average positive-count ties, assign absent motifs the last rank, and equally average images per patient. High means strictly above the patient median, including zero-median absence-versus-presence splits. The same unadjusted Efron Cox/Wald method is used for both cohorts. BZ retains 278 patients/79 deaths (months); METABRIC retains 454 patients/212 deaths (days). Original primary-screen p-values are checked against the recomputed results where both exist.','',
           'Graph IDs, ranks, and undefined fit statuses are separate concepts: an observed common graph can have no estimable survival association. No new multiplicity-controlled claims are made. The biological annotations differ in granularity, so this strict comparison cannot assess equivalence involving BZ-specific stromal or mixed T/B annotations.','',
           '## Reproduce','', '```bash','python breast_cancer_survival_common/compare_all_observed_motifs.py','```','',
           'Caches are local to this collection. The significant-only comparison in the parent folder is preserved.','']
    (out/'README.md').write_text('\n'.join(lines))
    log(json.dumps({k:v for k,v in summary.items() if k!='by_level'},indent=2))


if __name__=='__main__':main()
