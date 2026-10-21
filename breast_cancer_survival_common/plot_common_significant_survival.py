#!/usr/bin/env python3
"""Generate cohort-specific KM plots and auditable patient tables for common motifs."""
from pathlib import Path
from collections import Counter
import json
import numpy as np
import pandas as pd
from scipy import sparse
from PIL import Image
import compare_all_observed_motifs as common
from breast_cancer_survival_metabric import rank_methods as methods

HERE=Path(__file__).resolve().parent
OUT=HERE/'common_significant'

def main():
    pairs=pd.read_csv(OUT/'common_significant_motifs_metadata.csv')
    maps=common.label_maps(); records=[]
    for level, frame in pairs.groupby('triangle_level',sort=True):
        for cohort in ['bz','metabric']:
            hashes=frame[cohort+'_wl_hash'].tolist()
            if cohort=='bz':
                matrix,time,event=common.bz_scores(int(level),hashes,HERE/'all_motifs')
                ids=common.cohort_bz()[1].PID.to_numpy(); unit='months'
            else:
                features=pd.read_csv(common.META_RES/f'cache/tri{level}/features.csv.gz')
                columns=features.set_index('wl_hash').feature_index.loc[hashes].to_numpy(int)
                matrix=sparse.load_npz(common.META_RES/f'cache/tri{level}/patient_rank_scores.npz').tocsc()[:,columns]
                patients=pd.read_csv(common.META_RES/'patient_level_survival.csv')
                ids=patients.patient_id.to_numpy();time=patients.time_days.to_numpy(float);event=patients.event.to_numpy(int);unit='days'
            values=matrix.toarray()
            fits=common.vectorized_binary_cox(values > np.median(values,axis=0),time,event)
            for col,(_,row) in enumerate(frame.iterrows()):
                result,high,z=methods.analyze_rank_survival(values[:,col],time,event)
                # Use the same Efron solver as the all-common comparison; PHReg
                # can fail numerically for rare, small high-score groups.
                for target, source in [('p_value_rank_binary','p_value'),('hazard_ratio_rank_binary','hazard_ratio'),('ci95_low_rank_binary','ci95_low'),('ci95_high_rank_binary','ci95_high')]:
                    result[target]=float(fits[source][col])
                result['rank_binary_fit_status']=common.STATUS_LABELS[int(fits['status_code'][col])]
                result['rank_binary_fit_warning']=''
                for field in ['p_value_rank_binary','hazard_ratio_rank_binary','ci95_low_rank_binary','ci95_high_rank_binary','median_patient_rank_score','n_high_rank','n_low_rank']:
                    if not np.isclose(result[field],row[cohort+'_'+field],rtol=2e-4,atol=1e-8):
                        raise ValueError(f'Metadata mismatch: {cohort} {row.common_motif_id} {field}: {result[field]} vs {row[cohort+"_"+field]}')
                composition=Counter(maps[cohort][str(x)] for x in str(row[cohort+'_node_labels']).split('&'))
                result.update(triangle_level=int(level),wl_hash=row[cohort+'_wl_hash'],cell_type_composition='; '.join(f'{name} ×{count}' for name,count in composition.items()))
                data=pd.DataFrame({'patient_id':ids,'time_days':time,'event':event,'rank_score':values[:,col],'rank_score_z':z,'rank_group':np.where(high,'High rank score','Low rank score')})
                folder=OUT/row.motif_directory; path=folder/f'{cohort}_rank_normalized_survival_km.png'
                methods.plot_km(data,pd.Series(result),path,180,time_unit=unit,cohort=cohort.upper())
                with Image.open(path) as im: im.verify()
                data.rename(columns={'time_days':'time_'+unit}).to_csv(folder/f'{cohort}_survival_patient_scores.csv.gz',index=False)
                result.update(common_motif_id=row.common_motif_id,cohort=cohort,survival_time_unit=unit,plot=str(path.relative_to(OUT)))
                (folder/f'{cohort}_survival_statistics.json').write_text(json.dumps(methods.json_safe(result),indent=2)+'\n')
                records.append(result)
            print(f'TRI-{level} {cohort}: {len(frame)} plots validated',flush=True)
    pd.DataFrame(records).to_csv(OUT/'survival_plot_metadata.csv',index=False)
    report={'status':'passed','motif_pairs':len(pairs),'plots':len(records),'cohorts':['bz','metabric'],'selection':'existing 424 common significant motifs','group_definition':'rank_score > cohort median','time_units':{'bz':'months','metabric':'days'},'checks':['recomputed Cox statistics and group sizes match existing metadata','all PNGs verified','patient scores exported for every plot']}
    (OUT/'survival_plot_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    with (OUT/'README.md').open('a') as f:
        f.write('\n## Survival plots\n\nEach motif directory includes `bz_rank_normalized_survival_km.png` and `metabric_rank_normalized_survival_km.png`, with 95% confidence bands, censor marks, group sizes/deaths, binary Cox HR/CI/p, log-rank p, and continuous HR per SD. BZ time is months; METABRIC time is days. High rank means strictly above the cohort median. Matching `*_survival_patient_scores.csv.gz` and `*_survival_statistics.json` provide the underlying data. `survival_plot_metadata.csv` indexes all 848 plots.\n')

if __name__=='__main__': main()
