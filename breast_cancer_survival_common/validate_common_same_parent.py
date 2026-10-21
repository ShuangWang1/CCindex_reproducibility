#!/usr/bin/env python3
"""Validate the generated common-parent collection and save its audit report."""
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import mannwhitneyu

root = Path(__file__).resolve().parent / 'all_motifs'
out = root / 'same_parent_comparsion'
reports = []
parent_counts = pd.read_csv(out / 'heatmap_parent_counts.csv').set_index('heatmap_child_level')
through = int(os.environ.get('THROUGH_LEVEL', '7'))
for level in range(2, through + 1):
    pair = pd.read_csv(root / 'cache' / f'tri{level}_common_topologies.csv.gz')
    relationships = pd.read_csv(out / 'cache' / f'tri{level}_parent_child_relationships.csv.gz')
    features = pd.read_csv(out / 'cache' / f'tri{level}_intrinsic_criteria.csv.gz', float_precision='round_trip').set_index('common_motif_id')
    assert set(features.index) == set(pair.common_motif_id)
    assert not relationships.duplicated(['parent_common_motif_id', 'child_common_motif_id']).any()
    matrices = []
    orders = []
    for cohort in ['bz', 'metta']:
        folder = out / cohort / f'tri{level}'
        summary = json.loads((folder / 'analysis_summary.json').read_text())
        matrix = pd.read_csv(folder / 'clustered_location_bh_q_matrix.csv.gz', index_col=0)
        results = pd.read_csv(folder / 'all_parent_criteria_comparisons.csv.gz')
        audit = pd.read_csv(folder / 'parent_group_index.csv')
        children = pd.read_csv(folder / 'child_motif_audit.csv.gz')
        members = pd.read_csv(folder / 'all_parent_child_memberships.csv.gz')
        order = pd.read_csv(folder / 'clustered_location_bh_q_order.csv')
        assert set(children.common_motif_id) == set(pair.common_motif_id)
        assert len(children) == len(pair) == summary['common_children']
        assert set(members.parent_common_motif_id) == set(audit.parent_common_motif_id)
        assert set(matrix.columns) == set(audit.loc[audit.included_in_heatmap, 'parent_common_motif_id'])
        assert summary['common_parents'] == len(audit)
        assert summary['heatmap_parents'] == matrix.shape[1]
        assert summary['parents_removed_from_heatmap'] == len(audit)-matrix.shape[1]
        assert len(members) == len(relationships) == summary['parent_child_memberships']
        assert not members.duplicated(['parent_common_motif_id', 'child_common_motif_id']).any()
        assert order.loc[order.axis.eq('criterion'), 'id'].tolist() == matrix.index.tolist()
        assert order.loc[order.axis.eq('parent'), 'id'].tolist() == matrix.columns.tolist()
        displayed = audit.loc[audit.included_in_heatmap].sort_values('cluster_position')
        assert displayed.parent_common_motif_id.tolist() == matrix.columns.tolist()
        np.testing.assert_array_equal(displayed.n_testable_criteria, matrix.notna().sum(axis=0))
        assert audit.loc[~audit.included_in_heatmap, 'n_testable_criteria'].eq(0).all()
        assert int(audit.n_common_children.sum()) == len(members)
        assert (audit.n_common_children == audit.n_significant + audit.n_nonsignificant + audit.n_undefined).all()
        assert (audit.eligible == ((audit.n_significant >= 3) & (audit.n_nonsignificant >= 3))).all()
        assert int(matrix.notna().sum().sum()) == len(results) == summary['tests']
        assert not results.duplicated(['parent_common_motif_id', 'metric']).any()
        if len(results):
            actual = matrix.to_numpy()[matrix.index.get_indexer(results.metric), matrix.columns.get_indexer(results.parent_common_motif_id)]
            np.testing.assert_allclose(actual, results.location_q_bh, rtol=0, atol=0)
            assert results.location_q_bh.between(0, 1).all()
            for parent, group in results.groupby('parent_common_motif_id'):
                p = group.location_p_value.to_numpy()
                sort = np.argsort(p)
                q = np.minimum(np.minimum.accumulate((p[sort] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1], 1)
                np.testing.assert_allclose(group.location_q_bh.to_numpy()[sort], q, rtol=1e-12, atol=1e-14)
            samples = pd.concat([results.head(4), results.tail(4)]).drop_duplicates(['parent_common_motif_id', 'metric'])
            for row in samples.itertuples():
                group = members.loc[members.parent_common_motif_id.eq(row.parent_common_motif_id)]
                case = features.loc[group.loc[group.significance_group.eq('significant'), 'child_common_motif_id'], row.metric].dropna()
                control = features.loc[group.loc[group.significance_group.eq('nonsignificant'), 'child_common_motif_id'], row.metric].dropna()
                assert len(case) == row.n_significant and len(control) == row.n_nonsignificant
                scalar = mannwhitneyu(case, control, alternative='two-sided', method='asymptotic')
                np.testing.assert_allclose(scalar.pvalue, row.location_p_value, rtol=1e-12, atol=1e-14)
                np.testing.assert_allclose(2*scalar.statistic/(len(case)*len(control))-1, row.cliffs_delta, atol=1e-14)
        no_tests = audit.loc[~audit.eligible & audit.included_in_heatmap, 'parent_common_motif_id']
        assert matrix[no_tests].isna().all().all()
        if cohort == 'bz' and level in [2, 3]:
            assert matrix.isna().all().all()
        pdf = subprocess.run(['pdfinfo', str(folder / 'clustered_location_bh_q_heatmap.pdf')], capture_output=True, text=True, check=True).stdout
        pages = int(next(line.split(':')[1] for line in pdf.splitlines() if line.startswith('Pages:')))
        expected_pages = 1 + math.ceil(matrix.shape[0]/100)*math.ceil(matrix.shape[1]/80)
        assert pages == expected_pages == summary['pdf_pages']
        with Image.open(folder / 'clustered_location_bh_q_heatmap.png') as image:
            image.verify()
        matrices.append(matrix)
        orders.append(order[['axis', 'position', 'id']])
        reports.append({'level': level, 'cohort': cohort, 'common_children': len(pair),
                        'common_parents': len(audit), 'heatmap_parents': matrix.shape[1],
                        'parents_removed_from_heatmap': len(audit)-matrix.shape[1],
                        'criteria': matrix.shape[0],
                        'eligible_parents': int(audit.eligible.sum()), 'tests': len(results),
                        'q_below_0_05': int((results.location_q_bh < .05).sum()),
                        'pdf_pages': pages, 'status': 'passed'})
        print('PASS', reports[-1], flush=True)
    assert matrices[0].index.equals(matrices[1].index)
    assert matrices[0].columns.equals(matrices[1].columns)
    retained = matrices[0].notna().any() | matrices[1].notna().any()
    assert retained.all(), 'A parent is still gray across all criteria in both cohorts'
    tested_parents = set()
    for cohort in ['bz', 'metta']:
        results = pd.read_csv(out / cohort / f'tri{level}' / 'all_parent_criteria_comparisons.csv.gz')
        tested_parents.update(results.parent_common_motif_id)
    assert tested_parents == set(matrices[0].columns)
    assert parent_counts.loc[level, 'remaining_parent_motifs'] == matrices[0].shape[1]
    assert parent_counts.loc[level, 'original_parent_motifs'] == reports[-1]['common_parents']
    assert parent_counts.loc[level, 'removed_parent_motifs'] == reports[-1]['parents_removed_from_heatmap']
    pd.testing.assert_frame_equal(orders[0], orders[1])
report = {'status': 'passed', 'common_children_across_levels': sum(r['common_children'] for r in reports if r['cohort']=='bz'),
          'cohort_level_outputs': len(reports), 'regression_checks': 7,
          'checks': ['complete common-motif coverage', 'unique shared-parent memberships',
                     'matching axes and page boundaries', 'class counts and eligibility',
                     'raw q matrix and complete result-table agreement', 'within-parent BH recomputation',
                     'sample scalar Mann-Whitney and effect-size recomputation',
                     'all-gray invalid tests', 'no jointly all-gray plotted parents',
                     'per-level parent count CSV', 'all PDF page counts', 'all PNG integrity'],
          'outputs': reports}
destination = 'validation_summary.json' if through == 7 else f'validation_through_tri{through}.json'
(out / destination).write_text(json.dumps(report, indent=2)+'\n')
print(f'All {len(reports)} completed cohort-level outputs passed validation.')
