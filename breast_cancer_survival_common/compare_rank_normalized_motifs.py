#!/usr/bin/env python3
"""Compare rank-significant BZ and METABRIC motifs with explicit label semantics."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('MPLCONFIGDIR', '/tmp/common-breast-motifs-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import networkx as nx
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
KEYS = ['triangle_level', 'wl_hash']
ARTIFACTS = ['motif_delaunay_topology.png', 'rank_normalized_survival_km.png',
             'patient_rank_distribution.png', 'patient_rank_scores.csv', 'statistics.json']
RANKS = ['rank_within_level_rank_binary', 'rank_global_rank_binary', 'p_value_rank_binary',
         'q_value_bh_within_level_rank_binary', 'q_value_bh_global_rank_binary',
         'hazard_ratio_rank_binary', 'ci95_low_rank_binary', 'ci95_high_rank_binary',
         'median_patient_rank_score', 'n_survival', 'events_survival']


def normalize_name(value):
    return ' '.join(str(value).casefold().split())


def decode_graph(labels, edge_list, mapping, mode='cell-types', aliases=None):
    aliases = aliases or {}
    graph = nx.Graph()
    for i, code in enumerate(str(labels).split('&')):
        original = mapping[code]
        name = normalize_name(original)
        matched_name = aliases.get(name, name)
        graph.add_node(i, label='cell' if mode == 'topology-only' else matched_name,
                       original_cell_type=original, numeric_label=code, matched_cell_type=matched_name)
    graph.add_edges_from(tuple(map(int, edge.split('-'))) for edge in str(edge_list).split(';'))
    return graph


def same_graph(left, right):
    return nx.is_isomorphic(left, right, node_match=lambda a, b: a['label'] == b['label'])


def build_groups(records):
    """WL hashes narrow candidates; exact labeled isomorphism determines matches."""
    buckets = defaultdict(list)
    groups = []
    for record in records:
        graph = record['graph']
        digest = nx.weisfeiler_lehman_graph_hash(graph, node_attr='label')
        bucket = buckets[(int(record['triangle_level']), digest)]
        for group in bucket:
            if same_graph(graph, group['representative']):
                group['members'].append(record)
                break
        else:
            group = {'common_motif_id': f'tri{record["triangle_level"]}_{digest}_{len(bucket)+1}',
                     'triangle_level': int(record['triangle_level']), 'representative': graph, 'members': [record]}
            bucket.append(group)
            groups.append(group)
    return [g for g in groups if {r['cohort'] for r in g['members']} == {'bz', 'metabric'}]


def load_cohort(cohort, root, mapping_path, aliases, mode):
    path = root/'significant_rank_normalized_survival_results.csv'
    frame = pd.read_csv(path, dtype={'wl_hash': str})
    if frame.duplicated(KEYS).any():
        raise ValueError(f'Duplicate motif identities in {path}')
    if not frame.p_value_rank_binary.lt(.05).all() or not frame.median_patient_rank_score.gt(0).all():
        raise ValueError(f'Input is not the positive-median, nominal rank-significant collection: {path}')
    mapping_frame = pd.read_csv(mapping_path, dtype=str)
    mapping = dict(mapping_frame[['node_label', 'cell_type']].itertuples(index=False, name=None))
    if cohort == 'bz':
        topology_path = root.parents[1]/'significant_mtf_trace_ranking_normalized/cache/target_topologies.csv.gz'
        topology = pd.read_csv(topology_path, dtype={'wl_hash': str})[KEYS+['node_labels', 'edge_list']]
        topology = topology.loc[topology.set_index(KEYS).index.isin(frame.set_index(KEYS).index)]
    else:
        parts = []
        for level, subset in frame.groupby('triangle_level'):
            path = root.parents[1]/f'cache/tri{level}/topologies.csv.gz'
            top = pd.read_csv(path, usecols=['wl_hash', 'node_labels', 'edge_list'], dtype=str)
            top = top.loc[top.wl_hash.isin(subset.wl_hash)].copy()
            top['triangle_level'] = level
            parts.append(top)
        topology = pd.concat(parts, ignore_index=True)
    frame = frame.drop(columns=['node_labels', 'edge_list'], errors='ignore').merge(topology, on=KEYS, validate='one_to_one')
    records = []
    for row in frame.to_dict('records'):
        directory = root/row['motif_directory']
        for name in ARTIFACTS:
            if not (directory/name).is_file():
                raise FileNotFoundError(directory/name)
        graph = decode_graph(row['node_labels'], row['edge_list'], mapping, mode, aliases)
        encoded = graph.copy()
        nx.set_node_attributes(encoded, {n: graph.nodes[n]['numeric_label'] for n in graph}, 'label')
        if nx.weisfeiler_lehman_graph_hash(encoded, node_attr='label') != row['wl_hash']:
            raise ValueError(f'Source topology does not reproduce {cohort} graph ID {row["wl_hash"]}')
        records.append({**row, 'cohort': cohort, 'graph': graph, 'source_directory': str(directory.resolve())})
    return frame, records, mapping_frame


def graph_panel(group, output, pdf):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8))
    names = sorted({d['matched_cell_type'] for r in group['members'] for _, d in r['graph'].nodes(data=True)})
    palette = {name: plt.get_cmap('tab20')(i % 20) for i, name in enumerate(names)}
    reference = group['representative']
    positions = nx.planar_layout(reference) if nx.check_planarity(reference)[0] else nx.spring_layout(reference, seed=0)
    for ax, cohort in zip(axes, ['bz', 'metabric']):
        members = [r for r in group['members'] if r['cohort'] == cohort]
        best = min(members, key=lambda r: (r['p_value_rank_binary'], r['wl_hash']))
        graph = best['graph']
        matcher = nx.algorithms.isomorphism.GraphMatcher(graph, reference, node_match=lambda a, b: a['label'] == b['label'])
        if not matcher.is_isomorphic():
            raise ValueError('Comparison panel contains nonmatching graphs')
        layout = {n: positions[matcher.mapping[n]] for n in graph}
        nx.draw_networkx(graph, pos=layout, ax=ax, with_labels=False, node_size=800,
                         node_color=[palette[graph.nodes[n]['matched_cell_type']] for n in graph],
                         edgecolors='white', linewidths=1.5, edge_color='#666666', width=2)
        handles = [plt.Line2D([0], [0], marker='o', linestyle='none', color=palette[name], label=name)
                   for name in sorted({d['matched_cell_type'] for _, d in graph.nodes(data=True)})]
        ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, -.10), ncol=2, frameon=False)
        ax.set_title(f'{cohort.upper()} | {best["wl_hash"]}\nTRI rank {int(best["rank_within_level_rank_binary"])}; '
                     f'global rank {int(best["rank_global_rank_binary"])}; p={best["p_value_rank_binary"]:.3g}\n'
                     f'{len(members)} matching source motif(s); best-ranked example', fontsize=10)
        ax.set_axis_off()
    fig.suptitle(group['common_motif_id'], fontsize=12)
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.savefig(output, dpi=180, bbox_inches='tight')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bz-root', type=Path, default=REPO/'breast_cancer_survival/motif_results/normalized_by_ranking_motif_analysis/significant')
    parser.add_argument('--metabric-root', type=Path, default=REPO/'breast_cancer_survival_metabric/motif_results/normalized_by_ranking_motif_analysis/significant')
    parser.add_argument('--outdir', type=Path, default=Path(__file__).resolve().parent/'strict_positive_median')
    parser.add_argument('--matching', choices=['cell-types', 'topology-only'], default='cell-types')
    parser.add_argument('--cell-type-aliases', type=Path, help='Explicit JSON mapping of cell-type names to matched categories; no categories are merged by default')
    args = parser.parse_args()
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)
    aliases = {normalize_name(k): normalize_name(v) for k, v in json.loads(args.cell_type_aliases.read_text()).items()} if args.cell_type_aliases else {}
    maps = {'bz': REPO/'breast_cancer/BZ_samples_intra/node_label_cell_type_mapping.csv',
            'metabric': REPO/'breast_cancer/metabric_samples_intra/node_label_cell_type_mapping.csv'}
    frames, records, mapping_rows = {}, [], []
    for cohort, root in [('bz', args.bz_root), ('metabric', args.metabric_root)]:
        frame, cohort_records, mapping = load_cohort(cohort, root, maps[cohort], aliases, args.matching)
        frames[cohort] = frame
        records.extend(cohort_records)
        mapping['cohort'] = cohort
        mapping['matched_cell_type'] = mapping.cell_type.map(lambda s: aliases.get(normalize_name(s), normalize_name(s)))
        mapping_rows.append(mapping)
    pd.concat(mapping_rows, ignore_index=True).to_csv(out/'cell_type_label_mapping.csv', index=False)
    exact = frames['bz'].merge(frames['metabric'], on=KEYS, suffixes=('_bz', '_metabric'))
    exact.to_csv(out/'exact_hash_overlap.csv', index=False)
    groups = build_groups(records)
    groups.sort(key=lambda g: (g['triangle_level'], g['common_motif_id']))
    memberships = {}
    for group in groups:
        for r in group['members']: memberships[(r['cohort'], r['triangle_level'], r['wl_hash'])] = group['common_motif_id']
    inventory = []
    for r in records:
        data = {k: v for k, v in r.items() if k != 'graph'}
        data['common_motif_id'] = memberships.get((r['cohort'], r['triangle_level'], r['wl_hash']), '')
        data['is_common'] = bool(data['common_motif_id'])
        for name in ARTIFACTS: data[f'source_{Path(name).stem}'] = str(Path(r['source_directory'])/name)
        inventory.append(data)
    inventory = pd.DataFrame(inventory)
    inventory.to_csv(out/'all_significant_motif_inventory.csv', index=False)
    for cohort in frames:
        inventory.loc[inventory.cohort.eq(cohort)].to_csv(out/f'{cohort}_significant_motifs_with_rankings.csv', index=False)
    members = inventory.loc[inventory.is_common].copy()
    members.to_csv(out/'common_motif_members.csv', index=False)
    pair_rows = []
    for group in groups:
        for bz, meta in product([r for r in group['members'] if r['cohort'] == 'bz'], [r for r in group['members'] if r['cohort'] == 'metabric']):
            pair = {'common_motif_id': group['common_motif_id'], 'triangle_level': group['triangle_level']}
            for prefix, r in [('bz', bz), ('metabric', meta)]:
                pair.update({f'{prefix}_{k}': v for k, v in r.items() if k not in ['graph', 'triangle_level', 'cohort']})
            pair['same_hazard_direction'] = (bz['hazard_ratio_rank_binary'] > 1) == (meta['hazard_ratio_rank_binary'] > 1)
            pair_rows.append(pair)
    pair_columns = ['common_motif_id', 'triangle_level'] + [f'{cohort}_{col}' for cohort in ['bz', 'metabric'] for col in ['wl_hash', 'cell_type_composition', *RANKS]] + ['same_hazard_direction']
    pairs = pd.DataFrame(pair_rows) if pair_rows else pd.DataFrame(columns=pair_columns)
    pairs.to_csv(out/'common_motif_pairs.csv', index=False)
    counts = []
    for level in range(1, 8):
        row = {'triangle_level': level, 'bz_significant': int(frames['bz'].triangle_level.eq(level).sum()),
               'metabric_significant': int(frames['metabric'].triangle_level.eq(level).sum()),
               'exact_hash_pairs': int(exact.triangle_level.eq(level).sum()),
               'common_groups': sum(g['triangle_level'] == level for g in groups),
               'common_pairs': int(pairs.triangle_level.eq(level).sum())}
        for cohort in frames:
            row[f'{cohort}_common_members'] = int((members.triangle_level.eq(level) & members.cohort.eq(cohort)).sum())
        counts.append(row)
    counts = pd.DataFrame(counts)
    counts.to_csv(out/'overlap_by_level.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(1, 8)
    axes[0].bar(x-.18, counts.bz_significant, width=.36, label='BZ', color='#3568A8')
    axes[0].bar(x+.18, counts.metabric_significant, width=.36, label='METABRIC', color='#D1495B')
    axes[0].set(title='Rank-significant motifs supplied for comparison', xlabel='Triangle level', ylabel='Motifs', xticks=x)
    axes[0].legend(frameon=False)
    axes[1].bar(x, counts.common_groups, color='#54A24B')
    for level, n in zip(x, counts.common_groups): axes[1].text(level, n+.03, str(n), ha='center')
    axes[1].set(title=f'Common motifs: {args.matching}', xlabel='Triangle level', ylabel='Exact isomorphism groups', xticks=x,
                ylim=(0, max(1, counts.common_groups.max()*1.2)))
    fig.tight_layout()
    fig.savefig(out/'overlap_by_level.png', dpi=180)
    fig.savefig(out/'overlap_by_level.pdf')
    plt.close(fig)
    if groups:
        with PdfPages(out/'common_motif_comparisons.pdf') as pdf:
            for group in groups:
                directory = out/f'tri{group["triangle_level"]}'/group['common_motif_id']
                directory.mkdir(parents=True, exist_ok=True)
                graph_panel(group, directory/'cohort_topology_comparison.png', pdf)
                subset = members.loc[members.common_motif_id.eq(group['common_motif_id'])]
                subset.to_csv(directory/'graph_ids_and_rankings.csv', index=False)
                for r in group['members']:
                    target = directory/r['cohort']/r['wl_hash']
                    target.mkdir(parents=True, exist_ok=True)
                    for name in ARTIFACTS: shutil.copy2(Path(r['source_directory'])/name, target/name)
                    graph_json = {'cohort': r['cohort'], 'wl_hash': r['wl_hash'], 'triangle_level': int(r['triangle_level']),
                                  'nodes': [{'id': n, **d} for n, d in r['graph'].nodes(data=True)],
                                  'edges': [list(e) for e in r['graph'].edges]}
                    (target/'graph.json').write_text(json.dumps(graph_json, indent=2)+'\n')
    summary = {'matching': args.matching, 'cell_type_aliases': aliases, 'exact_hash_pairs': len(exact),
               'common_motif_groups': len(groups), 'common_motif_pairs': len(pairs),
               'bz_common_motifs': int(members.cohort.eq('bz').sum()), 'metabric_common_motifs': int(members.cohort.eq('metabric').sum()),
               'bz_significant_motifs': len(frames['bz']), 'metabric_significant_motifs': len(frames['metabric']),
               'by_level': counts.to_dict('records'), 'networkx_version': nx.__version__,
               'source_roots': {'bz': str(args.bz_root.resolve()), 'metabric': str(args.metabric_root.resolve())},
               'source_table_sha256': {cohort: hashlib.sha256((root/'significant_rank_normalized_survival_results.csv').read_bytes()).hexdigest()
                                       for cohort, root in [('bz', args.bz_root), ('metabric', args.metabric_root)]}}
    (out/'comparison_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# Common rank-normalized significant breast-cancer motifs', '',
             f'Compared **{len(frames["bz"])} BZ** and **{len(frames["metabric"])} METABRIC** significant motifs across TRI-1–7.',
             f'Found **{len(groups)} common motif groups**, **{len(pairs)} cross-cohort pairs**, and **{len(exact)} exact raw-hash pairs**.', '',
             '## Matching rule', '',
             f'Mode: `{args.matching}`. Motifs must have the same triangle level and be exactly graph-isomorphic. In cell-type mode, node cell-type names must match after case/whitespace normalization and any explicitly supplied aliases. WL hashes only narrow candidate pairs; exact labeled graph isomorphism confirms identity.',
             'The numeric cell labels differ between datasets. Original WL hashes are cohort-specific graph IDs and are always retained separately. For example, numeric label 6 means tumor cells in BZ and T cells in METABRIC.',
             f'Explicit cell-type aliases: `{json.dumps(aliases, sort_keys=True)}`. No cell types are merged when this dictionary is empty.',
             'Without explicit aliases, the shared cell types are B cells, T cells, endothelial, macrophages, and tumor cells. BZ stromal cells are not automatically equated with METABRIC fibroblasts, myofibroblasts, vascular SMA+, or myoepithelial cells.', '',
             '## Results', '', '\n'.join(['| '+' | '.join(counts.columns)+' |', '| '+' | '.join(['---']*len(counts.columns))+' |']+['| '+' | '.join(map(str, row))+' |' for row in counts.itertuples(index=False,name=None)]), '',
             '## Files', '',
             '- `common_motif_pairs.csv`: paired original graph IDs, within-TRI and global ranking information, p/q values, hazard ratios and the original metadata from both cohorts.',
             '- `common_motif_members.csv`: one row per common source motif. A matching group may have multiple source motifs after explicit label coarsening or when ignoring cell types.',
             '- `bz_significant_motifs_with_rankings.csv` and `metabric_significant_motifs_with_rankings.csv`: complete source inventories, with original IDs, rankings, cell types and absolute paths to each source visualization and patient/statistics table.',
             '- `cell_type_label_mapping.csv`: both numeric-to-cell-type mappings and the categories used for matching.',
             '- `overlap_by_level.csv`, `.png`, `.pdf`: the all-level overlap summary.',
             '- When matches exist, `triN/<common_motif_id>/` contains a paired topology visualization and ranking table, plus `bz/<original_hash>/` and `metabric/<original_hash>/` copies of the original topology, survival and distribution plots, patient scores, statistics, and decoded graph JSON. The combined figure shows the best-ranked example per cohort; the member table lists every matched source graph.', '',
             'Original ranking fields are preserved: `rank_within_level_rank_binary` ranks within that cohort’s complete positive-median TRI-level screen; `rank_global_rank_binary` ranks across all its screened levels. Ranks are not recalculated over only the overlaps. Selection is nominal unadjusted rank-binary Cox p < 0.05 and positive median score, not BH q < 0.05.', '',
             'If no common motifs exist, overlap tables retain their headers and no common-motif graph packages are fabricated. The complete inventories still provide both cohorts’ IDs, rankings and visualization paths.', '',
             '## Reproduction', '', '```bash', 'python breast_cancer_survival_common/compare_rank_normalized_motifs.py', '```', '',
             'Use a separate `--outdir` for alternative matching rules. `--matching topology-only` ignores all cell types and does not establish biological equivalence. `--cell-type-aliases mapping.json` applies an explicitly chosen mapping before cell-type-aware matching.','']
    (out/'README.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['by_level','source_roots','source_table_sha256']}, indent=2))


if __name__ == '__main__':
    main()
