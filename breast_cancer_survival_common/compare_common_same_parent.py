#!/usr/bin/env python3
"""Compare common TRI-2--7 children within common immediate structural parents."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import warnings

os.environ.setdefault('MPLCONFIGDIR', '/tmp/common-same-parent-matplotlib')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DEFAULT_INPUT = ROOT / 'all_motifs'
NAMES = {'0': 'B cells', '2': 'T cells', '3': 'endothelial',
         '4': 'macrophages', '6': 'tumor cells'}
VERSION = 1
REL_COLUMNS = ['parent_level', 'child_level', 'parent_common_motif_id',
               'child_common_motif_id', 'growth_operations', 'operation_witnesses']
RESULT_COLUMNS = ['parent_common_motif_id', 'metric', 'n_significant',
                  'n_nonsignificant', 'mann_whitney_u', 'location_p_value',
                  'location_q_bh', 'cliffs_delta', 'median_difference', 'mean_difference',
                  'standardized_median_difference', 'absolute_cliffs_delta',
                  'rank_within_parent', 'display_name', 'family']


def log(message):
    print(message, flush=True)


def write_csv(frame, path, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    frame.to_csv(temporary, compression='gzip' if path.suffix == '.gz' else None,
                 **({'index': False} | kwargs))
    temporary.replace(path)


def write_json(value, path):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read_features(path):
    # Default CSV float parsing can split/collapse ties and change rank tests.
    return pd.read_csv(path, float_precision='round_trip')


@lru_cache(maxsize=1)
def engine():
    path = REPO / 'breast_cancer_survival/motif_results/sibling_topology_analysis_new/run_analysis.py'
    spec = importlib.util.spec_from_file_location('common_intrinsic_criteria', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.configure_labels(NAMES, dict(zip(NAMES, ['#4C78A8', '#54A24B', '#F2B134', '#B279A2', '#E45756'])),
                            dict(zip(NAMES, ['B', 'T', 'End', 'Mac', 'Tum'])))
    return module


def decode(labels, edges, recoding=None):
    graph = nx.Graph()
    graph.add_nodes_from((i, {'label': recoding[v] if recoding else v})
                         for i, v in enumerate(labels.split('&')))
    graph.add_edges_from(tuple(map(int, edge.split('-'))) for edge in edges.split(';'))
    return graph


def same_graph(left, right):
    return nx.is_isomorphic(left, right, node_match=lambda a, b: a['label'] == b['label'])


def direct_parents(graph):
    """Existing reverse-growth rules, with changes recorded in child coordinates."""
    expected = sum(nx.triangles(graph).values()) // 3 - 1
    for node in graph:
        neighbors = list(graph.neighbors(node))
        if len(neighbors) == 2 and graph.has_edge(*neighbors):
            parent = graph.copy()
            parent.remove_node(node)
            if nx.is_connected(parent) and sum(nx.triangles(parent).values()) // 3 == expected:
                yield 'add_vertex', str(node), parent
    for u, v in graph.edges:
        # Removing an edge destroys exactly one triangle iff it has one common neighbor.
        if len(set(graph[u]) & set(graph[v])) != 1:
            continue
        parent = graph.copy()
        parent.remove_edge(u, v)
        if nx.is_connected(parent):
            yield 'add_edge', f'{u}-{v}', parent


def validate_pair(row, recoding):
    bz = decode(row['bz_node_labels'], row['bz_edge_list'])
    meta_native = decode(row['metabric_node_labels'], row['metabric_edge_list'])
    for graph, cohort in [(bz, 'bz'), (meta_native, 'metabric')]:
        if nx.weisfeiler_lehman_graph_hash(graph, node_attr='label') != row[f'{cohort}_wl_hash']:
            raise ValueError(f'Invalid {cohort} topology hash: {row["common_motif_id"]}')
        if sum(nx.triangles(graph).values()) // 3 != int(row['triangle_level']):
            raise ValueError('Triangle level does not agree with graph')
    meta = decode(row['metabric_node_labels'], row['metabric_edge_list'], recoding)
    if not same_graph(bz, meta):
        raise ValueError(f'Non-isomorphic common motif: {row["common_motif_id"]}')
    return bz, meta


def init_worker(parents, recoding, permutations, seed):
    global PARENTS, RECODING, PERMUTATIONS, SEED
    PARENTS, RECODING, PERMUTATIONS, SEED = parents, recoding, permutations, seed
    parent_graph.cache_clear()
    engine()


@lru_cache(maxsize=4096)
def parent_graph(key):
    row = PARENTS[key]
    return decode(row['bz_node_labels'], row['bz_edge_list'])


def process_graph(row):
    graph, meta = validate_pair(row, RECODING)
    features, _ = engine().extra_features(graph, row['common_motif_id'], PERMUTATIONS, SEED)
    found = {}
    for operation, change, parent in direct_parents(graph):
        key = nx.weisfeiler_lehman_graph_hash(parent, node_attr='label')
        if key in PARENTS and same_graph(parent, parent_graph(key)):
            found.setdefault(key, []).append({'operation': operation, 'changed': change})
    # Check the shared parent membership independently in METABRIC coordinates.
    meta_found = set()
    for _, _, parent in direct_parents(meta):
        key = nx.weisfeiler_lehman_graph_hash(parent, node_attr='label')
        if key in PARENTS and same_graph(parent, parent_graph(key)):
            meta_found.add(key)
    if set(found) != meta_found:
        raise ValueError('Parent membership differs between matched cohort graphs')
    relationships = [{
        'parent_level': int(row['triangle_level']) - 1,
        'child_level': int(row['triangle_level']),
        'parent_common_motif_id': PARENTS[key]['common_motif_id'],
        'child_common_motif_id': row['common_motif_id'],
        'growth_operations': ';'.join(sorted({v['operation'] for v in values})),
        'operation_witnesses': json.dumps(values, sort_keys=True),
    } for key, values in sorted(found.items())]
    return {'common_motif_id': row['common_motif_id'], **features}, relationships


def bh(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind='stable')
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.minimum(adjusted, 1)
    return result


def classify(statistics):
    p = pd.to_numeric(statistics.p_value_rank_binary, errors='coerce')
    valid = statistics.rank_binary_fit_status.eq('estimated') & np.isfinite(p) & p.gt(0) & p.le(1)
    return pd.Series(np.where(valid, np.where(p.lt(.05), 'significant', 'nonsignificant'),
                             'undefined'), index=statistics.index)


def compare_values(values, significant, nonsignificant, minimum=3):
    """Vectorized existing two-sided asymptotic Mann--Whitney test across criteria."""
    case, control = values[significant], values[nonsignificant]
    nc, nn = np.isfinite(case).sum(axis=0), np.isfinite(control).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        span = np.nanmax(values[significant | nonsignificant], axis=0) - np.nanmin(values[significant | nonsignificant], axis=0)
    selected = np.flatnonzero((nc >= minimum) & (nn >= minimum) & (span > 0))
    if not len(selected):
        return pd.DataFrame(columns=RESULT_COLUMNS[1:])
    case, control = case[:, selected], control[:, selected]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        test = mannwhitneyu(case, control, axis=0, alternative='two-sided',
                           method='asymptotic', nan_policy='omit')
        median = np.nanmedian(case, axis=0) - np.nanmedian(control, axis=0)
        sd = np.nanstd(np.concatenate([case, control]), axis=0)
        standardized = np.divide(median, sd, out=np.full_like(median, np.nan), where=sd > 0)
        mean = np.nanmean(case, axis=0) - np.nanmean(control, axis=0)
    return pd.DataFrame({'metric_index': selected, 'n_significant': nc[selected],
                         'n_nonsignificant': nn[selected], 'mann_whitney_u': test.statistic,
                         'location_p_value': test.pvalue, 'location_q_bh': bh(test.pvalue),
                         'cliffs_delta': 2 * test.statistic / (nc[selected] * nn[selected]) - 1,
                         'median_difference': median, 'mean_difference': mean,
                         'standardized_median_difference': standardized})


def cohort_analysis(pairs, parents, relationships, features, metrics, statistics, cohort):
    stats = statistics.loc[statistics.triangle_level.eq(int(pairs.triangle_level.iloc[0]))].copy()
    stats['significance_group'] = classify(stats)
    identities = pairs[['common_motif_id', f'{cohort}_wl_hash']].rename(columns={f'{cohort}_wl_hash': 'wl_hash'})
    children = identities.merge(stats, on='wl_hash', validate='one_to_one')
    if len(children) != len(pairs):
        raise ValueError('Missing cohort statistics for common children')
    children['has_common_parent'] = children.common_motif_id.isin(relationships.child_common_motif_id)
    members = relationships.merge(children, left_on='child_common_motif_id', right_on='common_motif_id', validate='many_to_one').drop(columns='common_motif_id')
    indexed_features = features.set_index('common_motif_id')[metrics]
    retained_parents = parents.loc[parents.common_motif_id.isin(relationships.parent_common_motif_id)].copy()
    grouped = {key: group for key, group in members.groupby('parent_common_motif_id', sort=True)}
    audit, tables = [], []
    for row in retained_parents.sort_values('common_motif_id').itertuples(index=False):
        group = grouped[row.common_motif_id]
        counts = group.significance_group.value_counts()
        ns, nn, nu = (int(counts.get(k, 0)) for k in ['significant', 'nonsignificant', 'undefined'])
        eligible = ns >= 3 and nn >= 3
        reason = '' if eligible else ('no_estimable_children' if not ns + nn else
                 'missing_significance_class' if not ns or not nn else 'fewer_than_3_children_in_one_or_both_classes')
        audit.append({'parent_common_motif_id': row.common_motif_id,
                      'parent_hash': getattr(row, f'{cohort}_wl_hash'),
                      'bz_parent_hash': row.bz_wl_hash, 'metabric_parent_hash': row.metabric_wl_hash,
                      'n_common_children': len(group), 'n_significant': ns,
                      'n_nonsignificant': nn, 'n_undefined': nu, 'eligible': eligible,
                      'exclusion_reason': reason})
        if not eligible:
            continue
        values = indexed_features.loc[group.child_common_motif_id].to_numpy(float)
        result = compare_values(values, group.significance_group.eq('significant').to_numpy(),
                                group.significance_group.eq('nonsignificant').to_numpy())
        if not result.empty:
            result['metric'] = np.asarray(metrics)[result.pop('metric_index').to_numpy(int)]
            result['parent_common_motif_id'] = row.common_motif_id
            result['absolute_cliffs_delta'] = result.cliffs_delta.abs()
            result = result.sort_values(['location_q_bh', 'absolute_cliffs_delta', 'metric'], ascending=[True, False, True])
            result['rank_within_parent'] = np.arange(1, len(result) + 1)
            result['display_name'] = result.metric.map(engine().display_name)
            result['family'] = result.metric.map(engine().family_for)
            tables.append(result)
    results = pd.concat(tables, ignore_index=True)[RESULT_COLUMNS] if tables else pd.DataFrame(columns=RESULT_COLUMNS)
    audit = pd.DataFrame(audit)
    columns = audit.parent_common_motif_id.tolist()
    matrix = pd.DataFrame(np.nan, index=metrics, columns=columns)
    if len(results):
        matrix.update(results.pivot(index='metric', columns='parent_common_motif_id', values='location_q_bh'))
    return matrix, results, audit, children, members


def common_order(matrices):
    """Joint evidence clustering, excluding parents unavailable in both cohorts."""
    rows, columns = matrices[0].index.tolist(), matrices[0].columns.tolist()
    arrays = [-np.log10(m.clip(lower=1e-12, upper=1).fillna(1).to_numpy(float)) for m in matrices]
    if len(rows) > 1:
        data = np.concatenate(arrays, axis=1)
        rows = [rows[i] for i in leaves_list(linkage(data, method='average', metric='euclidean', optimal_ordering=False))]
    observed = np.logical_or.reduce([m.notna().any(axis=0).to_numpy() for m in matrices])
    active = np.flatnonzero(observed)
    if len(active) > 1:
        data = np.concatenate([a[:, active].T for a in arrays], axis=1)
        active = active[leaves_list(linkage(data, method='average', metric='euclidean', optimal_ordering=False))]
    columns = [columns[i] for i in active]
    return rows, columns


def visible_parents(matrices):
    """Keep a shared column if at least one cohort has a finite criterion value."""
    reference = matrices[0]
    for matrix in matrices[1:]:
        if not reference.index.equals(matrix.index) or not reference.columns.equals(matrix.columns):
            raise ValueError('Paired heatmap axes differ')
    observed = np.logical_or.reduce([np.isfinite(m.to_numpy(float)).any(axis=0) for m in matrices])
    return reference.columns[observed].tolist()


def annotate_parent_audit(audit, matrix):
    """Retain the complete source audit while identifying displayed columns."""
    audit = audit.copy()
    audit['included_in_heatmap'] = audit.parent_common_motif_id.isin(matrix.columns)
    positions = {key: i+1 for i, key in enumerate(matrix.columns)}
    audit['cluster_position'] = audit.parent_common_motif_id.map(positions).astype('Int64')
    counts = matrix.notna().sum(axis=0)
    audit['n_testable_criteria'] = audit.parent_common_motif_id.map(counts).fillna(0).astype(int)
    audit['heatmap_exclusion_reason'] = np.where(audit.included_in_heatmap, '', 'all_criteria_missing_in_both_cohorts')
    return audit


def save_heatmap_tables(matrix, audit, destination):
    write_csv(matrix, destination / 'clustered_location_bh_q_matrix.csv.gz', index=True, index_label='metric')
    order = [{'axis': 'criterion', 'position': i+1, 'id': name,
              'display_name': engine().display_name(name), 'family': engine().family_for(name),
              'nonmissing_q_values': int(matrix.loc[name].notna().sum())} for i, name in enumerate(matrix.index)]
    order += [{'axis': 'parent', 'position': i+1, 'id': name,
               'display_name': name, 'nonmissing_q_values': int(matrix[name].notna().sum())}
              for i, name in enumerate(matrix.columns)]
    write_csv(pd.DataFrame(order), destination / 'clustered_location_bh_q_order.csv')
    write_csv(audit, destination / 'parent_group_index.csv')


def draw_heatmap(matrix, audit, title, detailed=False):
    nr, nc = matrix.shape
    fig, ax = plt.subplots(figsize=(24, 18) if not detailed else (26, max(10, .17 * nr + 4)))
    if not nr or not nc:
        ax.text(.5, .5, 'No parent has a testable criterion in either dataset', ha='center', va='center')
        ax.set_axis_off()
        fig.suptitle(title)
        return fig
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('#d9d9d9')
    image = ax.imshow(np.ma.masked_invalid(matrix.to_numpy(float)), aspect='auto',
                      interpolation='none', cmap=cmap, norm=Normalize(0, 1))
    if detailed:
        ax.set_yticks(np.arange(nr), [engine().display_name(m) for m in matrix.index], fontsize=6)
        counts = audit.set_index('parent_common_motif_id')
        labels = [f'{c.split("_", 1)[1][:8]} S{counts.loc[c, "n_significant"]}/N{counts.loc[c, "n_nonsignificant"]}/U{counts.loc[c, "n_undefined"]}' for c in matrix.columns]
        ax.set_xticks(np.arange(nc), labels, rotation=90, fontsize=6)
    else:
        rticks = np.unique(np.linspace(0, nr - 1, min(nr, 60), dtype=int))
        cticks = np.unique(np.linspace(0, nc - 1, min(nc, 80), dtype=int))
        ax.set_yticks(rticks, [engine().display_name(matrix.index[i]) for i in rticks], fontsize=6)
        ax.set_xticks(cticks, [matrix.columns[i].split('_', 1)[1][:8] for i in cticks], rotation=90, fontsize=6)
    ax.set_xlabel('Common parent ID (BZ hash prefix); details: S significant / N nonsignificant / U undefined')
    ax.set_ylabel('Intrinsic criterion (shared five-cell-type definitions)')
    cb = fig.colorbar(image, ax=ax, pad=.015, fraction=.025, ticks=[0, .05, .1, .25, .5, .75, 1])
    cb.set_label('Within-parent BH q-value; gray = not testable')
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .96))
    return fig


def render(matrix, audit, cohort, level, output):
    title = f'{cohort}: common TRI-{level} children within common TRI-{level-1} parents\n{len(matrix)} criteria × {matrix.shape[1]} parents; jointly ordered BZ/METABRIC'
    with PdfPages(output / 'clustered_location_bh_q_heatmap.pdf') as pdf:
        fig = draw_heatmap(matrix, audit, title)
        fig.savefig(output / 'clustered_location_bh_q_heatmap.png', dpi=150)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        page = 1
        for start_col in range(0, matrix.shape[1], 80):
            for start_row in range(0, matrix.shape[0], 100):
                page += 1
                part = matrix.iloc[start_row:start_row+100, start_col:start_col+80]
                subtitle = f'{title}\nPDF page {page}: parents {start_col+1}–{start_col+part.shape[1]}, criteria {start_row+1}–{start_row+len(part)}'
                fig = draw_heatmap(part, audit, subtitle, detailed=True)
                pdf.savefig(fig, dpi=150)
                plt.close(fig)
                if page % 50 == 0:
                    log(f'TRI-{level} {cohort}: rendered {page} PDF pages')
    return page


def fingerprint(paths, settings):
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for path in paths:
        digest.update(path.name.encode())
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def refresh_heatmaps(level, output):
    """Prune existing paired matrices and render without recomputing any tests."""
    destinations = [output / cohort / f'tri{level}' for cohort in ['bz', 'metta']]
    matrices = [pd.read_csv(p / 'clustered_location_bh_q_matrix.csv.gz', index_col=0,
                            float_precision='round_trip') for p in destinations]
    columns = visible_parents(matrices)
    summaries = {}
    for cohort, folder, destination, matrix in zip(['BZ', 'METABRIC'], ['bz', 'metta'], destinations, matrices):
        matrix = matrix.loc[:, columns]
        audit = annotate_parent_audit(pd.read_csv(destination / 'parent_group_index.csv'), matrix)
        save_heatmap_tables(matrix, audit, destination)
        log(f'TRI-{level} {cohort}: keeping {len(columns):,}/{len(audit):,} parent columns; rendering')
        pages = render(matrix, audit, cohort, level, destination)
        summary = json.loads((destination / 'analysis_summary.json').read_text())
        summary.update(common_parents=len(audit), heatmap_parents=len(columns),
                       parents_removed_from_heatmap=len(audit)-len(columns), pdf_pages=pages,
                       heatmap_parent_filter='at_least_one_finite_q_in_either_cohort')
        write_json(summary, destination / 'analysis_summary.json')
        summaries[folder] = summary
    return summaries


def write_parent_counts(output):
    rows = []
    for level in range(2, 8):
        paths = [output / cohort / f'tri{level}' for cohort in ['bz', 'metta']]
        if not all((p / 'analysis_summary.json').exists() for p in paths):
            continue
        matrices = [pd.read_csv(p / 'clustered_location_bh_q_matrix.csv.gz', index_col=0) for p in paths]
        columns = visible_parents(matrices)
        audit = pd.read_csv(paths[0] / 'parent_group_index.csv')
        memberships = pd.read_csv(output / 'cache' / f'tri{level}_parent_child_relationships.csv.gz')
        rows.append({'heatmap_child_level': level, 'parent_motif_level': level-1,
                     'original_parent_motifs': len(audit), 'remaining_parent_motifs': len(columns),
                     'removed_parent_motifs': len(audit)-len(columns),
                     'parents_with_values_bz': int(matrices[0].notna().any().sum()),
                     'parents_with_values_metta': int(matrices[1].notna().any().sum()),
                     'unique_child_motifs_under_remaining_parents': int(memberships.loc[
                         memberships.parent_common_motif_id.isin(columns), 'child_common_motif_id'].nunique())})
    write_csv(pd.DataFrame(rows), output / 'heatmap_parent_counts.csv')


def run_level(level, source, output, statistics, recoding, args):
    cache = output / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    child_path = source / 'cache' / f'tri{level}_common_topologies.csv.gz'
    parent_path = source / 'cache' / f'tri{level-1}_common_topologies.csv.gz'
    pairs, parents = pd.read_csv(child_path), pd.read_csv(parent_path)
    for frame in [pairs, parents]:
        for name in ['common_motif_id', 'bz_wl_hash', 'metabric_wl_hash']:
            if frame[name].duplicated().any():
                raise ValueError(f'Duplicate {name} in input matches')
    settings = {'version': VERSION, 'permutations': args.permutations, 'seed': args.seed,
                'shared_types': NAMES, 'recoding': recoding}
    marker = cache / f'tri{level}_fingerprint.json'
    feature_path = cache / f'tri{level}_intrinsic_criteria.csv.gz'
    relation_path = cache / f'tri{level}_parent_child_relationships.csv.gz'
    engine_path = REPO / 'breast_cancer_survival/motif_results/sibling_topology_analysis_new/run_analysis.py'
    checksum = fingerprint([child_path, parent_path, Path(__file__), engine_path], settings)
    if marker.exists() and json.loads(marker.read_text()).get('fingerprint') == checksum and feature_path.exists() and relation_path.exists():
        log(f'TRI-{level}: using validated graph/criterion cache')
        features, relationships = read_features(feature_path), pd.read_csv(relation_path)
    else:
        for row in parents.to_dict('records'):
            validate_pair(row, recoding)
        lookup = {r['bz_wl_hash']: r for r in parents.to_dict('records')}
        log(f'TRI-{level}: validating {len(pairs):,} common children and computing criteria / parent memberships')
        feature_rows, relations = [], []
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                                 initargs=(lookup, recoding, args.permutations, args.seed)) as pool:
            for i, (feature, edges) in enumerate(pool.map(process_graph, pairs.to_dict('records'), chunksize=50), 1):
                feature_rows.append(feature)
                relations.extend(edges)
                if i % 1000 == 0:
                    log(f'TRI-{level}: {i:,}/{len(pairs):,} graphs processed')
        features = pd.DataFrame(feature_rows)
        relationships = pd.DataFrame(relations, columns=REL_COLUMNS)
        write_csv(features, feature_path)
        write_csv(relationships, relation_path)
        write_json({'fingerprint': checksum, **settings}, marker)
    if relationships.duplicated(['parent_common_motif_id', 'child_common_motif_id']).any():
        raise ValueError('Duplicate parent-child memberships')
    metrics = [c for c in features if c != 'common_motif_id' and not c.startswith('dispersion_')
               and features[c].replace([np.inf, -np.inf], np.nan).nunique() > 1]
    metrics.sort()
    if not metrics or relationships.empty:
        raise ValueError(f'TRI-{level} has no varying criteria or common-parent relationships')
    log(f'TRI-{level}: {len(relationships):,} memberships, {relationships.parent_common_motif_id.nunique():,} parents, {len(metrics)} varying criteria')
    analyses = [cohort_analysis(pairs, parents, relationships, features, metrics, statistics[c], c)
                for c in ['bz', 'metabric']]
    rows, columns = common_order([a[0] for a in analyses])
    summaries = {}
    for cohort, folder, analysis in zip(['BZ', 'METABRIC'], ['bz', 'metta'], analyses):
        matrix, results, audit, children, members = analysis
        destination = output / folder / f'tri{level}'
        destination.mkdir(parents=True, exist_ok=True)
        matrix = matrix.loc[rows, columns]
        audit = annotate_parent_audit(audit, matrix)
        save_heatmap_tables(matrix, audit, destination)
        write_csv(results, destination / 'all_parent_criteria_comparisons.csv.gz')
        write_csv(children, destination / 'child_motif_audit.csv.gz')
        write_csv(members, destination / 'all_parent_child_memberships.csv.gz')
        log(f'TRI-{level} {cohort}: {int(audit.eligible.sum())} eligible parents; {len(results):,} tests; rendering heatmaps')
        pages = render(matrix, audit, cohort, level, destination)
        summary = {'cohort': cohort, 'child_level': level, 'parent_level': level-1,
                   'common_children': len(pairs), 'children_without_common_parent': int((~children.has_common_parent).sum()),
                   'common_parents': len(audit), 'heatmap_parents': len(columns),
                   'parents_removed_from_heatmap': len(audit)-len(columns),
                   'heatmap_parent_filter': 'at_least_one_finite_q_in_either_cohort',
                   'parent_child_memberships': len(members),
                   'eligible_parents': int(audit.eligible.sum()), 'criteria': len(rows),
                   'tests': len(results), 'q_below_0_05': int((results.location_q_bh < .05).sum()),
                   'pdf_pages': pages, 'graph_cache_fingerprint': checksum,
                   'statistics_fingerprint': fingerprint([source / f'{"bz" if folder == "bz" else "metabric"}_common_motif_statistics.csv.gz'], {}),
                   'p_threshold': .05, 'min_per_class': 3, **settings}
        write_json(summary, destination / 'analysis_summary.json')
        summaries[folder] = summary
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--outdir', type=Path, default=DEFAULT_INPUT / 'same_parent_comparsion')
    parser.add_argument('--levels', type=int, nargs='+', default=list(range(2, 8)), choices=range(2, 8))
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument('--permutations', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20260916)
    parser.add_argument('--heatmaps-only', action='store_true',
                        help='Filter and redraw existing paired heatmaps without recomputing features or tests')
    args = parser.parse_args()
    if args.workers < 1 or args.permutations < 2:
        parser.error('workers must be positive and permutations must be at least two')
    if args.heatmaps_only:
        path = args.outdir / 'analysis_summary.json'
        summaries = json.loads(path.read_text()) if path.exists() else {}
        for level in sorted(set(args.levels)):
            summaries[f'tri{level}'] = refresh_heatmaps(level, args.outdir)
            write_json(summaries, path)
        write_parent_counts(args.outdir)
        log('Finished filtering and rendering paired heatmaps.')
        return
    mapping = pd.read_csv(ROOT / 'cell_type_label_mapping.csv', dtype=str)
    bz = mapping.loc[mapping.cohort.eq('bz')].set_index('matched_cell_type').node_label.to_dict()
    recoding = {r.node_label: bz[r.matched_cell_type] for r in mapping.loc[mapping.cohort.eq('metabric')].itertuples()
                if r.matched_cell_type in bz}
    statistics = {c: pd.read_csv(args.input / f'{c}_common_motif_statistics.csv.gz') for c in ['bz', 'metabric']}
    for name, frame in statistics.items():
        if frame.duplicated(['triangle_level', 'wl_hash']).any():
            raise ValueError(f'Duplicate {name} cohort statistics')
    args.outdir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for level in sorted(set(args.levels)):
        summaries[f'tri{level}'] = run_level(level, args.input, args.outdir, statistics, recoding, args)
        write_json(summaries, args.outdir / 'analysis_summary.json')
    write_parent_counts(args.outdir)
    log('Finished all requested common-parent heatmaps.')


if __name__ == '__main__':
    main()
