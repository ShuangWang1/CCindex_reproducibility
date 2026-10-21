#!/usr/bin/env python3
"""Run METABRIC TRI-1--7 rank-normalized survival and expanded lineage analysis."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from breast_cancer_survival_metabric import analyze_motif_survival as raw
from breast_cancer_survival_metabric import rank_methods as methods
from breast_cancer_survival import generate_tri1_motif_frequencies as tri1

VERSION = 1
KEYS = ['triangle_level', 'wl_hash']
EDGE_COLUMNS = ['parent_level', 'parent_hash', 'child_level', 'child_hash',
                'growth_operation', 'changed_node_or_edge']


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def atomic_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    frame.to_csv(temporary, index=False, compression='gzip' if path.suffix == '.gz' else None)
    temporary.replace(path)


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(methods.json_safe(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_csv(path):
    return pd.read_csv(path, dtype={'wl_hash': str}, low_memory=False)


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts-root', type=Path, default=REPO/'breast_cancer_classification_workstation/artifacts/metabric')
    p.add_argument('--clinical', type=Path, default=REPO/'breast_cancer/41586_2019_1007_MOESM7_ESM.txt')
    p.add_argument('--graph-root', type=Path, default=REPO/'breast_cancer/metabric_samples_intra')
    p.add_argument('--motif-root', type=Path, default=REPO/'breast_cancer_meta/metabric_res_threshold_10_cpp_0811')
    p.add_argument('--label-map', type=Path, default=REPO/'breast_cancer/metabric_samples_intra/node_label_cell_type_mapping.csv')
    p.add_argument('--raw-results-root', type=Path, default=here/'results')
    p.add_argument('--outdir', type=Path, default=here/'motif_results')
    p.add_argument('--levels', nargs='+', type=int, default=list(range(1, 8)))
    p.add_argument('--target-levels', nargs='+', type=int, default=list(range(1, 8)))
    p.add_argument('--target-hashes', nargs='+')
    p.add_argument('--max-targets', type=int)
    p.add_argument('--stages', nargs='+', choices=['prepare', 'screen', 'trace', 'extend', 'report'],
                   default=['prepare', 'screen', 'trace', 'extend', 'report'])
    p.add_argument('--chunk-size', type=int, default=5000)
    p.add_argument('--workers', type=int, default=min(6, os.cpu_count() or 1))
    p.add_argument('--p-threshold', type=float, default=.05)
    p.add_argument('--dpi', type=int, default=180)
    p.add_argument('--rerun', action='store_true')
    p.add_argument('--skip-plots', action='store_true')
    return p.parse_args()


def source_signature(args):
    paths = [args.artifacts_root/name for name in ['motif_counts.npz', 'samples.csv',
             'feature_index.csv.gz', 'feature_triangle_levels.npy', 'audit.json']]
    paths += [args.clinical, args.label_map]
    paths += sorted(args.graph_root.glob('sample*.gml'))
    paths += sorted(args.motif_root.glob('sample*/ccindex_tri_*.csv'))
    paths += sorted(args.raw_results_root.glob('tri_*_cox_results_sorted.csv.gz'))
    records = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    data = {'version': VERSION, 'levels': args.levels, 'p_threshold': args.p_threshold,
            'networkx': nx.__version__, 'inputs': records}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


class RelationshipIndex:
    """Index each level transition once, instead of scanning all edges per target."""
    def __init__(self, edges):
        self.frames = {int(level): frame.reset_index(drop=True)
                       for level, frame in edges.groupby('parent_level', sort=True)}
        self.parents = {level: frame.groupby('parent_hash', sort=False).indices
                        for level, frame in self.frames.items()}
        self.children = {level: frame.groupby('child_hash', sort=False).indices
                         for level, frame in self.frames.items()}

    def lookup(self, level, hashes, by_parent):
        frame = self.frames.get(level)
        if frame is None:
            return pd.DataFrame(columns=EDGE_COLUMNS)
        groups = (self.parents if by_parent else self.children)[level]
        positions = [groups[h] for h in hashes if h in groups]
        return frame.iloc[np.unique(np.concatenate(positions))] if positions else frame.iloc[:0]

    def select(self, level, target_hash, levels):
        selected = {int(level): {target_hash}}
        for child in range(int(level), min(levels), -1):
            selected[child-1] = set(self.lookup(child-1, selected.get(child, set()), False).parent_hash)
        for parent in range(int(level), max(levels)):
            selected[parent+1] = set(self.lookup(parent, selected.get(parent, set()), True).child_hash)
        return selected

    def subset(self, selected):
        parts = []
        for level, hashes in sorted(selected.items()):
            frame = self.lookup(level, hashes, True)
            parts.append(frame.loc[frame.child_hash.isin(selected.get(level+1, set()))])
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=EDGE_COLUMNS)


def structural_trace(index, topologies, target, levels):
    selected = index.select(int(target.triangle_level), target.wl_hash, levels)
    parts = [topologies[level].loc[sorted(hashes)] for level, hashes in sorted(selected.items()) if hashes]
    nodes = pd.concat(parts, ignore_index=True)
    nodes['lineage_role'] = np.where(nodes.triangle_level.eq(target.triangle_level), 'target',
                                    np.where(nodes.triangle_level.lt(target.triangle_level), 'ancestor', 'descendant'))
    return nodes, index.subset(selected)


class Pipeline:
    def __init__(self, args):
        self.a = args
        self.root = args.outdir
        self.cache = self.root/'cache'
        self.rankroot = self.root/'normalized_by_ranking_motif_analysis'
        self.traceroot = self.root/'significant_mtf_trace_ranking_normalized'
        self.signature = source_signature(args)
        signature_path = self.cache/'input_signature.json'
        if signature_path.exists():
            saved = json.loads(signature_path.read_text())
            if saved['signature'] != self.signature:
                if not args.rerun:
                    raise ValueError('Inputs/settings changed; use a new --outdir or --rerun to rebuild generated outputs')
                for directory in [self.cache, self.rankroot, self.traceroot]:
                    if directory.exists():
                        shutil.rmtree(directory)
        self.cache.mkdir(parents=True, exist_ok=True)
        atomic_json({'signature': self.signature, 'version': VERSION}, signature_path)
        self.samples = pd.read_csv(args.artifacts_root/'samples.csv')
        clinical = pd.read_csv(args.clinical, sep='\t', low_memory=False)
        self.patients, self.sum_agg, self.linkage, self.audit = raw.prepare_cohort(self.samples, clinical)
        self.mean_agg = sparse.diags(1 / self.patients.n_images.to_numpy()) @ self.sum_agg
        labels = pd.read_csv(args.label_map, dtype={'node_label': str})
        if labels.node_label.duplicated().any():
            raise ValueError('Duplicate cell-type labels')
        self.labels = dict(labels[['node_label', 'cell_type']].itertuples(index=False, name=None))
        self.palette = {name: methods.plt.get_cmap('tab10')(i) for i, (_, name) in
                        enumerate(sorted(self.labels.items(), key=lambda x: int(x[0])))}
        self.time = self.patients.time_days.to_numpy(float)
        self.event = self.patients.event.to_numpy(int)
        self._features = {}
        self._scores = {}
        self._raw = {}

    def leveldir(self, level):
        return self.cache/f'tri{level}'

    def features(self, level):
        if level not in self._features:
            self._features[level] = read_csv(self.leveldir(level)/'features.csv.gz')
        return self._features[level]

    def scores(self, level):
        if level not in self._scores:
            self._scores[level] = sparse.load_npz(self.leveldir(level)/'patient_rank_scores.npz').tocsc()
        return self._scores[level]

    def raw_results(self, level):
        if level not in self._raw:
            if level == 1:
                self._raw[level] = read_csv(self.leveldir(level)/'raw_cox_results.csv.gz')
            else:
                frame = read_csv(self.a.raw_results_root/f'tri_{level}_cox_results_sorted.csv.gz')
                mapping = {'median_patient_count': 'median_patient_raw_count',
                           'image_document_frequency': 'sample_document_frequency',
                           'n_patients': 'n_unadjusted', 'n_deaths': 'events_unadjusted'}
                for name in ['coef','hazard_ratio','ci95_low','ci95_high','p_value','n_enriched',
                             'events_enriched','n_non_enriched','events_non_enriched','fit_status',
                             'q_value_bh_global','q_value_bh_within_level','rank_global','rank_within_level']:
                    mapping[name] = name + '_unadjusted'
                frame = frame.rename(columns=mapping)
                self._raw[level] = frame
        return self._raw[level]

    def prepare(self):
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_csv(self.patients, self.root/'patient_level_survival.csv')
        atomic_csv(self.linkage, self.root/'patient_linkage_audit.csv')
        atomic_json(self.audit, self.root/'cohort_audit.json')
        features = read_csv(self.a.artifacts_root/'feature_index.csv.gz')
        counts = sparse.load_npz(self.a.artifacts_root/'motif_counts.npz').tocsr()
        levels = np.load(self.a.artifacts_root/'feature_triangle_levels.npy')
        if counts.shape != (len(self.samples), len(features)) or not np.array_equal(levels, features.triangle_level):
            raise ValueError('Feature levels, matrix, and inventories disagree')
        if not np.array_equal(features.feature_index, np.arange(len(features))):
            raise ValueError('Feature indices are not in matrix order')
        if features.duplicated(KEYS).any() or not np.isfinite(counts.data).all() or (counts.data < 0).any():
            raise ValueError('Invalid motif identities or counts')
        source_audit = json.loads((self.a.artifacts_root/'audit.json').read_text())
        for relative in source_audit.get('search_exhausted_missing_level_files', []):
            if (self.a.motif_root/relative).exists():
                raise ValueError(f'Artifact audit is outdated for {relative}')
        atomic_json(source_audit, self.cache/'source_artifact_audit.json')
        for level in self.a.levels:
            directory = self.leveldir(level)
            if (directory/'prepared.json').exists() and not self.a.rerun:
                log(f'TRI-{level}: reused prepared rank matrices')
                continue
            directory.mkdir(parents=True, exist_ok=True)
            if level == 1:
                image_counts, feat = self.prepare_tri1()
            else:
                mask = features.triangle_level.eq(level).to_numpy()
                feat = features.loc[mask].copy().reset_index(drop=True)
                feat = feat.rename(columns={'feature_index': 'source_feature_index'})
                feat.insert(0, 'feature_index', np.arange(len(feat)))
                image_counts = counts[:, mask]
            feat['cell_type_composition'] = feat.representative_node_labels.map(lambda x: raw.cell_type_composition(x, self.labels))
            image_scores = methods.rank_score_sparse(image_counts, len(feat))
            patient_scores = (self.mean_agg @ image_scores).tocsr()
            patient_counts = (self.sum_agg @ image_counts).tocsr()
            atomic_csv(feat, directory/'features.csv.gz')
            sparse.save_npz(directory/'image_counts.npz', image_counts)
            sparse.save_npz(directory/'image_rank_scores.npz', image_scores)
            sparse.save_npz(directory/'patient_rank_scores.npz', patient_scores)
            sparse.save_npz(directory/'patient_raw_counts.npz', patient_counts)
            if level == 1:
                values = patient_counts.toarray()
                median = np.median(values, axis=0)
                fitted = raw.vectorized_binary_cox(values > median, self.time, self.event)
                result = feat.copy()
                result['median_patient_raw_count'] = median
                for key in ['coef','hazard_ratio','ci95_low','ci95_high','p_value','n_enriched',
                            'events_enriched','n_non_enriched','events_non_enriched']:
                    result[key+'_unadjusted'] = fitted[key]
                result['fit_status_unadjusted'] = [raw.STATUS_LABELS[int(x)] for x in fitted['status_code']]
                atomic_csv(result, directory/'raw_cox_results.csv.gz')
            atomic_json({'vocabulary_size': len(feat), 'images': len(self.samples),
                         'score_formula': '(M-rank)/(M-1)', 'patient_aggregation': 'equal_image_mean'}, directory/'prepared.json')
            log(f'TRI-{level}: ranked {len(feat):,} motifs across {len(self.samples)} images')

    def prepare_tri1(self):
        directory = self.leveldir(1)/'image_counts'
        directory.mkdir(exist_ok=True)
        inventory = []
        for i, sample in enumerate(self.samples.itertuples(index=False)):
            destination = directory/f'{sample.sample}.csv'
            if not destination.exists() or self.a.rerun:
                coords, labels = tri1.load_sample_nodes(self.a.graph_root/f'{sample.sample}.gml')
                if set(labels) - set(self.labels):
                    raise ValueError(f'Unknown labels in {sample.sample}')
                frequencies, n = tri1.delaunay_triangle_frequencies(coords, labels)
                rows = tri1.result_rows(frequencies)
                if sum(row['matches'] for row in rows) != n:
                    raise ValueError('TRI-1 triangle totals disagree')
                tri1.write_result(destination, rows)
            frame = read_csv(destination)
            inventory.append(frame)
            if (i+1) % 100 == 0:
                log(f'TRI-1: counted {i+1}/{len(self.samples)} source graphs')
        metadata = pd.concat(inventory, ignore_index=True).drop_duplicates('wl_hash').sort_values('wl_hash').reset_index(drop=True)
        metadata = metadata[['wl_hash','node_labels','edge_list']].rename(columns={'node_labels': 'representative_node_labels'})
        metadata.insert(0, 'feature_index', np.arange(len(metadata)))
        metadata['triangle_level'], metadata['nodes'], metadata['edges'] = 1, 3, 3
        lookup = metadata.set_index('wl_hash').feature_index
        rows, cols, values = [], [], []
        manifest = []
        for i, frame in enumerate(inventory):
            rows.extend([i]*len(frame)); cols.extend(lookup.loc[frame.wl_hash].tolist()); values.extend(frame.matches.tolist())
            manifest.append({'sample': self.samples.iloc[i]['sample'], 'delaunay_triangles': int(frame.matches.sum()), 'motif_types': len(frame)})
        counts = sparse.csr_matrix((values,(rows,cols)),shape=(len(self.samples),len(metadata)))
        metadata['document_frequency'] = np.asarray((counts > 0).sum(axis=0)).ravel()
        atomic_csv(pd.DataFrame(manifest), self.leveldir(1)/'triangle_count_manifest.csv')
        return counts, metadata

    def fit_rank(self, level, columns, kind, positive_only=False):
        feat = self.features(level)
        columns = np.asarray(columns, int)
        score_matrix = self.scores(level)
        parts = []
        identity = hashlib.sha256(columns.tobytes()+str(self.a.chunk_size).encode()).hexdigest()[:16]
        checkpoints = self.leveldir(level)/f'{kind}_{identity}'
        checkpoints.mkdir(exist_ok=True)
        batch_starts = list(range(0, len(columns), self.a.chunk_size))
        if getattr(self.a, 'reverse_batches', False):
            batch_starts.reverse()
        for start in batch_starts:
            chosen = columns[start:start+self.a.chunk_size]
            destination = checkpoints/f'{start}.csv.gz'
            if destination.exists() and not self.a.rerun:
                parts.append(read_csv(destination)); continue
            scores = score_matrix[:, chosen].toarray()
            medians = np.median(scores, axis=0)
            if positive_only:
                keep = medians > 0
                chosen, scores, medians = chosen[keep], scores[:,keep], medians[keep]
            fitted = raw.vectorized_binary_cox(scores > medians, self.time, self.event) if len(chosen) else None
            part = feat.iloc[chosen].copy()
            part['vocabulary_size'] = len(feat)
            part['median_patient_rank_score'] = medians
            part['rank_group_definition'] = 'patient_rank_score > cohort_median'
            part['n_survival'], part['events_survival'] = len(self.time), int(self.event.sum())
            if fitted is not None:
                mapping = {'n_enriched':'n_high_rank','events_enriched':'events_high_rank',
                           'n_non_enriched':'n_low_rank','events_non_enriched':'events_low_rank'}
                for key in ['coef','hazard_ratio','ci95_low','ci95_high','p_value']:
                    part[key+'_rank_binary'] = fitted[key]
                for key, name in mapping.items(): part[name] = fitted[key]
                part['rank_binary_fit_status'] = [raw.STATUS_LABELS[int(x)] for x in fitted['status_code']]
                part['rank_score_mean'] = scores.mean(axis=0)
                part['rank_score_sd'] = scores.std(axis=0, ddof=1)
            else:
                for name in ['coef_rank_binary','hazard_ratio_rank_binary','ci95_low_rank_binary','ci95_high_rank_binary',
                             'p_value_rank_binary','n_high_rank','events_high_rank','n_low_rank','events_low_rank',
                             'rank_binary_fit_status','rank_score_mean','rank_score_sd']:
                    part[name] = pd.Series(dtype=float if name != 'rank_binary_fit_status' else str)
            raw_frame = self.raw_results(level)
            raw_columns = [x for x in raw_frame if x.endswith('_unadjusted') or x == 'median_patient_raw_count']
            part = part.merge(raw_frame[KEYS+raw_columns], on=KEYS, how='left', validate='one_to_one')
            atomic_csv(part, destination)
            parts.append(part)
            if (start//self.a.chunk_size) % 10 == 0 or start+len(columns[start:start+self.a.chunk_size]) == len(columns):
                log(f'TRI-{level}: {kind} {min(start+self.a.chunk_size,len(columns)):,}/{len(columns):,}; retained {sum(len(x) for x in parts):,}')
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=KEYS+['p_value_rank_binary','median_patient_rank_score'])

    def screen(self):
        self.rankroot.mkdir(parents=True, exist_ok=True)
        parts = []
        for level in self.a.levels:
            # Raw-positive prevalence is a safe superset, followed by a direct rank-median check.
            patient_counts = sparse.load_npz(self.leveldir(level)/'patient_raw_counts.npz').tocsc()
            columns = np.flatnonzero(np.asarray((patient_counts > 0).sum(axis=0)).ravel() >= (len(self.time)+1)//2)
            parts.append(self.fit_rank(level, columns, 'primary', positive_only=True))
        results = pd.concat(parts, ignore_index=True)
        results['q_value_bh_global_rank_binary'] = raw.benjamini_hochberg(results.p_value_rank_binary.to_numpy())
        results['q_value_bh_within_level_rank_binary'] = results.groupby('triangle_level').p_value_rank_binary.transform(lambda x: raw.benjamini_hochberg(x.to_numpy()))
        results['rank_global_rank_binary'] = results.p_value_rank_binary.rank(method='min').astype('Int64')
        results['rank_within_level_rank_binary'] = results.groupby('triangle_level').p_value_rank_binary.rank(method='min').astype('Int64')
        results = results.sort_values(['p_value_rank_binary',*KEYS],kind='stable').reset_index(drop=True)
        significant = results.loc[np.isfinite(results.p_value_rank_binary)&results.p_value_rank_binary.lt(self.a.p_threshold)].copy()
        significant['motif_directory'] = [f'tri{int(r.triangle_level)}/{int(r.rank_within_level_rank_binary):03d}_{r.wl_hash}' for r in significant.itertuples()]
        significant_dir = self.rankroot/'significant'
        significant_dir.mkdir(exist_ok=True)
        atomic_csv(results, significant_dir/'all_nonzero_threshold_rank_results.csv.gz')
        atomic_csv(significant, significant_dir/'significant_rank_normalized_survival_results.csv')
        atomic_json({**self.audit, 'candidate_motifs':len(results), 'candidate_motifs_by_level': results.groupby('triangle_level').size().to_dict(),
                     'total_significant_motifs':len(significant), 'significant_motifs_by_level':significant.groupby('triangle_level').size().to_dict(),
                     'selection':'finite rank binary Cox p < threshold and patient rank median > 0', 'p_threshold':self.a.p_threshold,
                     'within_level_bh_q_below_0_05':int(results.q_value_bh_within_level_rank_binary.lt(.05).sum()),
                     'global_bh_q_below_0_05':int(results.q_value_bh_global_rank_binary.lt(.05).sum())}, significant_dir/'significant_summary.json')
        log(f'Screened {len(results):,} positive-median motifs; selected {len(significant):,} targets')

    def targets(self):
        table = read_csv(self.rankroot/'significant/significant_rank_normalized_survival_results.csv')
        table = table.loc[table.triangle_level.isin(self.a.target_levels)]
        if self.a.target_hashes:
            missing = set(self.a.target_hashes)-set(table.wl_hash)
            if missing: raise ValueError(f'Hashes not in selected target set: {sorted(missing)}')
            table = table.loc[table.wl_hash.isin(self.a.target_hashes)]
        table = table.sort_values(['p_value_rank_binary',*KEYS],kind='stable')
        if self.a.max_targets is not None: table = table.head(self.a.max_targets)
        return table.reset_index(drop=True)

    def recover_topologies(self):
        paths = []
        for level in self.a.levels:
            destination = self.leveldir(level)/'topologies.csv.gz'
            paths.append(destination)
            if destination.exists() and not self.a.rerun: continue
            feat = self.features(level).copy()
            if level == 1:
                feat['node_labels'] = feat.representative_node_labels
            else:
                counts = sparse.load_npz(self.leveldir(level)/'image_counts.npz').tocsc()
                if (np.diff(counts.indptr) == 0).any(): raise ValueError('Observed vocabulary contains an absent motif')
                first_rows = counts.indices[counts.indptr[:-1]]
                feat['node_labels'], feat['edge_list'] = '', ''
                for i, sample_row in enumerate(np.unique(first_rows)):
                    columns = np.flatnonzero(first_rows == sample_row)
                    sample = self.samples.iloc[int(sample_row)]['sample']
                    frame = pd.read_csv(self.a.motif_root/sample/f'ccindex_tri_{level}.csv',
                                        usecols=['wl_hash','node_labels','edge_list'],dtype=str)
                    found = frame.drop_duplicates('wl_hash').set_index('wl_hash').loc[feat.iloc[columns].wl_hash]
                    feat.loc[columns,'node_labels'] = found.node_labels.to_numpy()
                    feat.loc[columns,'edge_list'] = found.edge_list.to_numpy()
                    if (i+1)%100==0: log(f'TRI-{level}: recovered topologies from {i+1} source files')
            if feat.node_labels.eq('').any() or feat.edge_list.eq('').any(): raise ValueError('Incomplete topology recovery')
            atomic_csv(feat, destination)
            log(f'TRI-{level}: recovered {len(feat):,} topologies')
        return paths

    def build_relationships(self):
        self.recover_topologies()
        outputs = []
        for level in self.a.levels:
            if level == min(self.a.levels): continue
            destination = self.leveldir(level)/'parent_relationships.csv.gz'
            outputs.append(destination)
            if destination.exists() and not self.a.rerun: continue
            frame = read_csv(self.leveldir(level)/'topologies.csv.gz')
            valid_parents = set(self.features(level-1).wl_hash)
            records = frame[['wl_hash','node_labels','edge_list']].itertuples(index=False,name=None)
            edges = []
            with ProcessPoolExecutor(max_workers=self.a.workers) as pool:
                for i, relations in enumerate(pool.map(parent_records, records, chunksize=500),start=1):
                    for parent_hash, child_hash, operation, changed in relations:
                        if parent_hash in valid_parents:
                            edges.append((level-1,parent_hash,level,child_hash,operation,changed))
                    if i%25000 == 0: log(f'TRI-{level}: indexed parents for {i:,}/{len(frame):,} motifs')
            atomic_csv(pd.DataFrame(edges,columns=EDGE_COLUMNS).drop_duplicates(EDGE_COLUMNS[:4]), destination)
            log(f'TRI-{level}: saved {len(edges):,} parent relationships')
        return pd.concat([read_csv(p) for p in outputs],ignore_index=True) if outputs else pd.DataFrame(columns=EDGE_COLUMNS)

    def selected_nodes(self, edges, level, target_hash):
        selected = {int(level):{target_hash}}
        for child in range(int(level), min(self.a.levels), -1):
            sub = edges.loc[edges.child_level.eq(child)&edges.child_hash.isin(selected.get(child,set()))]
            selected[child-1] = set(sub.parent_hash)
        for parent in range(int(level), max(self.a.levels)):
            sub = edges.loc[edges.parent_level.eq(parent)&edges.parent_hash.isin(selected.get(parent,set()))]
            selected[parent+1] = set(sub.child_hash)
        return selected

    def trace(self):
        self.traceroot.mkdir(parents=True,exist_ok=True)
        targets = self.targets()
        atomic_csv(targets,self.traceroot/'potential_motif_pool.csv')
        edges = self.build_relationships() if len(targets) else pd.DataFrame(columns=EDGE_COLUMNS)
        index = RelationshipIndex(edges)
        topologies = {level: read_csv(self.leveldir(level)/'topologies.csv.gz').set_index('wl_hash', drop=False) for level in self.a.levels} if len(targets) else {}
        base = read_csv(self.rankroot/'significant/all_nonzero_threshold_rank_results.csv.gz')
        records, failures = [], []
        for i, target in enumerate(targets.itertuples(index=False), start=1):
            folder = self.traceroot/f'tri{target.triangle_level}'/target.wl_hash
            try:
                if not (folder/'trace_metadata.json').exists() or self.a.rerun:
                    nodes, target_edges = structural_trace(index, topologies, target, self.a.levels)
                    rank_columns = [c for c in base if c not in nodes or c in KEYS]
                    nodes = nodes.merge(base[rank_columns],on=KEYS,how='left',validate='one_to_one')
                    self.write_trace(folder,nodes,target_edges,target,expanded=False)
                metadata = json.loads((folder/'trace_metadata.json').read_text())
                records.append(metadata)
            except Exception as exc:
                records.append({'target_level':target.triangle_level,'target_hash':target.wl_hash,'status':'failed','error':f'{type(exc).__name__}: {exc}'})
                failures.append(target.wl_hash)
            if i%25==0 or i==len(targets): log(f'Traced {i}/{len(targets)} targets')
        atomic_csv(pd.DataFrame(records),self.traceroot/'trace_run_manifest.csv')
        atomic_json({'total_targets':len(targets),'targets_by_level':targets.groupby('triangle_level').size().to_dict(),
                     'selection':'positive patient median rank score; finite binary rank Cox p < threshold',
                     'p_threshold':self.a.p_threshold,'limited_run':self.a.max_targets is not None or self.a.target_hashes is not None},self.traceroot/'potential_motif_pool_summary.json')
        if failures: raise RuntimeError(f'{len(failures)} trace failures; inspect trace_run_manifest.csv')

    def write_trace(self, folder, nodes, edges, target, expanded):
        folder.mkdir(parents=True,exist_ok=True)
        nodes['rank_result_available'] = nodes.rank_binary_fit_status.notna()
        nodes['significant_rank_p'] = np.isfinite(nodes.p_value_rank_binary)&nodes.p_value_rank_binary.lt(self.a.p_threshold)
        nodes['positive_rank_median'] = nodes.median_patient_rank_score.gt(0)
        nodes['selected_focal_rule'] = nodes.significant_rank_p&nodes.positive_rank_median
        nodes = nodes.sort_values(['triangle_level','p_value_rank_binary','wl_hash'],na_position='last',kind='stable')
        metadata = {'status':'complete','target_level':int(target.triangle_level),'target_hash':target.wl_hash,
                    'target_p_value_rank_binary':float(target.p_value_rank_binary),'target_median_patient_rank_score':float(target.median_patient_rank_score),
                    'motifs_by_level':nodes.groupby('triangle_level').size().to_dict(),'structural_nodes':len(nodes),
                    'motifs_with_rank_results':int(nodes.rank_result_available.sum()),'finite_p_values':int(np.isfinite(nodes.p_value_rank_binary).sum()),
                    'potential_parent_child_edges':len(edges),'package_variant':'expanded_all_structural_rank_results' if expanded else 'primary_screen_rank_results',
                    'input_signature':self.signature,'p_threshold':self.a.p_threshold,'plots_written':not self.a.skip_plots,
                    'relationship_definition':'Potential labeled-topological one-triangle growth; original parent IDs and extendable-edge state were not persisted',
                    'rank_statistics_coverage':'every structural relative, including zero medians' if expanded else 'positive-median primary screen',
                    'reproduction_command':'python breast_cancer_survival_metabric/run_rank_normalized_analysis.py'}
        atomic_csv(nodes,folder/'trace_nodes.csv')
        atomic_csv(edges,folder/'trace_edges.csv')
        if not self.a.skip_plots:
            methods.plot_rank_p_lineage(nodes,edges,int(target.triangle_level),target.wl_hash,
                                       folder/'rank_normalized_cox_p_value_by_level.png',self.a.p_threshold)
        atomic_json(metadata,folder/'trace_metadata.json')

    def extend(self):
        targets = self.targets()
        base = read_csv(self.rankroot/'significant/all_nonzero_threshold_rank_results.csv.gz')
        wanted = {level:set() for level in self.a.levels}
        for target in targets.itertuples(index=False):
            path = self.traceroot/f'tri{target.triangle_level}'/target.wl_hash/'trace_nodes.csv'
            nodes = pd.read_csv(path,usecols=KEYS,dtype={'wl_hash':str})
            for level, group in nodes.groupby('triangle_level'): wanted[int(level)].update(group.wl_hash)
        parts = []
        for level in self.a.levels:
            missing = wanted[level]-set(base.loc[base.triangle_level.eq(level),'wl_hash'])
            feat = self.features(level)
            columns = feat.loc[feat.wl_hash.isin(missing),'feature_index'].to_numpy(int)
            if len(columns): parts.append(self.fit_rank(level,columns,'structural'))
        extended = pd.concat(parts,ignore_index=True) if parts else base.iloc[:0].copy()
        atomic_csv(extended,self.traceroot/'all_structural_relative_rank_results.csv.gz')
        combined = pd.concat([base,extended],ignore_index=True)
        if combined.duplicated(KEYS).any(): raise ValueError('Duplicate rank results')
        atomic_csv(combined,self.traceroot/'combined_rank_results.csv.gz')
        manifest = []
        for i,target in enumerate(targets.itertuples(index=False),start=1):
            source = self.traceroot/f'tri{target.triangle_level}'/target.wl_hash
            destination = source.with_name(source.name+'_new')
            if (destination/'trace_metadata.json').exists() and not self.a.rerun:
                metadata = json.loads((destination/'trace_metadata.json').read_text())
            else:
                nodes = read_csv(source/'trace_nodes.csv')
                retained = [c for c in KEYS+['lineage_role','node_labels','edge_list','nodes','edges','cell_type_composition','representative_node_labels','feature_index','source_feature_index','document_frequency'] if c in nodes]
                additions = [c for c in combined if c not in retained or c in KEYS]
                nodes = nodes[retained].merge(combined[additions],on=KEYS,how='left',validate='one_to_one')
                if nodes.rank_binary_fit_status.isna().any(): raise ValueError('A structural relative lacks an analysis result')
                edges = read_csv(source/'trace_edges.csv')
                self.write_trace(destination,nodes,edges,target,expanded=True)
                metadata = json.loads((destination/'trace_metadata.json').read_text())
            manifest.append(metadata)
            if i%25==0 or i==len(targets): log(f'Expanded {i}/{len(targets)} target packages')
        atomic_csv(pd.DataFrame(manifest),self.traceroot/'expanded_trace_run_manifest.csv')
        atomic_json({'additional_motifs':len(extended),'additional_motifs_by_level':extended.groupby('triangle_level').size().to_dict(),
                     'finite_binary_cox_p_values':int(np.isfinite(extended.p_value_rank_binary).sum()),
                     'undefined_binary_cox_p_values':int((~np.isfinite(extended.p_value_rank_binary)).sum()),
                     'zero_median_motifs':int(extended.median_patient_rank_score.eq(0).sum()),
                     'updated_trace_packages':len(manifest)},self.traceroot/'all_structural_relative_rank_results_summary.json')

    def packages(self):
        targets = self.targets()
        topology = {level:read_csv(self.leveldir(level)/'topologies.csv.gz').set_index('wl_hash') for level in targets.triangle_level.unique()}
        for i,target in enumerate(targets.itertuples(index=False),start=1):
            level = int(target.triangle_level)
            folder = self.rankroot/'significant'/target.motif_directory
            if (folder/'statistics.json').exists() and not self.a.rerun: continue
            folder.mkdir(parents=True,exist_ok=True)
            feature = self.features(level).set_index('wl_hash').loc[target.wl_hash]
            col = int(feature.feature_index)
            score = self.scores(level)[:,col].toarray().ravel()
            counts = sparse.load_npz(self.leveldir(level)/'patient_raw_counts.npz')[:,col].toarray().ravel()
            result = target._asdict()
            secondary, high, _ = methods.analyze_rank_survival(score,self.time,self.event)
            # Retain primary vectorized estimates; add only secondary statistics.
            result.update({k:v for k,v in secondary.items() if k not in result})
            result.update({'time_unit':'days','patient_image_aggregation':'equal_weight_mean','core_rank_tie_method':'average_for_equal_positive_counts'})
            data = self.patients.copy()
            data['triangle_level'],data['wl_hash'] = level,target.wl_hash
            data['number_of_images'] = data.n_images
            data['summed_raw_count'],data['rank_score'] = counts,score
            data['mean_image_rank'] = len(self.features(level))-score*(len(self.features(level))-1)
            data['median_rank_score'] = target.median_patient_rank_score
            data['rank_group'] = np.where(high,'High rank score','Low rank score')
            atomic_csv(data,folder/'patient_rank_scores.csv')
            if not self.a.skip_plots:
                methods.plot_km(data,pd.Series(result),folder/'rank_normalized_survival_km.png',self.a.dpi)
                methods.plot_distribution(data,pd.Series(result),folder/'patient_rank_distribution.png',self.a.dpi)
                row = topology[level].loc[target.wl_hash].copy()
                row['wl_hash'] = target.wl_hash
                plot_topology(row,self.labels,self.palette,folder/'motif_delaunay_topology.png',self.a.dpi)
            atomic_json(result,folder/'statistics.json')
            if i%25==0 or i==len(targets): log(f'Wrote survival/topology packages {i}/{len(targets)}')

    def report(self):
        self.packages()
        screen = json.loads((self.rankroot/'significant/significant_summary.json').read_text())
        expanded_path = self.traceroot/'expanded_trace_run_manifest.csv'
        manifest = read_csv(expanded_path) if expanded_path.exists() and expanded_path.stat().st_size > 1 else pd.DataFrame()
        summary = {**screen,'triangle_levels':self.a.levels,'target_levels':self.a.target_levels,
                   'expanded_trace_packages':len(manifest),'input_signature':self.signature,'time_unit':'days',
                   'vocabulary_sizes':{level:len(self.features(level)) for level in self.a.levels},
                   'limited_run':self.a.max_targets is not None or self.a.target_hashes is not None,
                   'endpoint':'overall survival: T (days), Death (all-cause death)',
                   'plots_written':not self.a.skip_plots}
        atomic_json(summary,self.root/'analysis_summary.json')
        lines = ['# METABRIC rank-normalized motif survival and lineage analysis','',
                 f"Cohort: **{len(self.patients)} patients**, **{int(self.event.sum())} deaths**, {self.sum_agg.nnz} included images.",
                 f"Screened **{screen['candidate_motifs']:,}** positive-median motifs; selected **{screen['total_significant_motifs']:,}** at nominal rank-normalized Cox p < {self.a.p_threshold}.",
                 f"Completed **{len(manifest)} expanded lineage packages**.",'',
                 '## Method','',
                 'Within each image and triangle level, positive motif counts are ranked descending with average ties; absent motifs receive rank M. Scores are (M-rank)/(M-1), using the complete observed 515-image level vocabulary. Repeated-image scores are averaged equally within patients. Raw counts are separately summed.',
                 'High rank means strictly above the patient median; ties are low. The primary model is unadjusted Cox with Efron ties and a two-sided Wald p-value. Overall survival uses T in days and Death. BH q-values are calculated over the positive-median screening family. Log-rank and continuous per-SD Cox statistics are secondary.', '',
                 '## Results by triangle level','',
                 '| TRI | Vocabulary | Screened | Nominal p < threshold |','|---:|---:|---:|---:|']
        for level in self.a.levels:
            lines.append(f"| {level} | {len(self.features(level)):,} | {screen['candidate_motifs_by_level'].get(str(level),0):,} | {screen['significant_motifs_by_level'].get(str(level),0):,} |")
        lines += ['',f"Global BH q < 0.05: **{screen['global_bh_q_below_0_05']}**.",'',
                  '## Outputs','',
                  '`normalized_by_ranking_motif_analysis/significant/` contains the complete primary screen, selected results and per-motif survival, patient-score and topology packages. Topology positions preserve adjacency and do not represent tissue coordinates.',
                  '`significant_mtf_trace_ranking_normalized/triN/<hash>_new/` contains the final expanded lineage plot, nodes, edges and metadata. All structural relatives have been analyzed, including zero-median motifs. Undefined Cox p-values retain explicit fit statuses in CSV; plot edges join estimable endpoints only. A p-value that underflows to zero is plotted at 1e-300 without changing the table.',
                  'Relationships are potential labeled-topological one-triangle growth relations, not recorded search histories. Target selection uses nominal exploratory p-values, not BH-adjusted significance. Clinical covariates are not included.', '',
                  '## Reproduction','', '```bash','python breast_cancer_survival_metabric/run_rank_normalized_analysis.py','```','',
                  'Runs resume from input-validated caches. Use --stages to run selected stages, --max-targets for a smoke run, --target-levels or --target-hashes for target subsets, and --rerun to regenerate outputs. Subset runs use a separate --outdir. Existing BZ and raw METABRIC artifacts are preserved.','']
        (self.root/'REPORT.md').write_text('\n'.join(lines))
        log(f'Completed report: {self.root / "REPORT.md"}')


def graph_from_encoding(node_labels, edge_list):
    graph = nx.Graph()
    graph.add_nodes_from((i, {'label':x}) for i,x in enumerate(node_labels.split('&')))
    graph.add_edges_from(tuple(map(int,x.split('-'))) for x in edge_list.split(';'))
    return graph


def parent_records(record):
    child_hash, labels, edge_list = record
    graph = graph_from_encoding(labels,edge_list)
    if methods.motif_hash(graph) != child_hash:
        raise ValueError(f'Topology does not reproduce saved WL hash: {child_hash}')
    results = {}
    for operation,changed,parent in methods.potential_direct_parents(graph):
        parent_hash = methods.motif_hash(parent)
        results.setdefault(parent_hash,(parent_hash,child_hash,operation.replace('remove_','add_'),changed))
    return list(results.values())


def plot_topology(row, labels, palette, output, dpi):
    graph = graph_from_encoding(row.node_labels,row.edge_list)
    if not nx.check_planarity(graph)[0]: raise ValueError('Non-planar motif topology')
    fig,ax = methods.plt.subplots(figsize=(7.2,6.4))
    names = [labels[graph.nodes[n]['label']] for n in graph]
    nx.draw_networkx(graph,pos=nx.planar_layout(graph),ax=ax,with_labels=False,
                     node_color=[palette[x] for x in names],node_size=920,edgecolors='white',linewidths=2,
                     edge_color='#5F6368',width=2)
    handles = [methods.plt.Line2D([0],[0],marker='o',linestyle='none',color=palette[x],label=x) for x in palette if x in names]
    ax.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,-.12),ncol=min(3,len(handles)),frameon=False)
    ax.set_title(f'TRI-{int(row.triangle_level)} {row.wl_hash[:8]}\n{row.cell_type_composition}',fontsize=12)
    ax.set_axis_off(); fig.tight_layout(rect=(0,.06,1,1)); fig.savefig(output,dpi=dpi); methods.plt.close(fig)


def main():
    args = parse_args()
    args.levels = sorted(set(args.levels))
    if not args.levels or args.levels != list(range(min(args.levels),max(args.levels)+1)) or min(args.levels)<1 or max(args.levels)>7:
        raise ValueError('--levels must be a contiguous subset of 1 through 7')
    if args.chunk_size<1 or args.workers<1 or (args.max_targets is not None and args.max_targets<1):
        raise ValueError('Chunk size, workers and target limits must be positive')
    if not 0<args.p_threshold<1: raise ValueError('p threshold must be between 0 and 1')
    pipeline = Pipeline(args)
    log(f'METABRIC: {len(pipeline.patients)} patients, {int(pipeline.event.sum())} deaths')
    for stage in args.stages:
        log(f'Starting {stage}')
        getattr(pipeline,stage)()
    log('Requested stages complete')


if __name__ == '__main__':
    main()
