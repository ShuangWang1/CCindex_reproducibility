#!/usr/bin/env python3
"""Plot parents with at least one finite criterion in each cohort."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import compare_common_same_parent as analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=analysis.DEFAULT_INPUT / 'same_parent_comparsion')
    parser.add_argument('--levels', type=int, nargs='+', default=list(range(2, 8)), choices=range(2, 8))
    args = parser.parse_args()
    counts = []
    for level in sorted(set(args.levels)):
        paths = [args.root / cohort / f'tri{level}' for cohort in ['bz', 'metta']]
        matrices = [pd.read_csv(p / 'clustered_location_bh_q_matrix.csv.gz', index_col=0,
                                float_precision='round_trip') for p in paths]
        left, right = matrices
        if not left.index.equals(right.index) or not left.columns.equals(right.columns):
            raise ValueError(f'TRI-{level}: paired source axes differ')
        keep = np.isfinite(left.to_numpy(float)).any(axis=0) & np.isfinite(right.to_numpy(float)).any(axis=0)
        columns = left.columns[keep].tolist()
        for cohort, path, matrix in zip(['BZ', 'METABRIC'], paths, matrices):
            matrix = matrix.loc[:, columns]
            title = (f'{cohort}: overlapped common TRI-{level-1} parents of TRI-{level} children\n'
                     f'{len(matrix)} criteria × {len(columns)} parents; each parent testable in both datasets')
            if columns:
                audit = pd.read_csv(path / 'parent_group_index.csv')
                fig = analysis.draw_heatmap(matrix, audit, title)
            else:
                fig, ax = analysis.plt.subplots(figsize=(12, 6))
                ax.text(.5, .5, 'No common parent has a testable criterion in both datasets.',
                        ha='center', va='center', fontsize=14)
                ax.set_axis_off()
                fig.suptitle(title, fontsize=13)
                fig.tight_layout(rect=(0, 0, 1, .9))
            fig.savefig(path / 'clustered_location_bh_q_heatmap_overlapped.png', dpi=150)
            analysis.plt.close(fig)
            analysis.write_csv(matrix, path / 'clustered_location_bh_q_matrix_overlapped.csv.gz',
                               index=True, index_label='metric')
        counts.append({'heatmap_child_level': level, 'parent_motif_level': level-1,
                       'overlapped_parent_motifs': len(columns)})
        analysis.log(f'TRI-{level}: saved both overlapped PNGs ({len(columns)} shared testable parents)')
    analysis.write_csv(pd.DataFrame(counts), args.root / 'overlapped_heatmap_counts.csv')


if __name__ == '__main__':
    main()
