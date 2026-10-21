#!/usr/bin/env python3
"""Validate and plot old-vs-current subgraph matching-time distributions."""

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


COLORS = {"Old": "#4C78A8", "current": "#F58518"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-dir", type=Path, required=True)
    parser.add_argument("--current-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--max-level", type=int, default=6)
    return parser.parse_args()


def load_level(folder, level):
    path = folder / f"ccindex_tri_{level}.csv"
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"Pattern", "Matches", "MatchingTimeSeconds"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        rows = [
            (
                row["Pattern"],
                int(row["Matches"]),
                float(row["MatchingTimeSeconds"]),
            )
            for row in reader
        ]
    if not rows:
        raise ValueError(f"No benchmark rows in {path}")
    if any(row[2] <= 0 for row in rows):
        raise ValueError(f"Non-positive matching time in {path}")
    return rows


def percentile(values, fraction):
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def style_violins(parts, color):
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.72)
    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)


def main():
    args = parse_args()
    if args.max_level < 2 or args.max_level > 6:
        raise ValueError("--max-level must be between 2 and 6")
    levels = range(2, args.max_level + 1)
    data = {"Old": {}, "current": {}}

    for level in levels:
        old_rows = load_level(args.old_dir, level)
        current_rows = load_level(args.current_dir, level)
        old_identity = [(pattern, matches) for pattern, matches, _ in old_rows]
        current_identity = [(pattern, matches) for pattern, matches, _ in current_rows]
        if old_identity != current_identity:
            raise ValueError(
                f"Old/current pattern IDs or match counts differ at tri-{level}"
            )
        data["Old"][level] = [elapsed for _, _, elapsed in old_rows]
        data["current"][level] = [elapsed for _, _, elapsed in current_rows]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bases = list(range(1, len(list(levels)) + 1))
    offsets = {"Old": -0.19, "current": 0.19}
    for version in ("Old", "current"):
        values = [data[version][level] for level in levels]
        positions = [base + offsets[version] for base in bases]
        parts = ax.violinplot(
            values,
            positions=positions,
            widths=0.34,
            showmeans=False,
            showmedians=True,
            showextrema=False,
            points=160,
        )
        style_violins(parts, COLORS[version])

    ax.set_yscale("log")
    ax.set_xticks(bases, [f"tri-{level}" for level in levels])
    ax.set_xlabel("Maximum triangle depth")
    ax.set_ylabel("Subgraph matching call time (seconds, log scale)")
    ax.set_title("Old vs current C++ Subgraph Matcher")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(
        handles=[Patch(facecolor=COLORS[name], label=name, alpha=0.72)
                 for name in ("Old", "current")],
        loc="upper left",
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "Level", "Version", "Patterns", "MeanSeconds",
            "MedianSeconds", "P95Seconds", "MedianSpeedupOldOverCurrent",
        ])
        for level in levels:
            old_median = statistics.median(data["Old"][level])
            current_median = statistics.median(data["current"][level])
            speedup = old_median / current_median
            for version in ("Old", "current"):
                values = data[version][level]
                writer.writerow([
                    level,
                    version,
                    len(values),
                    statistics.fmean(values),
                    statistics.median(values),
                    percentile(values, 0.95),
                    speedup,
                ])

    print(f"Saved violin plot to {args.output}")
    print(f"Saved timing summary to {args.summary}")


if __name__ == "__main__":
    main()
