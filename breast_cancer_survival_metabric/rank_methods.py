"""Dataset-independent methods ported from the BZ survival workflow.

Source functions retain the BZ definitions; patient IDs, time units and labels
are adapted for METABRIC. No BZ modules with ambiguous bare imports are loaded.
"""
from __future__ import annotations
import math
import os
import warnings
from itertools import combinations
from pathlib import Path
from typing import Iterable
os.environ.setdefault("MPLCONFIGDIR", "/tmp/metabric-rank-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import chi2, rankdata
from statsmodels.duration.hazard_regression import PHReg
MIN_LEVEL, MAX_LEVEL = 1, 7
IDENTITY_COLUMNS = ["triangle_level", "wl_hash"]
COLORS = {"Non-enriched": "#3568A8", "Enriched": "#D1495B"}


def fit_cox(
    time: np.ndarray,
    event: np.ndarray,
    enriched: np.ndarray,
    covariates: np.ndarray | None = None,
) -> dict[str, object]:
    if len(np.unique(enriched)) < 2:
        return {
            "coef": np.nan,
            "hazard_ratio": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "p_value": np.nan,
            "fit_status": "not_estimable_no_group_variation",
            "fit_warning": "",
        }
    design = enriched.astype(float)[:, None]
    if covariates is not None:
        design = np.column_stack([design, covariates])
    caught: list[warnings.WarningMessage]
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = PHReg(time, design, status=event, ties="efron")
            result = model.fit(disp=0)
        coefficient = float(result.params[0])
        interval = np.asarray(result.conf_int(), dtype=float)[0]
        p_value = float(result.pvalues[0])
        values = np.array([coefficient, interval[0], interval[1], p_value])
        warning_text = " | ".join(dict.fromkeys(str(item.message) for item in caught))
        if not np.isfinite(values).all():
            return {
                "coef": np.nan,
                "hazard_ratio": np.nan,
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "p_value": np.nan,
                "fit_status": "not_estimable_nonfinite_result",
                "fit_warning": warning_text or "Cox fit returned non-finite statistics",
            }
        if warning_text:
            status = "estimated_with_warning"
        else:
            status = "estimated"
        return {
            "coef": coefficient,
            "hazard_ratio": math.exp(coefficient) if coefficient < 710 else np.inf,
            "ci95_low": math.exp(float(interval[0])) if interval[0] < 710 else np.inf,
            "ci95_high": math.exp(float(interval[1])) if interval[1] < 710 else np.inf,
            "p_value": p_value,
            "fit_status": status,
            "fit_warning": warning_text,
        }
    except Exception as exc:  # Numerical failures vary by motif and LAPACK build.
        return {
            "coef": np.nan,
            "hazard_ratio": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "p_value": np.nan,
            "fit_status": "not_estimable_fit_error",
            "fit_warning": f"{type(exc).__name__}: {exc}",
        }


def step_function(time: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    """Kaplan-Meier estimate with log-log pointwise 95% confidence intervals."""
    event_times = np.sort(np.unique(time[event == 1]))
    survival = 1.0
    greenwood_sum = 0.0
    rows = [{"time": 0.0, "survival": 1.0, "lower": 1.0, "upper": 1.0}]
    for t in event_times:
        at_risk = int(np.sum(time >= t))
        deaths = int(np.sum((time == t) & (event == 1)))
        survival *= 1.0 - deaths / at_risk
        if at_risk > deaths:
            greenwood_sum += deaths / (at_risk * (at_risk - deaths))
        if 0.0 < survival < 1.0 and greenwood_sum > 0:
            se_loglog = math.sqrt(greenwood_sum) / abs(math.log(survival))
            loglog = math.log(-math.log(survival))
            lower = math.exp(-math.exp(loglog + 1.96 * se_loglog))
            upper = math.exp(-math.exp(loglog - 1.96 * se_loglog))
        else:
            lower = upper = survival
        rows.append({"time": float(t), "survival": survival, "lower": lower, "upper": upper})
    return pd.DataFrame(rows)


def logrank_test(time: np.ndarray, event: np.ndarray, group: np.ndarray) -> tuple[float, float]:
    """Two-sided, two-group Mantel-Cox log-rank test."""
    observed = expected = variance = 0.0
    for t in np.sort(np.unique(time[event == 1])):
        at_risk = time >= t
        n = int(at_risk.sum())
        n1 = int((at_risk & group).sum())
        d = int(((time == t) & (event == 1)).sum())
        d1 = int(((time == t) & (event == 1) & group).sum())
        observed += d1
        expected += d * n1 / n
        if n > 1:
            variance += n1 * (n - n1) * d * (n - d) / (n * n * (n - 1))
    statistic = (observed - expected) ** 2 / variance if variance > 0 else np.nan
    return float(statistic), float(chi2.sf(statistic, 1))


def analyze_rank_survival(patient_score: np.ndarray, time: np.ndarray, event: np.ndarray) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    score, time, event = np.asarray(patient_score, float), np.asarray(time, float), np.asarray(event, int)
    if len(score) != len(time) or len(time) != len(event) or not np.isfinite(score).all():
        raise ValueError("Patient arrays are invalid")
    median = float(np.median(score))
    high = score > median
    binary = fit_cox(time, event, high)
    if high.any() and (~high).any():
        try:
            logrank_stat, logrank_p = logrank_test(time, event, high)
            logrank_status = "estimated"
        except Exception as exc:
            logrank_stat = logrank_p = np.nan
            logrank_status = f"not_estimable_{type(exc).__name__}"
    else:
        logrank_stat = logrank_p = np.nan
        logrank_status = "not_estimable_no_group_variation"
    sd = float(np.std(score, ddof=1))
    if np.isfinite(sd) and sd > 0:
        score_z = (score - score.mean()) / sd
        continuous = fit_cox(time, event, score_z)
    else:
        score_z = np.full(len(score), np.nan)
        continuous = {"coef": np.nan, "hazard_ratio": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "p_value": np.nan, "fit_status": "not_estimable_constant_score", "fit_warning": "Rank score has zero variance"}
    result = {
        "median_patient_rank_score": median, "rank_group_definition": "patient_rank_score > cohort_median",
        "n_survival": len(score), "events_survival": int(event.sum()),
        "n_high_rank": int(high.sum()), "events_high_rank": int(event[high].sum()),
        "n_low_rank": int((~high).sum()), "events_low_rank": int(event[~high].sum()),
        "coef_rank_binary": binary["coef"], "hazard_ratio_rank_binary": binary["hazard_ratio"],
        "ci95_low_rank_binary": binary["ci95_low"], "ci95_high_rank_binary": binary["ci95_high"],
        "p_value_rank_binary": binary["p_value"], "rank_binary_fit_status": binary["fit_status"], "rank_binary_fit_warning": binary["fit_warning"],
        "logrank_statistic": logrank_stat, "logrank_p_value": logrank_p, "logrank_status": logrank_status,
        "rank_score_mean": float(score.mean()), "rank_score_sd": sd,
        "coef_rank_continuous_per_sd": continuous["coef"], "hazard_ratio_rank_continuous_per_sd": continuous["hazard_ratio"],
        "ci95_low_rank_continuous_per_sd": continuous["ci95_low"], "ci95_high_rank_continuous_per_sd": continuous["ci95_high"],
        "p_value_rank_continuous_per_sd": continuous["p_value"], "rank_continuous_fit_status": continuous["fit_status"], "rank_continuous_fit_warning": continuous["fit_warning"],
    }
    return result, high, score_z


def json_safe(value: object) -> object:
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if isinstance(value, np.generic): value = value.item()
    if isinstance(value, float) and not np.isfinite(value): return None
    if pd.isna(value): return None
    return value


def fmt(value: object) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.3g}"


def plot_km(data: pd.DataFrame, result: pd.Series, output: Path, dpi: int, time_unit: str = "days", cohort: str = "") -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    colors = {"Low rank score": COLORS["Non-enriched"], "High rank score": COLORS["Enriched"]}
    for group in colors:
        subset = data.loc[data["rank_group"].eq(group)]
        if subset.empty:
            continue
        km = step_function(subset["time_days"].to_numpy(float), subset["event"].to_numpy(int))
        ax.step(km["time"], km["survival"], where="post", lw=2.2, color=colors[group], label=f"{group} (n={len(subset)}, events={int(subset.event.sum())})")
        ax.fill_between(km["time"], km["lower"], km["upper"], step="post", alpha=.15, color=colors[group])
        censored = subset.loc[subset["event"].eq(0), "time_days"].to_numpy(float)
        if len(censored):
            positions = np.searchsorted(km["time"].to_numpy(), censored, side="right") - 1
            ax.scatter(censored, km["survival"].to_numpy()[positions], marker="|", s=55, color=colors[group])
    text = f"Binary Cox HR = {fmt(result['hazard_ratio_rank_binary'])}\n95% CI {fmt(result['ci95_low_rank_binary'])}–{fmt(result['ci95_high_rank_binary'])}\nCox p = {fmt(result['p_value_rank_binary'])}; log-rank p = {fmt(result['logrank_p_value'])}\nContinuous HR per SD = {fmt(result['hazard_ratio_rank_continuous_per_sd'])}"
    ax.text(.98, .98, text, transform=ax.transAxes, ha="right", va="top", fontsize=9.5, bbox={"boxstyle": "round", "facecolor": "white", "alpha": .9, "edgecolor": ".75"})
    ax.set(title=f"{cohort + ' | ' if cohort else ''}TRI-{int(result['triangle_level'])} {str(result['wl_hash'])[:8]}: survival by rank score\n{result['cell_type_composition']}", xlabel=f"Overall survival ({time_unit})", ylabel="Survival probability", ylim=(0, 1.03))
    ax.grid(axis="y", alpha=.2)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout(); fig.savefig(output, dpi=dpi); plt.close(fig)


def plot_distribution(data: pd.DataFrame, result: pd.Series, output: Path, dpi: int) -> None:
    ordered = data.sort_values(["rank_score", "patient_id"], kind="mergesort").reset_index(drop=True)
    colors = np.where(ordered["rank_group"].eq("High rank score"), COLORS["Enriched"], COLORS["Non-enriched"])
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.bar(np.arange(len(ordered)), ordered["rank_score"], color=colors, width=1, linewidth=0)
    ax.axhline(float(result["median_patient_rank_score"]), color="black", ls="--", lw=1.2, label=f"Median = {float(result['median_patient_rank_score']):.4g}")
    ax.set(title=f"TRI-{int(result['triangle_level'])} {str(result['wl_hash'])[:8]}: patient rank-normalized abundance", xlabel="Patients ordered by rank score", ylabel="Mean image rank score", xlim=(-.5, len(ordered)-.5), ylim=(0, 1.01))
    ax.grid(axis="y", alpha=.2); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(output, dpi=dpi); plt.close(fig)


def rank_score_sparse(counts: sparse.spmatrix, vocabulary_size: int) -> sparse.csr_matrix:
    if vocabulary_size < 2:
        raise ValueError("Ranking requires at least two observed motifs")
    scores = counts.tocsr(copy=True).astype(float)
    scores.sum_duplicates()
    scores.eliminate_zeros()
    if scores.shape[1] != vocabulary_size:
        raise ValueError("Count matrix and vocabulary size differ")
    for row in range(scores.shape[0]):
        start, stop = scores.indptr[row : row + 2]
        values = scores.data[start:stop]
        if len(values):
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError("Sparse motif counts must be finite and positive")
            ranks = rankdata(-values, method="average")
            scores.data[start:stop] = (
                vocabulary_size - ranks
            ) / (vocabulary_size - 1)
    return scores


def triangle_count(graph: nx.Graph) -> int:
    return sum(nx.triangles(graph).values()) // 3


def motif_hash(graph: nx.Graph) -> str:
    return nx.weisfeiler_lehman_graph_hash(graph, node_attr="label")


def graph_encoding(graph: nx.Graph) -> tuple[str, str]:
    nodes = sorted(graph.nodes)
    remap = {node: index for index, node in enumerate(nodes)}
    labels = "&".join(str(graph.nodes[node]["label"]) for node in nodes)
    edges = ";".join(
        f"{min(remap[left], remap[right])}-{max(remap[left], remap[right])}"
        for left, right in sorted(graph.edges)
    )
    return labels, edges


def potential_direct_parents(
    graph: nx.Graph,
) -> list[tuple[str, str, nx.Graph]]:
    """Reverse either supported one-triangle extension operation."""
    expected_triangles = triangle_count(graph) - 1
    candidates: list[tuple[str, str, nx.Graph]] = []

    # Reverse adding one new vertex on an existing edge.
    for vertex in list(graph.nodes):
        neighbors = list(graph.neighbors(vertex))
        if graph.degree(vertex) != 2 or not graph.has_edge(*neighbors):
            continue
        parent = graph.copy()
        parent.remove_node(vertex)
        if nx.is_connected(parent) and triangle_count(parent) == expected_triangles:
            candidates.append(("remove_vertex", str(vertex), parent))

    # Reverse adding an edge between two existing boundary vertices.
    for left, right in list(graph.edges):
        parent = graph.copy()
        parent.remove_edge(left, right)
        if nx.is_connected(parent) and triangle_count(parent) == expected_triangles:
            candidates.append(("remove_edge", f"{left}-{right}", parent))
    return candidates


def potential_direct_children(
    graph: nx.Graph, valid_hashes: set[str], labels: Iterable[str]
) -> list[tuple[str, str, nx.Graph]]:
    """Generate unique potential children that add exactly one triangle."""
    expected = triangle_count(graph) + 1
    children: dict[str, tuple[str, str, nx.Graph]] = {}
    new_node = max(graph.nodes, default=-1) + 1
    for left, right in graph.edges:
        for label in labels:
            child = graph.copy()
            child.add_node(new_node, label=str(label))
            child.add_edges_from([(left, new_node), (right, new_node)])
            child_hash = motif_hash(child)
            if child_hash in valid_hashes and triangle_count(child) == expected:
                children.setdefault(
                    child_hash,
                    ("add_vertex", f"{new_node}:{label}@{min(left,right)}-{max(left,right)}", child),
                )
    for left, right in combinations(graph.nodes, 2):
        if graph.has_edge(left, right):
            continue
        if len(set(graph.neighbors(left)).intersection(graph.neighbors(right))) != 1:
            continue
        child = graph.copy()
        child.add_edge(left, right)
        child_hash = motif_hash(child)
        if child_hash in valid_hashes and triangle_count(child) == expected:
            children.setdefault(
                child_hash,
                ("add_edge", f"{min(left,right)}-{max(left,right)}", child),
            )
    return list(children.values())


def trace_lineage(
    target: nx.Graph,
    target_level: int,
    vocabularies: dict[int, set[str]],
    labels: Iterable[str],
) -> tuple[dict[int, dict[str, nx.Graph]], pd.DataFrame]:
    target_hash = motif_hash(target)
    graphs: dict[int, dict[str, nx.Graph]] = {target_level: {target_hash: target}}
    relationships: dict[tuple[int, str, int, str], dict[str, object]] = {}

    for child_level in range(target_level, MIN_LEVEL, -1):
        parent_level = child_level - 1
        parents: dict[str, nx.Graph] = {}
        for child_hash, child in graphs.get(child_level, {}).items():
            for operation, removed, parent in potential_direct_parents(child):
                parent_hash = motif_hash(parent)
                if parent_hash not in vocabularies[parent_level]:
                    continue
                parents.setdefault(parent_hash, parent)
                key = (parent_level, parent_hash, child_level, child_hash)
                relationships.setdefault(
                    key,
                    {
                        "parent_level": parent_level,
                        "parent_hash": parent_hash,
                        "child_level": child_level,
                        "child_hash": child_hash,
                        "growth_operation": operation.replace("remove_", "add_"),
                        "changed_node_or_edge": removed,
                    },
                )
        graphs[parent_level] = parents
        if not parents:
            break

    for parent_level in range(target_level, MAX_LEVEL):
        child_level = parent_level + 1
        children: dict[str, nx.Graph] = {}
        for parent_hash, parent in graphs.get(parent_level, {}).items():
            for operation, changed, child in potential_direct_children(
                parent, vocabularies[child_level], labels
            ):
                child_hash = motif_hash(child)
                children.setdefault(child_hash, child)
                key = (parent_level, parent_hash, child_level, child_hash)
                relationships.setdefault(
                    key,
                    {
                        "parent_level": parent_level,
                        "parent_hash": parent_hash,
                        "child_level": child_level,
                        "child_hash": child_hash,
                        "growth_operation": operation,
                        "changed_node_or_edge": changed,
                    },
                )
        graphs[child_level] = children
        if not children:
            break

    edge_columns = [
        "parent_level",
        "parent_hash",
        "child_level",
        "child_hash",
        "growth_operation",
        "changed_node_or_edge",
    ]
    edges = pd.DataFrame(relationships.values(), columns=edge_columns)
    if len(edges):
        edges = edges.sort_values(edge_columns[:4], kind="mergesort").reset_index(drop=True)
    return graphs, edges


def plot_rank_p_lineage(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    target_level: int,
    target_hash: str,
    output: Path,
    p_threshold: float,
) -> None:
    p_values = pd.to_numeric(nodes["p_value_rank_binary"], errors="coerce")
    finite = nodes.loc[np.isfinite(p_values) & p_values.ge(0)].copy()
    finite["minus_log10_p"] = -np.log10(
        pd.to_numeric(finite["p_value_rank_binary"], errors="raise").clip(lower=1e-300)
    )
    positions: dict[tuple[int, str], tuple[float, float]] = {}
    for level, subset in finite.groupby("triangle_level", sort=True):
        ordered = subset.sort_values(["p_value_rank_binary", "wl_hash"])
        offsets = (
            np.linspace(-0.16, 0.16, len(ordered))
            if len(ordered) > 1
            else [0.0]
        )
        for offset, row in zip(offsets, ordered.itertuples(index=False)):
            positions[(int(level), row.wl_hash)] = (
                int(level) + float(offset),
                float(row.minus_log10_p),
            )

    fig, ax = plt.subplots(figsize=(15, 8.5))
    segments: list[list[tuple[float, float]]] = []
    for edge in edges.itertuples(index=False):
        parent_key = (int(edge.parent_level), edge.parent_hash)
        child_key = (int(edge.child_level), edge.child_hash)
        if parent_key in positions and child_key in positions:
            segments.append([positions[parent_key], positions[child_key]])
    if segments:
        ax.add_collection(
            LineCollection(
                segments,
                colors="0.72",
                linewidths=1.0,
                zorder=1,
            )
        )

    role_colors = {
        "ancestor": "#377eb8",
        "target": "#d1495b",
        "descendant": "#e69f00",
    }
    for role, subset in finite.groupby("lineage_role"):
        xy = [
            positions[(int(row.triangle_level), row.wl_hash)]
            for row in subset.itertuples(index=False)
        ]
        if xy:
            ax.scatter(
                [value[0] for value in xy],
                [value[1] for value in xy],
                s=95 if role == "target" else 52,
                color=role_colors[role],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
                label=role.capitalize(),
            )
    if len(finite) <= 50:
        label_rows = finite
    else:
        label_rows = pd.concat(
            [
                finite.nsmallest(20, "p_value_rank_binary"),
                finite.loc[finite["lineage_role"].eq("target")],
            ]
        ).drop_duplicates(IDENTITY_COLUMNS)
    for row in label_rows.itertuples(index=False):
        x, y = positions[(int(row.triangle_level), row.wl_hash)]
        ax.annotate(
            row.wl_hash[:6],
            (x, y),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )

    ax.axhline(
        -np.log10(p_threshold),
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=f"Rank-normalized p = {p_threshold:g}",
    )
    levels = sorted(nodes["triangle_level"].unique())
    ax.set_xticks(levels)
    ax.set_xlim(min(levels) - 0.45, max(levels) + 0.45)
    finite_counts = finite.groupby("triangle_level").size()
    structural_counts = nodes.groupby("triangle_level").size()
    for level in levels:
        ax.text(
            level,
            0.012,
            f"{int(finite_counts.get(level, 0))}/{int(structural_counts[level])}\nfinite p / structural",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7,
            color="0.35",
        )
    ax.set_xlabel("Triangle level", labelpad=31)
    ax.set_ylabel("−log10(rank-normalized binary Cox p value)")
    ax.set_title(
        f"Potential lineage of TRI-{target_level} motif {target_hash[:8]}: "
        "Rank-normalized Cox p\n"
        "Connected points have finite rank p values; all structural relatives remain in the CSV"
    )
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(output, dpi=190)
    plt.close(fig)
