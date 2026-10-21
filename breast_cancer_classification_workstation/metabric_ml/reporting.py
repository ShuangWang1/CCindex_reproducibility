"""Create plots and a concise Markdown report from baseline evaluation outputs."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "metabric-ml-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize


def _aggregate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "task",
        "representation",
        "model",
        "row_index",
        "sample",
        "source_sample",
        "true_label",
    ]
    probability_columns = ["probability_0", "probability_1", "probability_2"]
    return (
        frame.groupby(group_columns, as_index=False, dropna=False)[probability_columns]
        .mean()
        .sort_values(group_columns)
    )


def _plot_class_distribution(samples: pd.DataFrame, path: Path) -> None:
    labeled = samples[samples["is_labeled"].astype(str).str.lower().eq("true")]
    counts = labeled["grade"].astype(int).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    bars = ax.bar(counts.index.astype(str), counts.values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.bar_label(bars, padding=3)
    ax.set_title("METABRIC labeled sample distribution")
    ax.set_xlabel("Grade")
    ax.set_ylabel("Samples")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_model_comparison(summary: pd.DataFrame, path: Path) -> None:
    display = summary.copy()
    display["configuration"] = display["representation"] + " / " + display["model"]
    tasks = list(display["task"].drop_duplicates())
    fig, axes = plt.subplots(1, len(tasks), figsize=(7 * len(tasks), 5.3), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        subset = display[display["task"] == task].sort_values("roc_auc_mean")
        positions = np.arange(len(subset))
        ax.errorbar(
            subset["roc_auc_mean"],
            positions,
            xerr=subset["roc_auc_std"].fillna(0),
            fmt="o",
            capsize=3,
            color="#4C78A8",
        )
        ax.set_yticks(positions, subset["configuration"])
        ax.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("ROC AUC (mean ± fold SD)")
        ax.set_title(task.replace("_", " ").title())
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("METABRIC fixed-baseline comparison", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _best_predictions(
    predictions: pd.DataFrame, best: dict[str, object], task: str
) -> pd.DataFrame:
    return predictions[
        (predictions["task"] == task)
        & (predictions["representation"] == best["representation"])
        & (predictions["model"] == best["model"])
    ].copy()


def _plot_confusion(predictions: pd.DataFrame, task: str, path: Path) -> None:
    classes = np.asarray([0, 1, 2] if task == "multiclass" else [0, 1])
    probabilities = predictions[[f"probability_{label}" for label in classes]].to_numpy()
    predicted = classes[np.argmax(probabilities, axis=1)]
    matrix = confusion_matrix(predictions["true_label"], predicted, labels=classes)
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    annotations = np.asarray(
        [
            [f"{matrix[row, col]}\n{normalized[row, col]:.1%}" for col in range(len(classes))]
            for row in range(len(classes))
        ]
    )
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    sns.heatmap(
        normalized,
        annot=annotations,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={"label": "Row-normalized fraction"},
        ax=ax,
    )
    ax.set_title(f"Aggregated out-of-fold confusion matrix: {task}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_roc(predictions: pd.DataFrame, task: str, path: Path) -> None:
    classes = np.asarray([0, 1, 2] if task == "multiclass" else [0, 1])
    fig, ax = plt.subplots(figsize=(6.2, 5.3))
    if task == "binary_01_vs_2":
        false_positive, true_positive, _ = roc_curve(
            predictions["true_label"], predictions["probability_1"]
        )
        score = auc(false_positive, true_positive)
        ax.plot(false_positive, true_positive, linewidth=2.3, label=f"Grade 2 (AUC={score:.3f})")
    else:
        targets = label_binarize(predictions["true_label"], classes=classes)
        all_false_positive = np.unique(
            np.concatenate(
                [
                    roc_curve(targets[:, position], predictions[f"probability_{label}"])[0]
                    for position, label in enumerate(classes)
                ]
            )
        )
        mean_true_positive = np.zeros_like(all_false_positive)
        for position, label in enumerate(classes):
            false_positive, true_positive, _ = roc_curve(
                targets[:, position], predictions[f"probability_{label}"]
            )
            class_auc = auc(false_positive, true_positive)
            ax.plot(
                false_positive,
                true_positive,
                linewidth=1.7,
                label=f"Grade {label} (AUC={class_auc:.3f})",
            )
            mean_true_positive += np.interp(all_false_positive, false_positive, true_positive)
        mean_true_positive /= len(classes)
        macro_auc = auc(all_false_positive, mean_true_positive)
        ax.plot(
            all_false_positive,
            mean_true_positive,
            color="#222222",
            linestyle="--",
            linewidth=2.3,
            label=f"Macro OVR (AUC={macro_auc:.3f})",
        )
    ax.plot([0, 1], [0, 1], color="#888888", linestyle=":", label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title(f"Aggregated out-of-fold ROC: {task}")
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _markdown_table(summary: pd.DataFrame) -> str:
    columns = [
        "task",
        "representation",
        "model",
        "roc_auc_mean",
        "roc_auc_std",
        "macro_f1_mean",
        "balanced_accuracy_mean",
    ]
    display = summary[columns].sort_values(["task", "roc_auc_mean"], ascending=[True, False]).copy()
    for column in columns[3:]:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    headers = list(display.columns)
    rows = ["| " + " | ".join(headers) + " |"]
    rows.append("| " + " | ".join("---" for _ in headers) + " |")
    for values in display.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(rows)


def create_report(artifacts_dir: Path, results_dir: Path) -> Path:
    """Create figures and metabric_report.md from saved results."""
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    with (artifacts_dir / "audit.json").open(encoding="utf-8") as source:
        audit = json.load(source)
    with (results_dir / "evaluation.json").open(encoding="utf-8") as source:
        evaluation = json.load(source)
    samples = pd.read_csv(artifacts_dir / "samples.csv", keep_default_na=False)
    summary = pd.read_csv(results_dir / "summary_metrics.csv")
    predictions = pd.read_csv(results_dir / "oof_predictions.csv.gz")
    aggregated = _aggregate_predictions(predictions)

    _plot_class_distribution(samples, figures_dir / "class_distribution.png")
    _plot_model_comparison(summary, figures_dir / "model_comparison.png")
    for task, best in evaluation["best"].items():
        best_predictions = _best_predictions(aggregated, best, task)
        _plot_confusion(best_predictions, task, figures_dir / f"confusion_{task}.png")
        _plot_roc(best_predictions, task, figures_dir / f"roc_{task}.png")

    labeled = samples[samples["is_labeled"].astype(str).str.lower().eq("true")]
    patient_counts = labeled.loc[labeled["patient_id"] != "", "patient_id"].value_counts()
    repeated_patients = int((patient_counts > 1).sum())
    repeated_images = int(patient_counts[patient_counts > 1].sum())
    excess_images = int((patient_counts - 1).clip(lower=0).sum())
    best_lines = []
    for task, best in evaluation["best"].items():
        best_lines.append(
            f"- **{task}**: `{best['representation']} / {best['model']}` — "
            f"ROC AUC {best['mean_roc_auc']:.3f} ± {best['std_roc_auc']:.3f}."
        )

    report = f"""# METABRIC Motif Classification Baselines

## Data audit

- Motif samples prepared: **{audit['sample_count']:,}**
- Labeled samples evaluated: **{audit['labeled_sample_count']:,}**
- Unlabeled samples retained in artifacts: **{audit['unlabeled_sample_count']:,}**
- Grade counts: **{audit['class_counts']}**
- Exact motif features: **{audit['feature_count']:,}** across triangle levels {audit['triangle_levels']}
- Sparse nonzero occurrence counts: **{audit['nonzero_count']:,}**
- Higher-level files omitted after motif search exhaustion: **{len(audit.get('search_exhausted_missing_level_files', []))}** (treated as empty levels)

![Class distribution](figures/class_distribution.png)

## Evaluation

The fixed baselines use sample-level {evaluation['folds']}-fold stratified cross-validation repeated {evaluation['repeats']} times. Feature prevalence filtering and chi-square selection are fitted independently within each training fold. ROC AUC is the primary metric.

{chr(10).join(best_lines)}

![Model comparison](figures/model_comparison.png)

### All configurations

{_markdown_table(summary)}

## Best-model diagnostics

### Three-class grade 0/1/2

![Multiclass ROC](figures/roc_multiclass.png)

![Multiclass confusion matrix](figures/confusion_multiclass.png)

### Binary grade 0/1 versus grade 2

Grade 2 is the positive class.

![Binary ROC](figures/roc_binary_01_vs_2.png)

![Binary confusion matrix](figures/confusion_binary_01_vs_2.png)

## Interpretation and limitations

- Binary and multiclass ROC AUC describe different targets; the binary task is intrinsically simpler, so a higher score does not prove that grade merging is biologically preferable.
- Validation is intentionally sample-level. The labeled data contain **{repeated_patients} patients represented by multiple images**, covering **{repeated_images} images** ({excess_images} images beyond one per patient). Images from one patient may occur in both training and test folds, so the reported scores may be optimistic for patient-independent deployment.
- Raw log counts retain absolute motif abundance; level-relative features emphasize motif composition and reduce graph-size effects. The metadata-only logistic baseline helps reveal whether graph size alone explains performance.
- The saved final models were refitted on all labeled samples after configuration selection. Their performance estimates are the repeated cross-validation scores above, not a separate held-out test result.
- Top-motif tables are model-association summaries, not evidence of causality.
"""
    report_path = results_dir / "metabric_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved METABRIC report to {report_path}")
    return report_path
