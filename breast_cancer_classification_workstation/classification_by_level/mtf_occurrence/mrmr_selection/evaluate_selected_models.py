#!/usr/bin/env python3
"""Cross-validate five classifiers on each level's mRMR-selected motifs."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS = PROJECT_DIR.parents[1] / "artifacts"
DEFAULT_SELECTIONS = PROJECT_DIR
DEFAULT_RESULTS = PROJECT_DIR / "classification_results"
DEFAULT_LEVELS = tuple(range(2, 8))
TASKS = ("multiclass", "binary_01_vs_2")
MODELS = ("logistic_regression", "random_forest", "xgboost", "mlp", "svm")
TASK_LABELS = {
    "multiclass": "Grade 0/1/2",
    "binary_01_vs_2": "Grade 0/1 versus grade 2",
}


def target_for_task(grades: np.ndarray, task: str) -> np.ndarray:
    if task == "multiclass":
        return grades.astype(np.int8, copy=True)
    if task == "binary_01_vs_2":
        return (grades == 2).astype(np.int8)
    raise ValueError(f"Unknown task: {task}")


def make_model(
    name: str,
    *,
    seed: int,
    n_jobs: int,
    trees: int,
    xgb_estimators: int,
    mlp_max_iter: int,
) -> tuple[Pipeline, str]:
    if name == "logistic_regression":
        estimator = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=5_000,
            random_state=seed,
        )
        return Pipeline([("scale", StandardScaler()), ("model", estimator)]), "log1p_scaled"
    if name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=trees,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=n_jobs,
        )
        return Pipeline([("model", estimator)]), "raw_counts"
    if name == "xgboost":
        estimator = XGBClassifier(
            n_estimators=xgb_estimators,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=n_jobs,
        )
        return Pipeline([("model", estimator)]), "raw_counts"
    if name == "mlp":
        estimator = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size=32,
            learning_rate_init=0.001,
            max_iter=mlp_max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=seed,
        )
        return Pipeline([("scale", StandardScaler()), ("model", estimator)]), "log1p_scaled"
    if name == "svm":
        estimator = CalibratedClassifierCV(
            estimator=SVC(C=1.0, kernel="rbf"),
            method="sigmoid",
            cv=3,
            ensemble=False,
            n_jobs=n_jobs,
        )
        return Pipeline([("scale", StandardScaler()), ("model", estimator)]), "log1p_scaled"
    raise ValueError(f"Unknown model: {name}")


def aligned_probabilities(model: Pipeline, X: np.ndarray, classes: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(X)
    fitted_classes = model.named_steps["model"].classes_
    output = np.zeros((len(X), len(classes)), dtype=np.float64)
    positions = {int(label): position for position, label in enumerate(classes)}
    for source, label in enumerate(fitted_classes):
        output[:, positions[int(label)]] = raw[:, source]
    output /= output.sum(axis=1, keepdims=True)
    return output


def calculate_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray
) -> dict[str, float]:
    predicted = classes[np.argmax(probabilities, axis=1)]
    if len(classes) == 2:
        roc_auc = roc_auc_score(y_true, probabilities[:, 1])
        average_precision = average_precision_score(y_true, probabilities[:, 1])
    else:
        roc_auc = roc_auc_score(
            y_true,
            probabilities,
            labels=classes,
            multi_class="ovr",
            average="macro",
        )
        targets = label_binarize(y_true, classes=classes)
        average_precision = average_precision_score(
            targets, probabilities, average="macro"
        )
    precision, recall, per_class_f1, _ = precision_recall_fscore_support(
        y_true, predicted, labels=classes, zero_division=0
    )
    metrics = {
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "macro_f1": float(f1_score(y_true, predicted, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "log_loss": float(log_loss(y_true, probabilities, labels=classes)),
    }
    for position, label in enumerate(classes):
        metrics[f"precision_{int(label)}"] = float(precision[position])
        metrics[f"recall_{int(label)}"] = float(recall[position])
        metrics[f"f1_{int(label)}"] = float(per_class_f1[position])
    return metrics


def summarize_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    identifiers = ["triangle_level", "task", "model", "representation"]
    metric_columns = [
        column
        for column in fold_metrics.columns
        if column not in {*identifiers, "repeat", "fold"}
    ]
    grouped = fold_metrics.groupby(identifiers, sort=False)[metric_columns]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std().add_suffix("_std")
    return mean.join(std).reset_index()


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for values in frame[columns].itertuples(index=False, name=None):
        formatted = [f"{value:.3f}" if isinstance(value, float) else str(value) for value in values]
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, results_dir: Path, folds: int, repeats: int) -> None:
    lines = [
        "# Classification using per-level mRMR-selected motifs",
        "",
        f"Results use {folds}-fold stratified cross-validation repeated {repeats} time(s). ",
        "Each TRI level contributes its 100 Feature-engine MRMR-selected motifs.",
        "",
        "> **Interpretation warning:** mRMR was fitted once using all 487 labeled samples ",
        "> before cross-validation. Consequently, these are exploratory comparisons and ",
        "> may overestimate performance. A leakage-free estimate requires refitting mRMR ",
        "> inside every training fold.",
        "> Splits are also sample-stratified rather than patient-grouped. The 487 labeled ",
        "> samples represent 441 patients, and samples from 42 repeated patients may occur ",
        "> on both sides of a fold.",
        "",
    ]
    columns = [
        "triangle_level",
        "model",
        "roc_auc_mean",
        "roc_auc_std",
        "average_precision_mean",
        "macro_f1_mean",
        "balanced_accuracy_mean",
        "accuracy_mean",
    ]
    for task in TASKS:
        lines.extend([f"## {TASK_LABELS[task]}", ""])
        task_rows = summary[summary["task"] == task].copy()
        lines.extend([markdown_table(task_rows, columns), ""])
        winner = task_rows.loc[task_rows["roc_auc_mean"].idxmax()]
        lines.extend(
            [
                f"Highest mean ROC AUC: TRI {int(winner['triangle_level'])} "
                f"with {winner['model']} ({winner['roc_auc_mean']:.3f} ± "
                f"{winner['roc_auc_std']:.3f}).",
                "",
            ]
        )
    lines.extend(
        [
            "## Preprocessing and models",
            "",
            "- Logistic Regression, MLP, and RBF-SVM use `log1p(count)` followed by "
            "training-fold-only standardization.",
            "- Random Forest and XGBoost use raw occurrence counts.",
            "- Balanced sample weights are computed separately within every training fold.",
            "- Multiclass ROC AUC is macro one-vs-rest; binary positive class is grade 2.",
            "",
        ]
    )
    (results_dir / "classification_report.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    args.results.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for level in args.levels:
        source = args.artifacts / f"tri_{level}"
        samples = pd.read_csv(source / "samples.csv")
        selected = pd.read_csv(args.selections / f"tri_{level}_selected_features.csv")
        matrix = sparse.load_npz(source / "motif_counts.npz").tocsr()
        labeled = samples["is_labeled"].astype(bool).to_numpy() & samples["grade"].notna().to_numpy()
        labeled_samples = samples.loc[labeled].reset_index(drop=True)
        grades = samples.loc[labeled, "grade"].astype(int).to_numpy()
        indices = selected["feature_index"].astype(int).to_numpy()
        counts = matrix[labeled][:, indices].toarray().astype(np.float32)
        log_counts = np.log1p(counts)

        for task in TASKS:
            y = target_for_task(grades, task)
            classes = np.unique(y)
            splitter = RepeatedStratifiedKFold(
                n_splits=args.folds,
                n_repeats=args.repeats,
                random_state=args.seed,
            )
            for model_name in MODELS:
                print(f"TRI {level} | {task} | {model_name}", flush=True)
                for split_number, (train, test) in enumerate(splitter.split(counts, y)):
                    repeat = split_number // args.folds + 1
                    fold = split_number % args.folds + 1
                    model, representation = make_model(
                        model_name,
                        seed=args.seed + split_number,
                        n_jobs=args.n_jobs,
                        trees=args.trees,
                        xgb_estimators=args.xgb_estimators,
                        mlp_max_iter=args.mlp_max_iter,
                    )
                    X = log_counts if representation == "log1p_scaled" else counts
                    weights = compute_sample_weight(class_weight="balanced", y=y[train])
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                        model.fit(X[train], y[train], model__sample_weight=weights)
                    probabilities = aligned_probabilities(model, X[test], classes)
                    metrics = calculate_metrics(y[test], probabilities, classes)
                    fold_rows.append(
                        {
                            "triangle_level": level,
                            "task": task,
                            "model": model_name,
                            "representation": representation,
                            "repeat": repeat,
                            "fold": fold,
                            **metrics,
                        }
                    )
                    predicted = classes[np.argmax(probabilities, axis=1)]
                    for position, sample_index in enumerate(test):
                        sample = labeled_samples.iloc[sample_index]
                        row: dict[str, object] = {
                            "triangle_level": level,
                            "task": task,
                            "model": model_name,
                            "repeat": repeat,
                            "fold": fold,
                            "row_index": int(sample["row_index"]),
                            "sample": sample["sample"],
                            "patient_id": sample.get("patient_id", ""),
                            "grade": int(grades[sample_index]),
                            "true_label": int(y[sample_index]),
                            "predicted_label": int(predicted[position]),
                        }
                        for label in (0, 1, 2):
                            matches = np.flatnonzero(classes == label)
                            row[f"probability_{label}"] = (
                                float(probabilities[position, matches[0]])
                                if matches.size
                                else np.nan
                            )
                        prediction_rows.append(row)

    fold_metrics = pd.DataFrame(fold_rows)
    summary = summarize_metrics(fold_metrics)
    best_by_level = (
        summary.sort_values("roc_auc_mean", ascending=False)
        .groupby(["task", "triangle_level"], as_index=False)
        .first()
        .sort_values(["task", "triangle_level"])
    )
    predictions = pd.DataFrame(prediction_rows)
    fold_metrics.to_csv(args.results / "fold_metrics.csv", index=False)
    summary.to_csv(args.results / "summary_metrics.csv", index=False)
    best_by_level.to_csv(args.results / "best_models_by_level.csv", index=False)
    predictions.to_csv(
        args.results / "oof_predictions.csv.gz", index=False, compression="gzip"
    )
    config = {
        "levels": list(args.levels),
        "tasks": list(TASKS),
        "models": list(MODELS),
        "folds": args.folds,
        "repeats": args.repeats,
        "seed": args.seed,
        "selected_features_per_level": 100,
        "random_forest_trees": args.trees,
        "xgboost_estimators": args.xgb_estimators,
        "mlp_max_iterations": args.mlp_max_iter,
        "n_jobs": args.n_jobs,
        "selection_leakage_warning": "mRMR was fitted on all labeled samples before CV",
    }
    (args.results / "evaluation_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    write_report(summary, args.results, args.folds, args.repeats)
    print(f"Saved results to {args.results}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--levels", type=int, nargs="+", default=DEFAULT_LEVELS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20_260_820)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--xgb-estimators", type=int, default=300)
    parser.add_argument("--mlp-max-iter", type=int, default=1_000)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    args.levels = tuple(sorted(set(args.levels)))
    return args


if __name__ == "__main__":
    evaluate(parse_args())
