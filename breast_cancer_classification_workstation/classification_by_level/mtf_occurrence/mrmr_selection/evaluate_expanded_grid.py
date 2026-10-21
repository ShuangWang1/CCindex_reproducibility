#!/usr/bin/env python3
"""Evaluate expanded motif sets across five preprocessing methods and models."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer, StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS = PROJECT_DIR.parents[1] / "artifacts"
DEFAULT_FEATURE_SETS = PROJECT_DIR / "expanded_feature_sets"
DEFAULT_RESULTS = PROJECT_DIR / "expanded_grid_results"
TASKS = ("multiclass", "binary_01_vs_2")
PREPROCESSORS = ("raw_count", "normalized_count", "rank", "quantile", "z_score")
MODELS = ("logistic_regression", "random_forest", "xgboost", "mlp", "svm")


def target_for_task(grades: np.ndarray, task: str) -> np.ndarray:
    return grades.astype(np.int8) if task == "multiclass" else (grades == 2).astype(np.int8)


def normalized_counts(X: np.ndarray) -> np.ndarray:
    totals = X.sum(axis=1, keepdims=True)
    return X / np.where(totals > 0, totals, 1)


def nonzero_rank(X: np.ndarray) -> np.ndarray:
    output = np.zeros_like(X, dtype=np.float32)
    for row in range(len(X)):
        columns = np.flatnonzero(X[row] > 0)
        if columns.size:
            output[row, columns] = rankdata(X[row, columns], method="average") / columns.size
    return output


def transform_fold(
    X: np.ndarray,
    precomputed: dict[str, np.ndarray],
    method: str,
    train: np.ndarray,
    test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if method in precomputed:
        return precomputed[method][train], precomputed[method][test]
    if method == "z_score":
        transformer = StandardScaler()
    elif method == "quantile":
        transformer = QuantileTransformer(
            n_quantiles=min(100, len(train)),
            output_distribution="normal",
            random_state=seed,
        )
    else:
        raise ValueError(method)
    return transformer.fit_transform(X[train]), transformer.transform(X[test])


def make_model(
    name: str,
    task: str,
    seed: int,
    n_jobs: int,
    trees: int,
    xgb_estimators: int,
    mlp_max_iter: int,
):
    if name == "logistic_regression":
        return LogisticRegression(
            C=1.0,
            solver="saga",
            max_iter=60,
            tol=1e-2,
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=trees,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=n_jobs,
        )
    if name == "xgboost":
        parameters = dict(
            n_estimators=xgb_estimators,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=n_jobs,
        )
        if task == "multiclass":
            parameters.update(objective="multi:softprob", eval_metric="mlogloss", num_class=3)
        else:
            parameters.update(objective="binary:logistic", eval_metric="logloss")
        return XGBClassifier(**parameters)
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(32,),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size=32,
            learning_rate_init=0.001,
            max_iter=mlp_max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=seed,
        )
    if name == "svm":
        return CalibratedClassifierCV(
            estimator=SVC(C=1.0, kernel="rbf"),
            method="sigmoid",
            cv=3,
            ensemble=False,
            n_jobs=n_jobs,
        )
    raise ValueError(name)


def probabilities_for(estimator, X: np.ndarray, classes: np.ndarray) -> np.ndarray:
    raw = estimator.predict_proba(X)
    output = np.zeros((len(X), len(classes)), dtype=float)
    positions = {int(label): position for position, label in enumerate(classes)}
    for source, label in enumerate(estimator.classes_):
        output[:, positions[int(label)]] = raw[:, source]
    output /= output.sum(axis=1, keepdims=True)
    return output


def metrics(y: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    predicted = classes[np.argmax(probabilities, axis=1)]
    if len(classes) == 2:
        auc = roc_auc_score(y, probabilities[:, 1])
        ap = average_precision_score(y, probabilities[:, 1])
    else:
        auc = roc_auc_score(y, probabilities, labels=classes, multi_class="ovr", average="macro")
        ap = average_precision_score(label_binarize(y, classes=classes), probabilities, average="macro")
    return {
        "roc_auc": float(auc),
        "average_precision": float(ap),
        "macro_f1": float(f1_score(y, predicted, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "accuracy": float(accuracy_score(y, predicted)),
        "log_loss": float(log_loss(y, probabilities, labels=classes)),
    }


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    ids = [
        "triangle_level",
        "task",
        "requested_feature_count",
        "actual_feature_count",
        "preprocessing",
        "model",
    ]
    metric_columns = [c for c in folds.columns if c not in {*ids, "fold"}]
    grouped = folds.groupby(ids, sort=False)[metric_columns]
    return grouped.mean().add_suffix("_mean").join(grouped.std().add_suffix("_std")).reset_index()


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for values in frame[columns].itertuples(index=False, name=None):
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, results: Path, folds: int) -> None:
    best = (
        summary.sort_values("roc_auc_mean", ascending=False)
        .groupby(["task", "triangle_level", "requested_feature_count"], as_index=False)
        .first()
        .sort_values(["task", "triangle_level", "requested_feature_count"])
    )
    preprocessing = (
        summary.groupby(["task", "preprocessing"], as_index=False)
        .agg(mean_roc_auc=("roc_auc_mean", "mean"), mean_macro_f1=("macro_f1_mean", "mean"))
        .sort_values(["task", "mean_roc_auc"], ascending=[True, False])
    )
    sizes = (
        summary.groupby(["task", "requested_feature_count"], as_index=False)
        .agg(mean_roc_auc=("roc_auc_mean", "mean"), mean_macro_f1=("macro_f1_mean", "mean"))
        .sort_values(["task", "requested_feature_count"])
    )
    best.to_csv(results / "best_configuration_by_level_and_size.csv", index=False)
    preprocessing.to_csv(results / "preprocessing_summary.csv", index=False)
    sizes.to_csv(results / "feature_size_summary.csv", index=False)

    columns = [
        "triangle_level", "requested_feature_count", "actual_feature_count",
        "preprocessing", "model", "roc_auc_mean", "roc_auc_std",
        "macro_f1_mean", "balanced_accuracy_mean", "accuracy_mean",
    ]
    lines = [
        "# Expanded mRMR-feature preprocessing grid",
        "",
        f"This broad screening experiment uses {folds}-fold stratified cross-validation across 1,000, 3,000, and 5,000 requested features, five preprocessing methods, five classifiers, and two tasks.",
        "",
        "> The first 100 features are the Feature-engine MRMR selection; larger sets are nested raw-count mutual-information expansions. TRI 2 is capped at its 1,080 available motifs.",
        "> Feature selection used the complete labeled dataset and folds are not patient-grouped, so scores are exploratory and may be optimistic.",
        "",
    ]
    for task, label in [("multiclass", "Grade 0/1/2"), ("binary_01_vs_2", "Grade 0/1 versus grade 2")]:
        lines.extend([f"## Best configuration for each level and requested size: {label}", ""])
        lines.extend([markdown_table(best[best.task == task], columns), ""])
    lines.extend(["## Average performance by preprocessing", ""])
    lines.extend([markdown_table(preprocessing, ["task", "preprocessing", "mean_roc_auc", "mean_macro_f1"]), ""])
    lines.extend(["## Average performance by requested feature count", ""])
    lines.extend([markdown_table(sizes, ["task", "requested_feature_count", "mean_roc_auc", "mean_macro_f1"]), ""])
    lines.extend(
        [
            "## Preprocessing definitions",
            "",
            "- `raw_count`: unmodified motif occurrence counts.",
            "- `normalized_count`: each count divided by the sample's total count across the selected motifs.",
            "- `rank`: nonzero counts ranked within each sample and scaled to (0, 1]; absent motifs remain zero.",
            "- `quantile`: feature-wise quantile transform to a normal distribution, fitted on each training fold.",
            "- `z_score`: feature-wise mean removal and standard-deviation scaling, fitted on each training fold.",
            "",
        ]
    )
    (results / "expanded_grid_report.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    args.results.mkdir(parents=True, exist_ok=True)
    partial = args.results / "fold_metrics.partial.csv"
    expected_rows_per_grid = len(TASKS) * len(PREPROCESSORS) * len(MODELS) * args.folds
    if args.resume and partial.exists():
        checkpoint = pd.read_csv(partial)
        group_sizes = checkpoint.groupby(
            ["triangle_level", "requested_feature_count"]
        )["fold"].transform("size")
        checkpoint = checkpoint[group_sizes == expected_rows_per_grid].copy()
        rows: list[dict[str, object]] = checkpoint.to_dict("records")
        print(
            f"Resuming from {len(rows):,} validated checkpoint rows "
            f"({len(rows) // expected_rows_per_grid} completed grids)",
            flush=True,
        )
    else:
        rows = []
    for level in args.levels:
        source = args.artifacts / f"tri_{level}"
        samples = pd.read_csv(source / "samples.csv")
        matrix = sparse.load_npz(source / "motif_counts.npz").tocsr()
        labeled = samples["is_labeled"].astype(bool).to_numpy() & samples["grade"].notna().to_numpy()
        grades = samples.loc[labeled, "grade"].astype(int).to_numpy()
        evaluated_by_indices: dict[tuple[int, ...], list[dict[str, object]]] = {}

        for requested_size in args.sizes:
            completed = sum(
                row["triangle_level"] == level
                and row["requested_feature_count"] == requested_size
                for row in rows
            )
            if completed == expected_rows_per_grid:
                print(
                    f"TRI {level} | requested {requested_size}: resumed completed grid",
                    flush=True,
                )
                continue
            selected = pd.read_csv(args.feature_sets / f"tri_{level}_features_{requested_size}.csv")
            indices = selected["feature_index"].astype(int).to_numpy()
            key = tuple(indices)
            if key in evaluated_by_indices:
                clones = []
                for original in evaluated_by_indices[key]:
                    clone = {**original, "requested_feature_count": requested_size}
                    rows.append(clone)
                    clones.append(clone)
                print(f"TRI {level} | requested {requested_size}: reused identical {len(indices)}-feature results", flush=True)
                continue

            X = matrix[labeled][:, indices].toarray().astype(np.float32)
            precomputed = {
                "raw_count": X,
                "normalized_count": normalized_counts(X),
                "rank": nonzero_rank(X),
            }
            configuration_rows: list[dict[str, object]] = []
            for task in TASKS:
                y = target_for_task(grades, task)
                classes = np.unique(y)
                splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
                splits = list(splitter.split(X, y))
                for preprocessing in PREPROCESSORS:
                    print(
                        f"TRI {level} | {len(indices)} features | {task} | {preprocessing}",
                        flush=True,
                    )
                    for fold, (train, test) in enumerate(splits, start=1):
                        X_train, X_test = transform_fold(
                            X, precomputed, preprocessing, train, test, args.seed + fold
                        )
                        weights = compute_sample_weight("balanced", y[train])
                        for model_name in MODELS:
                            estimator = make_model(
                                model_name,
                                task,
                                args.seed + fold,
                                args.n_jobs,
                                args.trees,
                                args.xgb_estimators,
                                args.mlp_max_iter,
                            )
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                                estimator.fit(X_train, y[train], sample_weight=weights)
                            probabilities = probabilities_for(estimator, X_test, classes)
                            row = {
                                "triangle_level": level,
                                "task": task,
                                "requested_feature_count": requested_size,
                                "actual_feature_count": len(indices),
                                "preprocessing": preprocessing,
                                "model": model_name,
                                "fold": fold,
                                **metrics(y[test], probabilities, classes),
                            }
                            rows.append(row)
                            configuration_rows.append(row)
            evaluated_by_indices[key] = configuration_rows
            pd.DataFrame(rows).to_csv(args.results / "fold_metrics.partial.csv", index=False)

    folds = pd.DataFrame(rows)
    summary = summarize(folds)
    folds.to_csv(args.results / "fold_metrics.csv", index=False)
    summary.to_csv(args.results / "summary_metrics.csv", index=False)
    write_report(summary, args.results, args.folds)
    config = {
        "levels": list(args.levels), "sizes": list(args.sizes),
        "preprocessing": list(PREPROCESSORS), "models": list(MODELS),
        "tasks": list(TASKS), "folds": args.folds, "seed": args.seed,
        "trees": args.trees, "xgb_estimators": args.xgb_estimators,
        "mlp_max_iter": args.mlp_max_iter, "n_jobs": args.n_jobs,
    }
    (args.results / "evaluation_config.json").write_text(json.dumps(config, indent=2) + "\n")
    if partial.exists():
        partial.unlink()
    print(f"Saved expanded grid results to {args.results}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--feature-sets", type=Path, default=DEFAULT_FEATURE_SETS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--levels", type=int, nargs="+", default=range(2, 8))
    parser.add_argument("--sizes", type=int, nargs="+", default=(1000, 3000, 5000))
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20_260_820)
    parser.add_argument("--trees", type=int, default=50)
    parser.add_argument("--xgb-estimators", type=int, default=50)
    parser.add_argument("--mlp-max-iter", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume completed level/size grids from fold_metrics.partial.csv",
    )
    args = parser.parse_args()
    args.levels = tuple(sorted(set(args.levels)))
    args.sizes = tuple(sorted(set(args.sizes)))
    return args


if __name__ == "__main__":
    evaluate(parse_args())
