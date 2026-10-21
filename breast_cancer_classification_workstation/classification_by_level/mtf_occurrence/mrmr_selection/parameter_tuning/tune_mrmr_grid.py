#!/usr/bin/env python3
"""Tune prefixes of the existing expanded-feature ordering with repeated CV."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler, label_binarize
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS = PROJECT_DIR.parents[2] / "artifacts"
DEFAULT_FEATURE_SETS = PROJECT_DIR.parent / "expanded_feature_sets"
DEFAULT_RESULTS = PROJECT_DIR / "results" / "expanded_style"
TASKS = ("multiclass", "binary_01_vs_2")
PREPROCESSORS = ("raw", "rank", "quantile", "z_score", "log1p")
MODELS = ("logistic_regression", "random_forest", "xgboost", "linear_svm")
METRICS = (
    "roc_auc",
    "aupr",
    "macro_f1",
    "balanced_accuracy",
    "accuracy",
    "log_loss",
)


def target_for_task(grades: np.ndarray, task: str) -> np.ndarray:
    if task == "multiclass":
        return grades.astype(np.int8, copy=True)
    if task == "binary_01_vs_2":
        return (grades == 2).astype(np.int8)
    raise ValueError(f"Unknown task: {task}")


def nonzero_rank(X: np.ndarray) -> np.ndarray:
    output = np.zeros_like(X, dtype=np.float32)
    for row in range(X.shape[0]):
        columns = np.flatnonzero(X[row] > 0)
        if columns.size:
            output[row, columns] = (
                rankdata(X[row, columns], method="average") / columns.size
            )
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
        raise ValueError(f"Unknown preprocessing method: {method}")
    return transformer.fit_transform(X[train]), transformer.transform(X[test])


def load_expanded_prefix(
    feature_sets: Path,
    results: Path,
    level: int,
    max_features: int,
) -> tuple[np.ndarray, Path]:
    """Load and verify the exact nested prefix used by the expanded grid."""
    sources = [
        feature_sets / f"tri_{level}_features_{size}.csv"
        for size in (1000, 3000, 5000)
    ]
    frames = [pd.read_csv(path) for path in sources]
    if any(len(frame) < max_features for frame in frames):
        raise ValueError(f"TRI {level} has fewer than {max_features} expanded features")
    reference = frames[0].iloc[:max_features].copy().reset_index(drop=True)
    reference_indices = reference["feature_index"].astype(int).to_numpy()
    for path, frame in zip(sources[1:], frames[1:]):
        candidate = frame["feature_index"].iloc[:max_features].astype(int).to_numpy()
        if not np.array_equal(reference_indices, candidate):
            raise ValueError(
                f"Expanded feature prefixes disagree for TRI {level}: "
                f"{sources[0].name} versus {path.name}"
            )
    output_dir = results / "selected_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"tri_{level}_top_{max_features}_expanded_features.csv"
    reference.to_csv(output, index=False)
    return reference_indices, output


def make_model(name: str, task: str, seed: int, args: argparse.Namespace):
    if name == "logistic_regression":
        return LogisticRegression(
            C=1.0,
            solver="saga",
            max_iter=args.logistic_max_iter,
            tol=1e-2,
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=args.trees,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=args.n_jobs,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        parameters = dict(
            n_estimators=args.xgb_estimators,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=args.n_jobs,
        )
        if task == "multiclass":
            parameters.update(
                objective="multi:softprob", eval_metric="mlogloss", num_class=3
            )
        else:
            parameters.update(objective="binary:logistic", eval_metric="logloss")
        return XGBClassifier(**parameters)
    if name == "linear_svm":
        return LinearSVC(
            C=1.0,
            dual="auto",
            max_iter=args.svm_max_iter,
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {name}")


def prediction_outputs(
    estimator, X: np.ndarray, classes: np.ndarray
) -> tuple[np.ndarray, np.ndarray | None]:
    ranking_scores: np.ndarray | None = None
    if hasattr(estimator, "predict_proba"):
        raw = estimator.predict_proba(X)
    else:
        ranking_scores = estimator.decision_function(X)
        if np.ndim(ranking_scores) == 1:
            positive = 1.0 / (
                1.0 + np.exp(-np.clip(ranking_scores, -50, 50))
            )
            raw = np.column_stack([1.0 - positive, positive])
        else:
            shifted = ranking_scores - np.max(ranking_scores, axis=1, keepdims=True)
            exponentiated = np.exp(np.clip(shifted, -50, 50))
            raw = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    output = np.zeros((X.shape[0], len(classes)), dtype=np.float64)
    positions = {int(label): position for position, label in enumerate(classes)}
    for source, label in enumerate(estimator.classes_):
        output[:, positions[int(label)]] = raw[:, source]
    totals = output.sum(axis=1, keepdims=True)
    return output / np.where(totals > 0, totals, 1.0), ranking_scores


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    ranking_scores: np.ndarray | None = None,
) -> dict[str, float]:
    predicted = classes[np.argmax(probabilities, axis=1)]
    if len(classes) == 2:
        scores = probabilities[:, 1] if ranking_scores is None else ranking_scores
        roc_auc = roc_auc_score(y_true, scores)
        aupr = average_precision_score(y_true, scores)
    else:
        targets = label_binarize(y_true, classes=classes)
        if ranking_scores is None:
            roc_auc = roc_auc_score(
                y_true,
                probabilities,
                labels=classes,
                multi_class="ovr",
                average="macro",
            )
            aupr = average_precision_score(targets, probabilities, average="macro")
        else:
            roc_auc = float(
                np.mean(
                    [
                        roc_auc_score(targets[:, i], ranking_scores[:, i])
                        for i in range(len(classes))
                    ]
                )
            )
            aupr = float(
                np.mean(
                    [
                        average_precision_score(targets[:, i], ranking_scores[:, i])
                        for i in range(len(classes))
                    ]
                )
            )
    return {
        "roc_auc": float(roc_auc),
        "aupr": float(aupr),
        "macro_f1": float(f1_score(y_true, predicted, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "log_loss": float(log_loss(y_true, probabilities, labels=classes)),
    }


def completed_keys(checkpoint: pd.DataFrame) -> set[tuple[object, ...]]:
    columns = [
        "triangle_level",
        "task",
        "requested_feature_count",
        "preprocessing",
        "model",
        "repeat",
        "fold",
    ]
    return (
        set(checkpoint[columns].itertuples(index=False, name=None))
        if not checkpoint.empty
        else set()
    )


def append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame.from_records(rows).to_csv(
        path, mode="a", header=not path.exists(), index=False
    )


def summarize_results(folds: pd.DataFrame, results: Path) -> None:
    ids = [
        "triangle_level",
        "task",
        "requested_feature_count",
        "preprocessing",
        "model",
    ]
    grouped = folds.groupby(ids, sort=False)[list(METRICS)]
    summary = grouped.mean().add_suffix("_mean").join(
        grouped.std().add_suffix("_std")
    ).reset_index()
    summary["joint_auc_aupr_mean"] = (
        summary["roc_auc_mean"] + summary["aupr_mean"]
    ) / 2
    summary.to_csv(results / "summary_metrics.csv", index=False)

    best_auc = (
        summary.sort_values(
            ["roc_auc_mean", "aupr_mean", "requested_feature_count"],
            ascending=[False, False, True],
        )
        .groupby(["triangle_level", "task"], as_index=False)
        .first()
    )
    best_aupr = (
        summary.sort_values(
            ["aupr_mean", "roc_auc_mean", "requested_feature_count"],
            ascending=[False, False, True],
        )
        .groupby(["triangle_level", "task"], as_index=False)
        .first()
    )
    best_auc.to_csv(results / "best_by_roc_auc.csv", index=False)
    best_aupr.to_csv(results / "best_by_aupr.csv", index=False)

    profile = (
        summary.sort_values(
            ["joint_auc_aupr_mean", "requested_feature_count"],
            ascending=[False, True],
        )
        .groupby(["triangle_level", "task", "requested_feature_count"], as_index=False)
        .first()
        .sort_values(["triangle_level", "task", "requested_feature_count"])
    )
    profile.to_csv(results / "feature_count_profile.csv", index=False)

    recommendation_rows: list[dict[str, object]] = []
    for (level, task), task_profile in profile.groupby(
        ["triangle_level", "task"], sort=True
    ):
        peak = task_profile.sort_values(
            ["joint_auc_aupr_mean", "requested_feature_count"],
            ascending=[False, True],
        ).iloc[0]
        peak_folds = folds[
            (folds["triangle_level"] == level)
            & (folds["task"] == task)
            & (folds["requested_feature_count"] == peak["requested_feature_count"])
            & (folds["preprocessing"] == peak["preprocessing"])
            & (folds["model"] == peak["model"])
        ]
        margin = float(
            np.sqrt(peak_folds["roc_auc"].var() + peak_folds["aupr"].var())
            / (2 * np.sqrt(len(peak_folds)))
        )
        recommended = task_profile.loc[
            task_profile["joint_auc_aupr_mean"]
            >= peak["joint_auc_aupr_mean"] - margin,
            "requested_feature_count",
        ].astype(int).sort_values()
        recommendation_rows.append(
            {
                "triangle_level": level,
                "task": task,
                "peak_feature_count": int(peak["requested_feature_count"]),
                "recommended_feature_counts": ";".join(map(str, recommended)),
                "recommended_min_features": int(recommended.min()),
                "recommended_max_features": int(recommended.max()),
                "smallest_one_se_feature_count": int(recommended.min()),
                "peak_roc_auc_mean": float(peak["roc_auc_mean"]),
                "peak_aupr_mean": float(peak["aupr_mean"]),
                "one_se_joint_margin": margin,
            }
        )
    pd.DataFrame(recommendation_rows).to_csv(
        results / "recommended_feature_ranges.csv", index=False
    )


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    labels = {
        "triangle_level": "TRI",
        "task": "Task",
        "requested_feature_count": "Features",
        "preprocessing": "Preprocessing",
        "model": "Model",
        "roc_auc_mean": "ROC-AUC",
        "aupr_mean": "AUPR",
        "peak_feature_count": "Peak",
        "recommended_feature_counts": "Counts within 1 SE",
        "smallest_one_se_feature_count": "Smallest",
    }
    lines = ["| " + " | ".join(labels[column] for column in columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for record in frame[columns].to_dict("records"):
        values = []
        for column in columns:
            value = record[column]
            values.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(results: Path, config: dict[str, object]) -> None:
    folds = pd.read_csv(results / "fold_metrics.csv")
    best_auc = pd.read_csv(results / "best_by_roc_auc.csv")
    best_aupr = pd.read_csv(results / "best_by_aupr.csv")
    recommendations = pd.read_csv(results / "recommended_feature_ranges.csv")
    best_overall_auc = best_auc.sort_values("roc_auc_mean", ascending=False).iloc[0]
    best_overall_aupr = best_aupr.sort_values("aupr_mean", ascending=False).iloc[0]
    best_columns = [
        "triangle_level",
        "task",
        "requested_feature_count",
        "preprocessing",
        "model",
        "roc_auc_mean",
        "aupr_mean",
    ]
    recommendation_columns = [
        "triangle_level",
        "task",
        "peak_feature_count",
        "recommended_feature_counts",
        "smallest_one_se_feature_count",
    ]
    dataset = str(config.get("dataset", "METABRIC"))
    sample_count = config.get("labeled_sample_count", "unknown")
    class_counts = config.get("binary_class_counts", {})
    report = f"""# {dataset} expanded-style feature-count tuning results

## Outcome

- Best overall ROC-AUC: **{best_overall_auc['roc_auc_mean']:.3f}**, using TRI {int(best_overall_auc['triangle_level'])}, {int(best_overall_auc['requested_feature_count'])} features, {best_overall_auc['preprocessing']}, and {best_overall_auc['model']}.
- Best overall AUPR: **{best_overall_aupr['aupr_mean']:.3f}**, using TRI {int(best_overall_aupr['triangle_level'])}, {int(best_overall_aupr['requested_feature_count'])} features, {best_overall_aupr['preprocessing']}, and {best_overall_aupr['model']}.

## Best configuration by ROC-AUC

{_markdown_table(best_auc, best_columns)}

## Best configuration by AUPR

{_markdown_table(best_aupr, best_columns)}

## Feature-count interpretation

Counts below are the tested feature widths whose best joint mean `(ROC-AUC + AUPR) / 2` is within one standard-error margin of the peak for that TRI/task.

{_markdown_table(recommendations, recommendation_columns)}

## Validation integrity

- Dataset: {dataset}; labeled samples: {sample_count}; binary class counts: {class_counts}.
- {len(folds):,} fold/model records with {config.get('results_per_configuration', 'unknown')} held-out scores per configuration.
- Validation used sample-level repeated stratified cross-validation with seed {config.get('seed')}.
- Feature selection target: `{config.get('selection_task')}`.

## Caveats

- The stored mRMR/MI feature ordering was selected using all labeled samples before cross-validation, so scores are exploratory and may be optimistic.
- Folds are sample-stratified rather than patient/group-aware; related BZ cores or images can occur in both training and validation folds.
- The source inventory details unavailable motif-result samples and any legitimately exhausted TRI-level searches.
"""
    (results / "REPORT.md").write_text(report, encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    args.results.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.results / "fold_metrics.partial.csv"
    checkpoint = (
        pd.read_csv(checkpoint_path)
        if args.resume and checkpoint_path.exists()
        else pd.DataFrame()
    )
    completed = completed_keys(checkpoint)
    config = {
        "dataset": args.dataset,
        "selection_task": args.selection_task,
        "feature_source": "expanded_feature_sets/tri_N_features_1000.csv prefix",
        "feature_prefix_verified_against": [3000, 5000],
        "validation": "sample-level RepeatedStratifiedKFold",
        "levels": list(args.levels),
        "tasks": list(args.tasks),
        "sizes": list(args.sizes),
        "preprocessing": list(args.preprocessing),
        "models": list(args.models),
        "folds": args.folds,
        "repeats": args.repeats,
        "seed": args.seed,
        "trees": args.trees,
        "xgb_estimators": args.xgb_estimators,
        "logistic_max_iter": args.logistic_max_iter,
        "svm": "LinearSVC; decision scores used for ROC-AUC/AUPR",
        "n_jobs": args.n_jobs,
    }
    (args.results / "evaluation_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    for level in args.levels:
        source = args.artifacts / f"tri_{level}"
        samples = pd.read_csv(source / "samples.csv")
        matrix = sparse.load_npz(source / "motif_counts.npz").tocsr()
        labeled = (
            samples["is_labeled"].astype(bool).to_numpy()
            & samples["grade"].notna().to_numpy()
        )
        grades = samples.loc[labeled, "grade"].astype(int).to_numpy()
        ordered_indices, exported = load_expanded_prefix(
            args.feature_sets, args.results, level, max(args.sizes)
        )
        print(f"TRI {level} | verified/exported {exported.name}", flush=True)
        level_counts = matrix[labeled][:, ordered_indices].toarray().astype(np.float32)

        for requested_size in args.sizes:
            X = level_counts[:, :requested_size]
            precomputed = {
                "raw": X,
                "rank": nonzero_rank(X),
                "log1p": np.log1p(X),
            }
            for task in args.tasks:
                y = target_for_task(grades, task)
                classes = np.unique(y)
                splitter = RepeatedStratifiedKFold(
                    n_splits=args.folds,
                    n_repeats=args.repeats,
                    random_state=args.seed,
                )
                for split_index, (train, test) in enumerate(splitter.split(X, y)):
                    repeat = split_index // args.folds + 1
                    fold = split_index % args.folds + 1
                    split_seed = args.seed + split_index
                    print(
                        f"TRI {level} | {task} | {requested_size} features | "
                        f"repeat {repeat}/{args.repeats} fold {fold}/{args.folds}",
                        flush=True,
                    )
                    weights = compute_sample_weight("balanced", y[train])
                    for preprocessing in args.preprocessing:
                        X_train, X_test = transform_fold(
                            X, precomputed, preprocessing, train, test, split_seed
                        )
                        rows: list[dict[str, object]] = []
                        for model_name in args.models:
                            key = (
                                level,
                                task,
                                requested_size,
                                preprocessing,
                                model_name,
                                repeat,
                                fold,
                            )
                            if key in completed:
                                continue
                            estimator = make_model(model_name, task, split_seed, args)
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                                estimator.fit(X_train, y[train], sample_weight=weights)
                            probabilities, ranking_scores = prediction_outputs(
                                estimator, X_test, classes
                            )
                            row = {
                                "triangle_level": level,
                                "task": task,
                                "requested_feature_count": requested_size,
                                "actual_feature_count": requested_size,
                                "preprocessing": preprocessing,
                                "model": model_name,
                                "repeat": repeat,
                                "fold": fold,
                                **calculate_metrics(
                                    y[test], probabilities, classes, ranking_scores
                                ),
                            }
                            rows.append(row)
                            completed.add(key)
                        if rows:
                            append_rows(checkpoint_path, rows)

    folds = pd.read_csv(checkpoint_path)
    folds.to_csv(args.results / "fold_metrics.csv", index=False)
    summarize_results(folds, args.results)
    config["total_fold_metric_rows"] = len(folds)
    config["results_per_configuration"] = args.folds * args.repeats
    if args.samples_metadata:
        metadata = json.loads(args.samples_metadata.read_text(encoding="utf-8"))
        config.update(metadata)
    (args.results / "evaluation_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    write_report(args.results, config)
    print(f"Saved parameter-tuning results to {args.results}", flush=True)


def unique(values: Iterable) -> tuple:
    return tuple(dict.fromkeys(values))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--feature-sets", type=Path, default=DEFAULT_FEATURE_SETS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--levels", type=int, nargs="+", default=range(2, 8))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=(10, *range(50, 451, 50))
    )
    parser.add_argument(
        "--preprocessing", nargs="+", choices=PREPROCESSORS, default=PREPROCESSORS
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=int, default=50)
    parser.add_argument("--xgb-estimators", type=int, default=50)
    parser.add_argument("--logistic-max-iter", type=int, default=60)
    parser.add_argument("--svm-max-iter", type=int, default=10_000)
    parser.add_argument("--n-jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset", default="METABRIC")
    parser.add_argument("--selection-task", choices=TASKS, default="multiclass")
    parser.add_argument("--samples-metadata", type=Path)
    args = parser.parse_args(argv)
    args.levels = tuple(sorted(set(args.levels)))
    args.tasks = unique(args.tasks)
    args.sizes = tuple(sorted(set(args.sizes)))
    args.preprocessing = unique(args.preprocessing)
    args.models = unique(args.models)
    if args.folds < 2 or args.repeats < 1:
        parser.error("--folds must be >=2 and --repeats must be >=1")
    if not args.sizes or min(args.sizes) < 1 or max(args.sizes) > 1000:
        parser.error("feature sizes must be between 1 and 1000")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
