"""Fixed, leakage-aware baseline evaluation for METABRIC motif features."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
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
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, label_binarize


TASKS = {
    "multiclass": np.asarray([0, 1, 2], dtype=np.int8),
    "binary_01_vs_2": np.asarray([0, 1], dtype=np.int8),
}
REPRESENTATIONS = ("raw_log", "level_relative")
MOTIF_MODELS = ("logistic", "random_forest", "complement_nb")


@dataclass
class FittedMotifPipeline:
    """Serializable preprocessing and estimator for a fixed motif vocabulary."""

    task: str
    representation: str
    feature_levels: np.ndarray
    selected_global_indices: np.ndarray
    scaler: MaxAbsScaler
    estimator: BaseEstimator
    classes: np.ndarray

    def transform(self, matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        if matrix.shape[1] != len(self.feature_levels):
            raise ValueError(
                f"Expected {len(self.feature_levels)} motif columns, found {matrix.shape[1]}"
            )
        represented = apply_representation(matrix, self.feature_levels, self.representation)
        return self.scaler.transform(represented[:, self.selected_global_indices])

    def predict(self, matrix: sparse.csr_matrix) -> np.ndarray:
        return self.estimator.predict(self.transform(matrix))

    def predict_proba(self, matrix: sparse.csr_matrix) -> np.ndarray:
        return aligned_probabilities(self.estimator, self.transform(matrix), self.classes)


def normalize_within_triangle_levels(
    matrix: sparse.csr_matrix, feature_levels: np.ndarray
) -> sparse.csr_matrix:
    """Convert counts to within-sample, within-level relative frequencies."""
    output = matrix.astype(np.float32, copy=True).tocsr()
    if output.nnz == 0:
        return output
    unique_levels = np.unique(feature_levels)
    level_to_position = {int(level): position for position, level in enumerate(unique_levels)}
    nonzero_levels = np.fromiter(
        (level_to_position[int(feature_levels[column])] for column in output.indices),
        dtype=np.int16,
        count=output.nnz,
    )
    rows = np.repeat(np.arange(output.shape[0], dtype=np.int32), np.diff(output.indptr))
    flattened = rows.astype(np.int64) * len(unique_levels) + nonzero_levels
    totals = np.bincount(
        flattened,
        weights=output.data,
        minlength=output.shape[0] * len(unique_levels),
    ).reshape(output.shape[0], len(unique_levels))
    denominators = totals[rows, nonzero_levels]
    output.data /= denominators.astype(np.float32, copy=False)
    return output


def apply_representation(
    matrix: sparse.csr_matrix, feature_levels: np.ndarray, representation: str
) -> sparse.csr_matrix:
    if representation == "raw_log":
        output = matrix.astype(np.float32, copy=True).tocsr()
        np.log1p(output.data, out=output.data)
        return output
    if representation == "level_relative":
        return normalize_within_triangle_levels(matrix, feature_levels)
    raise ValueError(f"Unknown feature representation: {representation}")


def _target_labels(grades: np.ndarray, task: str) -> np.ndarray:
    if task == "multiclass":
        return grades.astype(np.int8, copy=True)
    if task == "binary_01_vs_2":
        return (grades == 2).astype(np.int8)
    raise ValueError(f"Unknown task: {task}")


def _model(name: str, seed: int, n_jobs: int, trees: int) -> BaseEstimator:
    if name == "logistic":
        return LogisticRegression(
            solver="saga",
            C=1.0,
            class_weight="balanced",
            max_iter=5_000,
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=trees,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=n_jobs,
        )
    if name == "complement_nb":
        return ComplementNB(alpha=1.0)
    if name == "dummy_prior":
        return DummyClassifier(strategy="prior", random_state=seed)
    raise ValueError(f"Unknown model: {name}")


def aligned_probabilities(
    estimator: BaseEstimator, matrix: Any, classes: np.ndarray
) -> np.ndarray:
    raw = estimator.predict_proba(matrix)
    output = np.zeros((matrix.shape[0], len(classes)), dtype=float)
    positions = {int(label): index for index, label in enumerate(classes)}
    for source_column, label in enumerate(estimator.classes_):
        output[:, positions[int(label)]] = raw[:, source_column]
    row_sums = output.sum(axis=1, keepdims=True)
    output /= np.where(row_sums > 0, row_sums, 1.0)
    return output


def _metrics(y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> dict[str, float]:
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
        binary_targets = label_binarize(y_true, classes=classes)
        average_precision = average_precision_score(
            binary_targets, probabilities, average="macro"
        )
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predicted, labels=classes, zero_division=0
    )
    output = {
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "macro_f1": float(f1_score(y_true, predicted, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "log_loss": float(log_loss(y_true, probabilities, labels=classes)),
    }
    for position, label in enumerate(classes):
        output[f"precision_{int(label)}"] = float(precision[position])
        output[f"recall_{int(label)}"] = float(recall[position])
        output[f"f1_{int(label)}"] = float(f1[position])
    return output


def _select_features(
    train_matrix: sparse.csr_matrix,
    y_train: np.ndarray,
    min_prevalence: float,
    max_features: int,
) -> np.ndarray:
    document_frequency = np.bincount(
        train_matrix.indices, minlength=train_matrix.shape[1]
    )
    minimum_count = max(1, math.ceil(min_prevalence * train_matrix.shape[0]))
    prevalent = np.flatnonzero(document_frequency >= minimum_count)
    if prevalent.size == 0:
        raise ValueError("No motif features survived the prevalence filter")
    scores, _ = chi2(train_matrix[:, prevalent], y_train)
    scores = np.nan_to_num(scores, nan=-np.inf, neginf=-np.inf, posinf=np.finfo(float).max)
    keep_count = min(max_features, prevalent.size)
    if keep_count == prevalent.size:
        selected_local = np.arange(prevalent.size)
    else:
        selected_local = np.argpartition(scores, -keep_count)[-keep_count:]
        selected_local = selected_local[np.argsort(scores[selected_local])[::-1]]
    return prevalent[selected_local].astype(np.int32, copy=False)


def _prediction_rows(
    *,
    task: str,
    representation: str,
    model: str,
    repeat: int,
    fold: int,
    sample_rows: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> list[dict[str, object]]:
    predicted = classes[np.argmax(probabilities, axis=1)]
    rows: list[dict[str, object]] = []
    for position, (_, sample) in enumerate(sample_rows.iterrows()):
        row: dict[str, object] = {
            "task": task,
            "representation": representation,
            "model": model,
            "repeat": repeat,
            "fold": fold,
            "row_index": int(sample["row_index"]),
            "sample": sample["sample"],
            "source_sample": sample["source_sample"],
            "patient_id": sample.get("patient_id", ""),
            "true_label": int(y_true[position]),
            "predicted_label": int(predicted[position]),
        }
        for label in (0, 1, 2):
            row[f"probability_{label}"] = (
                float(probabilities[position, np.flatnonzero(classes == label)[0]])
                if label in classes
                else np.nan
            )
        rows.append(row)
    return rows


def _fit_final(
    raw_matrix: sparse.csr_matrix,
    represented_matrix: sparse.csr_matrix,
    feature_levels: np.ndarray,
    y: np.ndarray,
    task: str,
    representation: str,
    model_name: str,
    min_prevalence: float,
    max_features: int,
    seed: int,
    n_jobs: int,
    trees: int,
) -> FittedMotifPipeline:
    selected = _select_features(raw_matrix, y, min_prevalence, max_features)
    scaler = MaxAbsScaler()
    transformed = scaler.fit_transform(represented_matrix[:, selected])
    estimator = _model(model_name, seed, n_jobs, trees)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        estimator.fit(transformed, y)
    return FittedMotifPipeline(
        task=task,
        representation=representation,
        feature_levels=feature_levels,
        selected_global_indices=selected,
        scaler=scaler,
        estimator=estimator,
        classes=TASKS[task],
    )


def _importance_values(pipeline: FittedMotifPipeline) -> np.ndarray:
    estimator = pipeline.estimator
    if hasattr(estimator, "coef_"):
        return np.max(np.abs(estimator.coef_), axis=0)
    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_)
    if hasattr(estimator, "feature_log_prob_"):
        values = np.asarray(estimator.feature_log_prob_)
        return values.max(axis=0) - values.min(axis=0)
    return np.zeros(len(pipeline.selected_global_indices), dtype=float)


def _write_top_features(
    artifacts_dir: Path,
    results_dir: Path,
    task: str,
    pipeline: FittedMotifPipeline,
) -> None:
    feature_table = pd.read_csv(artifacts_dir / "feature_index.csv.gz")
    selected = feature_table.iloc[pipeline.selected_global_indices].copy()
    selected["importance"] = _importance_values(pipeline)
    selected.sort_values("importance", ascending=False).head(200).to_csv(
        results_dir / f"top_motifs_{task}.csv", index=False
    )


def evaluate_baselines(
    artifacts_dir: Path,
    results_dir: Path,
    *,
    folds: int = 5,
    repeats: int = 5,
    seed: int = 20_260_818,
    min_prevalence: float = 0.05,
    max_features: int = 5_000,
    trees: int = 300,
    n_jobs: int = -1,
) -> dict[str, object]:
    """Evaluate fixed models with repeated sample-level stratified CV."""
    if folds < 2 or repeats < 1:
        raise ValueError("folds must be >=2 and repeats must be >=1")
    if not (0 < min_prevalence <= 1):
        raise ValueError("min_prevalence must be in (0, 1]")
    if max_features < 1 or trees < 1:
        raise ValueError("max_features and trees must be positive")

    results_dir.mkdir(parents=True, exist_ok=True)
    matrix = sparse.load_npz(artifacts_dir / "motif_counts.npz").tocsr()
    feature_levels = np.load(artifacts_dir / "feature_triangle_levels.npy")
    samples = pd.read_csv(artifacts_dir / "samples.csv", keep_default_na=False)
    if len(samples) != matrix.shape[0] or len(feature_levels) != matrix.shape[1]:
        raise ValueError("Prepared matrix and metadata dimensions do not agree")

    labeled_mask = samples["is_labeled"].astype(str).str.lower().eq("true").to_numpy()
    labeled_samples = samples.loc[labeled_mask].reset_index(drop=True)
    grades = labeled_samples["grade"].astype(int).to_numpy()
    raw_counts = matrix[labeled_mask].tocsr()
    representations = {
        name: apply_representation(raw_counts, feature_levels, name)
        for name in REPRESENTATIONS
    }
    metadata_columns = [
        "nodes",
        "edges",
        "source_triangle_types",
        "retained_triangle_types",
    ]
    metadata = labeled_samples[metadata_columns].astype(float).to_numpy()

    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for task, classes in TASKS.items():
        y = _target_labels(grades, task)
        splitter = RepeatedStratifiedKFold(
            n_splits=folds, n_repeats=repeats, random_state=seed
        )
        total_splits = folds * repeats
        for split_index, (train_indices, test_indices) in enumerate(
            splitter.split(np.zeros(len(y)), y)
        ):
            repeat = split_index // folds
            fold = split_index % folds
            split_seed = seed + split_index
            y_train = y[train_indices]
            y_test = y[test_indices]
            print(
                f"[{task}] split {split_index + 1}/{total_splits} "
                f"(repeat={repeat + 1}, fold={fold + 1})"
            )

            dummy = _model("dummy_prior", split_seed, n_jobs, trees)
            dummy.fit(np.zeros((len(train_indices), 1)), y_train)
            dummy_probabilities = aligned_probabilities(
                dummy, np.zeros((len(test_indices), 1)), classes
            )
            metrics = _metrics(y_test, dummy_probabilities, classes)
            fold_rows.append(
                {
                    "task": task,
                    "representation": "baseline",
                    "model": "dummy_prior",
                    "repeat": repeat,
                    "fold": fold,
                    "selected_features": 0,
                    **metrics,
                }
            )
            prediction_rows.extend(
                _prediction_rows(
                    task=task,
                    representation="baseline",
                    model="dummy_prior",
                    repeat=repeat,
                    fold=fold,
                    sample_rows=labeled_samples.iloc[test_indices],
                    y_true=y_test,
                    probabilities=dummy_probabilities,
                    classes=classes,
                )
            )

            metadata_scaler = StandardScaler()
            meta_train = metadata_scaler.fit_transform(metadata[train_indices])
            meta_test = metadata_scaler.transform(metadata[test_indices])
            metadata_model = _model("logistic", split_seed, n_jobs, trees)
            metadata_model.fit(meta_train, y_train)
            metadata_probabilities = aligned_probabilities(
                metadata_model, meta_test, classes
            )
            metrics = _metrics(y_test, metadata_probabilities, classes)
            fold_rows.append(
                {
                    "task": task,
                    "representation": "metadata",
                    "model": "logistic_metadata",
                    "repeat": repeat,
                    "fold": fold,
                    "selected_features": len(metadata_columns),
                    **metrics,
                }
            )
            prediction_rows.extend(
                _prediction_rows(
                    task=task,
                    representation="metadata",
                    model="logistic_metadata",
                    repeat=repeat,
                    fold=fold,
                    sample_rows=labeled_samples.iloc[test_indices],
                    y_true=y_test,
                    probabilities=metadata_probabilities,
                    classes=classes,
                )
            )

            selected = _select_features(
                raw_counts[train_indices], y_train, min_prevalence, max_features
            )
            for representation, represented in representations.items():
                scaler = MaxAbsScaler()
                train_matrix = scaler.fit_transform(represented[train_indices][:, selected])
                test_matrix = scaler.transform(represented[test_indices][:, selected])
                for model_name in MOTIF_MODELS:
                    estimator = _model(model_name, split_seed, n_jobs, trees)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        estimator.fit(train_matrix, y_train)
                    convergence_warnings = sum(
                        "converg" in str(item.message).lower() for item in caught
                    )
                    probabilities = aligned_probabilities(estimator, test_matrix, classes)
                    metrics = _metrics(y_test, probabilities, classes)
                    fold_rows.append(
                        {
                            "task": task,
                            "representation": representation,
                            "model": model_name,
                            "repeat": repeat,
                            "fold": fold,
                            "selected_features": len(selected),
                            "convergence_warnings": convergence_warnings,
                            **metrics,
                        }
                    )
                    prediction_rows.extend(
                        _prediction_rows(
                            task=task,
                            representation=representation,
                            model=model_name,
                            repeat=repeat,
                            fold=fold,
                            sample_rows=labeled_samples.iloc[test_indices],
                            y_true=y_test,
                            probabilities=probabilities,
                            classes=classes,
                        )
                    )

    fold_metrics = pd.DataFrame.from_records(fold_rows)
    predictions = pd.DataFrame.from_records(prediction_rows)
    fold_metrics.to_csv(results_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(results_dir / "oof_predictions.csv.gz", index=False, compression="gzip")

    metric_columns = [
        "roc_auc",
        "average_precision",
        "macro_f1",
        "balanced_accuracy",
        "accuracy",
        "log_loss",
    ]
    summary = (
        fold_metrics.groupby(["task", "representation", "model"], as_index=False)[
            metric_columns
        ]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "_".join(part for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary = summary.reset_index()
    summary.to_csv(results_dir / "summary_metrics.csv", index=False)

    best: dict[str, dict[str, object]] = {}
    for task in TASKS:
        candidates = summary[
            (summary["task"] == task)
            & (summary["representation"].isin(REPRESENTATIONS))
            & (summary["model"].isin(MOTIF_MODELS))
        ]
        winner = candidates.sort_values("roc_auc_mean", ascending=False).iloc[0]
        representation = str(winner["representation"])
        model_name = str(winner["model"])
        y = _target_labels(grades, task)
        pipeline = _fit_final(
            raw_counts,
            representations[representation],
            feature_levels,
            y,
            task,
            representation,
            model_name,
            min_prevalence,
            max_features,
            seed,
            n_jobs,
            trees,
        )
        model_path = results_dir / f"best_model_{task}.joblib"
        joblib.dump(pipeline, model_path, compress=3)
        _write_top_features(artifacts_dir, results_dir, task, pipeline)
        best[task] = {
            "representation": representation,
            "model": model_name,
            "mean_roc_auc": float(winner["roc_auc_mean"]),
            "std_roc_auc": float(winner["roc_auc_std"]),
            "selected_features": int(len(pipeline.selected_global_indices)),
            "model_path": str(model_path),
        }

    run_config: dict[str, object] = {
        "validation": "sample-level RepeatedStratifiedKFold",
        "folds": folds,
        "repeats": repeats,
        "seed": seed,
        "min_prevalence": min_prevalence,
        "max_features": max_features,
        "random_forest_trees": trees,
        "tasks": list(TASKS),
        "representations": list(REPRESENTATIONS),
        "models": list(MOTIF_MODELS),
        "best": best,
    }
    with (results_dir / "evaluation.json").open("w", encoding="utf-8") as output:
        json.dump(run_config, output, indent=2)
        output.write("\n")
    print(f"Saved evaluation results to {results_dir}")
    return run_config
