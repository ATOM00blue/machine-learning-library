#!/usr/bin/env python3
"""Reproducible APRL baseline benchmark using scikit-learn toy datasets."""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
    make_friedman1,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aprl import APRLClassifier, APRLRegressor

SEED = 42


def fresh(estimator):
    """Create a new estimator without requiring sklearn inheritance in APRL."""
    return clone(estimator)


def classification_models():
    return {
        "APRL": APRLClassifier(random_state=SEED),
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2_000, random_state=SEED)
        ),
        "K-nearest neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "Random forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1),
    }


def regression_models():
    return {
        "APRL": APRLRegressor(random_state=SEED),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "K-nearest neighbors": make_pipeline(StandardScaler(), KNeighborsRegressor()),
        "Random forest": RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=1),
    }


def evaluate(models, X, y, splitter, metric):
    rows = []
    for name, template in models.items():
        scores = []
        started = time.perf_counter()
        split_args = (X, y) if isinstance(splitter, StratifiedKFold) else (X,)
        for train, test in splitter.split(*split_args):
            model = fresh(template)
            model.fit(X[train], y[train])
            scores.append(metric(y[test], model.predict(X[test])))
        elapsed = time.perf_counter() - started
        rows.append((name, float(np.mean(scores)), float(np.std(scores)), elapsed))
    return rows


def run(folds: int):
    classification = {
        "Iris": load_iris(return_X_y=True),
        "Wine": load_wine(return_X_y=True),
        "Breast cancer": load_breast_cancer(return_X_y=True),
    }
    friedman_X, friedman_y = make_friedman1(
        n_samples=600, n_features=10, noise=1.0, random_state=SEED
    )
    regression = {
        "Diabetes": load_diabetes(return_X_y=True),
        "Friedman #1": (friedman_X, friedman_y),
    }

    results = []
    for dataset, (X, y) in classification.items():
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
        rows = evaluate(classification_models(), X, y, splitter, accuracy_score)
        results.extend(("Classification", dataset, *row) for row in rows)
    for dataset, (X, y) in regression.items():
        splitter = KFold(n_splits=folds, shuffle=True, random_state=SEED)
        rows = evaluate(regression_models(), X, y, splitter, r2_score)
        results.extend(("Regression", dataset, *row) for row in rows)
    return results


def markdown(results, folds: int) -> str:
    lines = [
        "# Custom APRL Algorithm: Baseline Benchmark",
        "",
        (
            f"Results use {folds}-fold cross-validation with seed `{SEED}`. "
            "Classification reports accuracy; regression reports R². "
            "Values are mean ± population standard deviation across folds."
        ),
        "",
        (
            "> These small built-in datasets are smoke benchmarks, not evidence of "
            "state-of-the-art performance or scientific novelty. Runtime is machine-dependent."
        ),
        "",
    ]
    for task in ("Classification", "Regression"):
        lines.extend(
            [
                f"## {task}",
                "",
                "| Dataset | Model | Score | Runtime (s) |",
                "|---|---|---:|---:|",
            ]
        )
        for row_task, dataset, model, mean, std, elapsed in results:
            if row_task == task:
                lines.append(f"| {dataset} | {model} | {mean:.3f} ± {std:.3f} | {elapsed:.3f} |")
        lines.append("")
    lines.extend(
        [
            "## Environment",
            "",
            f"- Python: `{platform.python_version()}`",
            f"- NumPy: `{np.__version__}`",
            f"- scikit-learn: `{sklearn.__version__}`",
            "",
            "Regenerate from the repository root:",
            "",
            "```bash",
            "python benchmarks/benchmark_aprl.py --output algorithms/aprl/BENCHMARKS.md",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least 2")

    report = markdown(run(args.folds), args.folds)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
