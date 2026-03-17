"""
src/models.py
-------------
Train and evaluate four fraud detection models:
  1. Logistic Regression  (baseline)
  2. Random Forest
  3. XGBoost
  4. LightGBM

Evaluation uses AUC-ROC, AUC-PR, F1, Precision, Recall.
NOTE: Never use raw accuracy on imbalanced fraud data.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)
import xgboost as xgb
import lightgbm as lgb


MODELS_DIR = Path(__file__).parent.parent / "models"


def get_models():
    """Return dict of model name → unfitted estimator."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=8,
            random_state=42, n_jobs=-1, class_weight="balanced"
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=6,
            learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=1,
            random_state=42, eval_metric="aucpr",
            verbosity=0,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200, max_depth=6,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8,
            random_state=42, verbose=-1,
        ),
    }


def evaluate(model, X_test, y_test, threshold=0.5) -> dict:
    """
    Compute full evaluation metrics for a fitted model.

    Parameters
    ----------
    threshold : float
        Decision threshold for converting probabilities to labels.
        Lowering threshold increases recall (catches more fraud)
        at the cost of more false positives.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cm   = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    prec_curve, rec_curve, pr_thresholds = precision_recall_curve(y_test, y_prob)

    return {
        "auc_roc":    round(roc_auc_score(y_test, y_prob), 4),
        "auc_pr":     round(average_precision_score(y_test, y_prob), 4),
        "f1":         round(f1_score(y_test, y_pred), 4),
        "precision":  round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":     round(recall_score(y_test, y_pred), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "threshold":  threshold,
        "y_prob":     y_prob,
        "fpr":        fpr,
        "tpr":        tpr,
        "prec_curve": prec_curve,
        "rec_curve":  rec_curve,
    }


def train_all(X_train, X_test, y_train, y_test, threshold=0.5) -> dict:
    """
    Train all four models and return results dict.

    Returns
    -------
    dict: {model_name: {"model": fitted, "metrics": dict}}
    """
    results = {}
    for name, model in get_models().items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test, threshold)
        results[name] = {"model": model, "metrics": metrics}
        print(f"    AUC-ROC: {metrics['auc_roc']:.4f}  "
              f"AUC-PR: {metrics['auc_pr']:.4f}  "
              f"Recall: {metrics['recall']:.4f}")
    return results


def save_best_model(results: dict, feature_names: list):
    """Save the best model (highest AUC-PR) to models/ directory."""
    MODELS_DIR.mkdir(exist_ok=True)
    best_name = max(results, key=lambda k: results[k]["metrics"]["auc_pr"])
    best      = results[best_name]["model"]

    joblib.dump(best, MODELS_DIR / "best_model.pkl")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")
    print(f"\nBest model: {best_name} (AUC-PR: {results[best_name]['metrics']['auc_pr']:.4f})")
    print(f"Saved to models/best_model.pkl")
    return best_name


def load_best_model():
    """Load the saved best model and feature names."""
    model    = joblib.load(MODELS_DIR / "best_model.pkl")
    features = joblib.load(MODELS_DIR / "feature_names.pkl")
    return model, features


def metrics_table(results: dict) -> pd.DataFrame:
    """Return a clean comparison DataFrame of all model metrics."""
    rows = []
    for name, r in results.items():
        m = r["metrics"]
        rows.append({
            "Model":     name,
            "AUC-ROC":   f"{m['auc_roc']:.4f}",
            "AUC-PR":    f"{m['auc_pr']:.4f}",
            "F1":        f"{m['f1']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall":    f"{m['recall']:.4f}",
            "TP":        m["tp"],
            "FP":        m["fp"],
            "FN":        m["fn"],
        })
    df = pd.DataFrame(rows).set_index("Model")
    return df


def find_optimal_threshold(model, X_test, y_test,
                            fp_cost=1.0, fn_cost=150.0) -> float:
    """
    Find the threshold that minimizes total business cost.

    Parameters
    ----------
    fp_cost : float
        Cost of a false positive (blocking a legitimate transaction).
        Typically $1–5 in customer friction / manual review.
    fn_cost : float
        Cost of a false negative (missing a fraud).
        Typically $50–200 in chargebacks + investigation.

    Returns
    -------
    float : optimal probability threshold
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.01, 1.0, 0.01)
    best_thresh = 0.5
    best_cost   = float("inf")

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        cost = fp * fp_cost + fn * fn_cost
        if cost < best_cost:
            best_cost   = cost
            best_thresh = t

    return round(float(best_thresh), 2)
