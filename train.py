"""
train.py
--------
End-to-end training script. Run this once to train all models
and save the best one to models/.

Usage:
    python train.py

Requires: data/creditcard.csv
Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
"""

import joblib
from pathlib import Path
from src.preprocessor import FraudPreprocessor
from src.models import train_all, save_best_model, metrics_table
from src.explainer import FraudExplainer
from src.business_impact import BusinessImpact


def main():
    print("=" * 55)
    print("  FRAUD DETECTION — TRAINING PIPELINE")
    print("=" * 55)

    # ── 1. Load & preprocess ─────────────────────────────────
    print("\n[1/4] Loading and preprocessing data...")
    prep = FraudPreprocessor()
    df   = prep.load()

    summary = prep.eda_summary(df)
    print(f"  Total transactions : {summary['total_transactions']:,}")
    print(f"  Fraud cases        : {summary['fraud_count']:,} ({summary['fraud_rate']:.3%})")
    print(f"  Avg fraud amount   : ${summary['avg_fraud_amount']:.2f}")
    print(f"  Avg legit amount   : ${summary['avg_legit_amount']:.2f}")

    X_train, X_test, y_train, y_test, feature_names = prep.preprocess(df, balance=True)
    print(f"  Train set          : {X_train.shape[0]:,} samples after SMOTE")
    print(f"  Test set           : {X_test.shape[0]:,} samples (original distribution)")

    # ── 2. Train all models ──────────────────────────────────
    print("\n[2/4] Training models...")
    results = train_all(X_train, X_test, y_train, y_test)

    print("\n  Model comparison:")
    print(metrics_table(results).to_string())

    # ── 3. Save best model ───────────────────────────────────
    print("\n[3/4] Saving best model...")
    best_name = save_best_model(results, feature_names)

    # ── 4. SHAP global explainability ────────────────────────
    print("\n[4/4] Computing SHAP global importance...")
    best_model = results[best_name]["model"]

    import numpy as np
    sample_idx = np.random.choice(X_test.shape[0], size=500, replace=False)
    X_sample   = X_test[sample_idx]

    explainer = FraudExplainer(best_model, X_train[:500], feature_names)
    imp_df    = explainer.global_importance(X_sample, top_n=10)

    print("\n  Top 10 fraud-driving features (global SHAP):")
    for _, row in imp_df.iterrows():
        bar = "█" * int(row["Importance"] * 100)
        print(f"  {row['Feature']:<20} {row['Importance']:.4f}  {bar}")

    # Save explainer-related objects for dashboard
    Path("models").mkdir(exist_ok=True)
    joblib.dump(explainer, "models/explainer.pkl")
    joblib.dump(X_sample,  "models/X_sample.pkl")
    joblib.dump(y_test[sample_idx], "models/y_sample.pkl")

    # ── Business impact ──────────────────────────────────────
    print("\n  Business impact (best model):")
    bi = BusinessImpact()
    m  = results[best_name]["metrics"]
    impact = bi.compute_savings(m["tp"], m["fp"], m["tn"], m["fn"])
    print(f"  Fraud caught          : {impact['fraud_caught_pct']:.1%}")
    print(f"  Savings vs no model   : ${impact['net_savings_vs_no_model']:,.0f}")
    print(f"  Savings per 1,000 txns: ${impact['savings_per_1000_txns']:,.2f}")

    print("\n✓ Training complete. Run: streamlit run app/main.py")


if __name__ == "__main__":
    main()
