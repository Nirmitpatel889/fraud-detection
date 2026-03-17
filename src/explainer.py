"""
src/explainer.py
----------------
SHAP explainability for the fraud detection model.

Provides:
  - Global explainability: which features drive fraud predictions overall?
  - Local explainability: why was THIS specific transaction flagged?
  - Business translation: convert SHAP values into plain English

SHAP (SHapley Additive exPlanations) assigns each feature a contribution
value for each prediction. Positive SHAP = pushed toward fraud.
Negative SHAP = pushed toward legitimate.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class FraudExplainer:
    """
    SHAP-based explainability wrapper for fraud detection models.

    Parameters
    ----------
    model        : Fitted XGBoost / LightGBM / sklearn model
    X_train      : Training data (used to build SHAP explainer background)
    feature_names: List of feature column names
    """

    def __init__(self, model, X_train, feature_names):
        self.model         = model
        self.feature_names = feature_names
        self.explainer     = None
        self._init_explainer(X_train)

    def _init_explainer(self, X_train):
        """Initialize the appropriate SHAP explainer."""
        try:
            # TreeExplainer is fast and exact for tree-based models
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            # KernelExplainer as fallback (slower but universal)
            background = shap.sample(X_train, 100)
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, background
            )

    def get_shap_values(self, X):
        """
        Compute SHAP values for a set of transactions.

        Returns array of shape (n_samples, n_features).
        Positive values = push toward fraud class.
        """
        vals = self.explainer.shap_values(X)
        # Tree models return [class0_vals, class1_vals] — take class 1 (fraud)
        if isinstance(vals, list):
            return vals[1]
        return vals

    def global_importance(self, X, top_n=15) -> pd.DataFrame:
        """
        Global feature importance: mean |SHAP| across all transactions.

        Returns DataFrame sorted by importance descending.
        """
        shap_vals = self.get_shap_values(X)
        importance = np.abs(shap_vals).mean(axis=0)
        df = pd.DataFrame({
            "Feature":    self.feature_names,
            "Importance": importance,
        }).sort_values("Importance", ascending=False).head(top_n)
        return df.reset_index(drop=True)

    def explain_transaction(self, x_single) -> pd.DataFrame:
        """
        Local explanation: SHAP values for one transaction.

        Parameters
        ----------
        x_single : 1D numpy array of feature values

        Returns
        -------
        DataFrame with Feature, Value, SHAP, Direction columns
        """
        if x_single.ndim == 1:
            x_single = x_single.reshape(1, -1)

        shap_vals = self.get_shap_values(x_single)[0]
        fraud_prob = float(self.model.predict_proba(x_single)[0, 1])

        df = pd.DataFrame({
            "Feature":   self.feature_names,
            "Value":     x_single[0],
            "SHAP":      shap_vals,
            "Direction": ["Toward fraud" if v > 0 else "Away from fraud"
                          for v in shap_vals],
        })
        df["abs_shap"] = df["SHAP"].abs()
        df = df.sort_values("abs_shap", ascending=False).drop("abs_shap", axis=1)
        df["fraud_probability"] = fraud_prob
        return df.reset_index(drop=True)

    def waterfall_plotly(self, x_single, top_n=12) -> go.Figure:
        """
        Plotly waterfall chart showing top SHAP contributors for one transaction.
        """
        exp_df   = self.explain_transaction(x_single).head(top_n)
        fraud_prob = float(exp_df["fraud_probability"].iloc[0])

        # Base value (expected model output across all training data)
        base_val = float(self.explainer.expected_value
                         if not isinstance(self.explainer.expected_value, list)
                         else self.explainer.expected_value[1])

        features = exp_df["Feature"].tolist()
        shap_v   = exp_df["SHAP"].tolist()
        colors   = ["#f87171" if v > 0 else "#34d399" for v in shap_v]

        fig = go.Figure(go.Bar(
            x=shap_v,
            y=features,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.4f}" for v in shap_v],
            textposition="outside",
            textfont=dict(size=11),
        ))

        fig.update_layout(
            title=dict(
                text=f"Transaction explanation — Fraud probability: {fraud_prob:.1%}",
                font=dict(size=14),
            ),
            xaxis_title="SHAP value (impact on fraud prediction)",
            yaxis=dict(autorange="reversed"),
            height=420,
            plot_bgcolor="#0d0f17",
            paper_bgcolor="#0d0f17",
            font=dict(color="#9ca3af", size=11,
                      family="JetBrains Mono, monospace"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)",
                       linecolor="#1c2033", zerolinecolor="#374151"),
            yaxis_gridcolor="rgba(255,255,255,0.04)",
            margin=dict(l=130, r=80, t=50, b=40),
        )
        return fig

    def global_importance_plotly(self, X, top_n=15) -> go.Figure:
        """
        Plotly horizontal bar chart of global SHAP feature importance.
        """
        imp_df = self.global_importance(X, top_n)

        fig = go.Figure(go.Bar(
            x=imp_df["Importance"],
            y=imp_df["Feature"],
            orientation="h",
            marker_color="#3b82f6",
            text=[f"{v:.4f}" for v in imp_df["Importance"]],
            textposition="outside",
            textfont=dict(size=11),
        ))

        fig.update_layout(
            title=dict(
                text="Global feature importance (mean |SHAP|)",
                font=dict(size=14),
            ),
            xaxis_title="Mean |SHAP value|",
            yaxis=dict(autorange="reversed"),
            height=460,
            plot_bgcolor="#0d0f17",
            paper_bgcolor="#0d0f17",
            font=dict(color="#9ca3af", size=11,
                      family="JetBrains Mono, monospace"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)",
                       linecolor="#1c2033"),
            yaxis_gridcolor="rgba(255,255,255,0.04)",
            margin=dict(l=130, r=80, t=50, b=40),
        )
        return fig

    def business_reason(self, x_single, top_n=3) -> str:
        """
        Convert SHAP values into a plain-English fraud explanation.

        Returns a string like:
        "Flagged because: unusually high V14 signal (-3.2), large transaction
         amount ($4,891), and atypical V10 pattern (-2.1)."
        """
        exp_df = self.explain_transaction(x_single)
        fraud_drivers = exp_df[exp_df["SHAP"] > 0].head(top_n)

        if fraud_drivers.empty:
            return "No strong fraud signals detected in this transaction."

        reasons = []
        for _, row in fraud_drivers.iterrows():
            feat  = row["Feature"]
            val   = row["Value"]
            shap  = row["SHAP"]

            if feat == "Amount_scaled":
                reasons.append(f"large transaction amount (scaled: {val:.2f})")
            elif feat == "Time_scaled":
                reasons.append(f"unusual transaction timing (scaled: {val:.2f})")
            else:
                reasons.append(
                    f"anomalous {feat} signal ({val:.2f}, "
                    f"SHAP: +{shap:.3f})"
                )

        return "Flagged because: " + ", ".join(reasons) + "."

    def shap_beeswarm_matplotlib(self, X, max_display=15) -> plt.Figure:
        """
        Classic SHAP beeswarm summary plot (matplotlib).
        Useful for notebooks / README screenshots.
        """
        shap_vals = self.get_shap_values(X)
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            shap_vals, X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
            plot_type="dot",
        )
        plt.tight_layout()
        return fig
