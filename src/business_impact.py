"""
src/business_impact.py
----------------------
Translates model metrics into business dollar impact.

Key insight: in fraud detection, the cost of a missed fraud (FN)
is far higher than the cost of a false alert (FP). This module
quantifies that tradeoff and computes the model's financial value.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


class BusinessImpact:
    """
    Computes dollar savings and cost analysis for fraud detection models.

    Parameters
    ----------
    fp_cost : float   Cost of blocking a legitimate transaction ($)
    fn_cost : float   Cost of missing a fraud ($) — includes chargeback,
                      investigation, and customer loss
    """

    def __init__(self, fp_cost=2.0, fn_cost=150.0):
        self.fp_cost = fp_cost
        self.fn_cost = fn_cost

    def compute_savings(self, tp, fp, tn, fn,
                        total_fraud_amount=None) -> dict:
        """
        Compute business metrics for a given confusion matrix.

        Returns a dict with savings, costs, and net benefit.
        """
        # Costs
        cost_fp = fp * self.fp_cost
        cost_fn = fn * self.fn_cost
        total_cost = cost_fp + cost_fn

        # Baseline: flag nothing (no model) — all fraud goes undetected
        baseline_cost = (tp + fn) * self.fn_cost

        # Naive rule: flag all transactions > $500 — catches ~60% of fraud
        # but creates massive false positives (rough estimate)
        naive_fp_est  = int((tp + tn) * 0.15)
        naive_fn_est  = int((tp + fn) * 0.4)
        naive_cost    = naive_fp_est * self.fp_cost + naive_fn_est * self.fn_cost

        net_savings_vs_nothing = baseline_cost - total_cost
        net_savings_vs_naive   = naive_cost - total_cost

        per_1000_txns = net_savings_vs_nothing / max((tp + fp + tn + fn), 1) * 1000

        return {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "cost_false_positives":      round(cost_fp, 2),
            "cost_false_negatives":      round(cost_fn, 2),
            "total_model_cost":          round(total_cost, 2),
            "baseline_cost_no_model":    round(baseline_cost, 2),
            "naive_rule_cost":           round(naive_cost, 2),
            "net_savings_vs_no_model":   round(net_savings_vs_nothing, 2),
            "net_savings_vs_naive_rule": round(net_savings_vs_naive, 2),
            "savings_per_1000_txns":     round(per_1000_txns, 2),
            "fraud_caught_pct":          round(tp / max(tp + fn, 1), 4),
            "false_alert_rate":          round(fp / max(fp + tn, 1), 4),
        }

    def threshold_cost_curve(self, model, X_test, y_test) -> go.Figure:
        """
        Plot total cost vs decision threshold.
        Helps find the optimal threshold for a given business context.
        """
        from sklearn.metrics import confusion_matrix

        y_prob     = model.predict_proba(X_test)[:, 1]
        thresholds = np.arange(0.01, 1.0, 0.01)
        costs      = []
        recalls    = []
        precisions = []

        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            cost = fp * self.fp_cost + fn * self.fn_cost
            costs.append(cost)
            recalls.append(tp / max(tp + fn, 1))
            precisions.append(tp / max(tp + fp, 1))

        best_idx   = int(np.argmin(costs))
        best_thresh = thresholds[best_idx]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=thresholds, y=costs, mode="lines",
            name="Total cost ($)",
            line=dict(color="#3b82f6", width=2),
        ))
        fig.add_vline(
            x=best_thresh,
            line_dash="dash", line_color="#34d399",
            annotation_text=f"Optimal: {best_thresh:.2f}",
            annotation_font=dict(color="#34d399", size=11),
        )
        fig.update_layout(
            title=dict(text="Business cost vs decision threshold",
                       font=dict(size=14)),
            xaxis_title="Decision threshold",
            yaxis_title="Total cost ($)",
            height=360,
            plot_bgcolor="#0d0f17",
            paper_bgcolor="#0d0f17",
            font=dict(color="#9ca3af", size=11,
                      family="JetBrains Mono, monospace"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)",
                       linecolor="#1c2033"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)",
                       linecolor="#1c2033", tickprefix="$"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=60, r=30, t=50, b=45),
        )
        return fig

    def savings_summary_table(self, results: dict) -> pd.DataFrame:
        """
        Compare business impact across all trained models.
        """
        rows = []
        for name, r in results.items():
            m  = r["metrics"]
            bi = self.compute_savings(m["tp"], m["fp"], m["tn"], m["fn"])
            rows.append({
                "Model":                 name,
                "Fraud caught":          f"{bi['fraud_caught_pct']:.1%}",
                "False alert rate":      f"{bi['false_alert_rate']:.2%}",
                "Savings vs no model":   f"${bi['net_savings_vs_no_model']:,.0f}",
                "Savings vs naive rule": f"${bi['net_savings_vs_naive_rule']:,.0f}",
                "Per 1,000 txns":        f"${bi['savings_per_1000_txns']:,.2f}",
            })
        return pd.DataFrame(rows).set_index("Model")
