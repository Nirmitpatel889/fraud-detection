"""
src/preprocessor.py
-------------------
Data loading, EDA utilities, and preprocessing pipeline
for the credit card fraud detection project.

Dataset: Kaggle Credit Card Fraud Detection
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
284,807 transactions · 492 fraud cases (0.17%)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


DATA_PATH = Path(__file__).parent.parent / "data" / "creditcard.csv"


class FraudPreprocessor:
    """
    Loads and preprocesses the credit card fraud dataset.

    Steps:
        1. Load CSV
        2. Scale Amount and Time
        3. Split train/test (stratified)
        4. Apply SMOTE + undersampling to training set only
    """

    def __init__(self, data_path=DATA_PATH, test_size=0.2, random_state=42):
        self.data_path    = Path(data_path)
        self.test_size    = test_size
        self.random_state = random_state
        self.scaler       = StandardScaler()
        self.feature_cols = None

    def load(self) -> pd.DataFrame:
        """Load the raw CSV."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}\n"
                "Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
                "and place creditcard.csv in the data/ folder."
            )
        df = pd.read_csv(self.data_path)
        return df

    def eda_summary(self, df: pd.DataFrame) -> dict:
        """Return key EDA statistics."""
        n_fraud = df["Class"].sum()
        n_legit = len(df) - n_fraud
        return {
            "total_transactions": len(df),
            "fraud_count":        int(n_fraud),
            "legitimate_count":   int(n_legit),
            "fraud_rate":         float(n_fraud / len(df)),
            "avg_fraud_amount":   float(df[df["Class"] == 1]["Amount"].mean()),
            "avg_legit_amount":   float(df[df["Class"] == 0]["Amount"].mean()),
            "max_fraud_amount":   float(df[df["Class"] == 1]["Amount"].max()),
            "features":           [c for c in df.columns if c != "Class"],
        }

    def preprocess(self, df: pd.DataFrame, balance=True):
        """
        Full preprocessing pipeline.

        Parameters
        ----------
        df      : Raw DataFrame from load()
        balance : Whether to apply SMOTE + undersampling on training set

        Returns
        -------
        X_train, X_test, y_train, y_test, feature_names
        """
        # Scale Amount and Time (V1-V28 are already PCA-transformed)
        df = df.copy()
        df["Amount_scaled"] = self.scaler.fit_transform(df[["Amount"]])
        df["Time_scaled"]   = self.scaler.fit_transform(df[["Time"]])
        df.drop(["Amount", "Time"], axis=1, inplace=True)

        feature_cols = [c for c in df.columns if c != "Class"]
        self.feature_cols = feature_cols

        X = df[feature_cols].values
        y = df["Class"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )

        if balance:
            # SMOTE oversamples minority (fraud), then undersample majority
            pipeline = ImbPipeline([
                ("smote",      SMOTE(random_state=self.random_state,
                                     sampling_strategy=0.1)),
                ("undersample", RandomUnderSampler(random_state=self.random_state,
                                                    sampling_strategy=0.5)),
            ])
            X_train, y_train = pipeline.fit_resample(X_train, y_train)

        return X_train, X_test, y_train, y_test, feature_cols

    def get_sample_transactions(self, df: pd.DataFrame, n_fraud=5, n_legit=5):
        """
        Return sample transactions for dashboard demo.
        Returns a DataFrame with n fraud + n legit transactions.
        """
        fraud = df[df["Class"] == 1].sample(n_fraud, random_state=42)
        legit = df[df["Class"] == 0].sample(n_legit, random_state=42)
        return pd.concat([fraud, legit]).reset_index(drop=True)
