"""
download_data.py
----------------
Auto-downloads creditcard.csv from Kaggle if not present.
Called automatically by the app on startup.
"""

import os
from pathlib import Path


def download_if_needed():
    data_path = Path(__file__).parent / "data" / "creditcard.csv"

    if data_path.exists():
        return  # Already downloaded

    data_path.parent.mkdir(exist_ok=True)

    try:
        import kaggle
        print("Downloading creditcard.csv from Kaggle...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "mlg-ulb/creditcardfraud",
            path=str(data_path.parent),
            unzip=True,
        )
        print("Download complete.")
    except Exception as e:
        raise RuntimeError(
            f"Could not download dataset: {e}\n"
            "Make sure KAGGLE_USERNAME and KAGGLE_KEY are set in Streamlit secrets."
        )


if __name__ == "__main__":
    download_if_needed()
