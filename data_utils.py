from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "actual_price_num",
    "discounted_price_num",
    "discount_pct",
    "rating_count_num",
]
CATEGORICAL_FEATURES = ["main_category"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "amazon.csv"


def parse_price(value) -> float | None:
    if pd.isna(value):
        return None
    cleaned = str(value).replace("₹", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_count(value) -> int:
    if pd.isna(value):
        return 0
    cleaned = str(value).replace(",", "").strip()
    try:
        return int(cleaned)
    except ValueError:
        return 0


def load_raw_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["actual_price_num"] = df["actual_price"].apply(parse_price)
    df["discounted_price_num"] = df["discounted_price"].apply(parse_price)
    df["discount_pct"] = (
        df["discount_percentage"].astype(str).str.replace("%", "", regex=False)
    )
    df["discount_pct"] = pd.to_numeric(df["discount_pct"], errors="coerce")
    df["rating_count_num"] = df["rating_count"].apply(parse_count)
    df["rating_num"] = pd.to_numeric(df["rating"], errors="coerce")
    df["main_category"] = df["category"].astype(str).str.split("|").str[0]

    return df.dropna(
        subset=NUMERIC_FEATURES + ["rating_num"],
    )


def load_and_prepare(path: Path = DEFAULT_DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Ngarkon veçoritë dhe etiketën binare për klasifikim."""
    df = load_raw_dataframe(path)
    df["high_rating"] = (df["rating_num"] >= 4.0).astype(int)
    return df[FEATURE_COLUMNS].copy(), df["high_rating"]


def load_for_clustering(
    path: Path = DEFAULT_DATA_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    """Ngarkon veçoritë dhe etiketën (etiketa përdoret vetëm pas grupimit)."""
    df = load_raw_dataframe(path)
    df["high_rating"] = (df["rating_num"] >= 4.0).astype(int)
    return df[FEATURE_COLUMNS].copy(), df["high_rating"]


def preprocess_features(X: pd.DataFrame) -> np.ndarray:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return preprocessor.fit_transform(X)
