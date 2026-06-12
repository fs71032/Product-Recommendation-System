from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA

from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)

from data_utils import (
    DEFAULT_DATA_PATH,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    load_for_clustering,
    preprocess_features,
)

warnings.filterwarnings("ignore")

DATA_PATH = DEFAULT_DATA_PATH
RESULTS_DIR = Path(__file__).parent / "results" / "clustering"
RANDOM_STATE = 42

ALG_KMEANS = "K-Means"
ALG_AGLOMERATIV = "Grupim aglomerativ"

NUMERIC = NUMERIC_FEATURES
FEATURE_COLS = FEATURE_COLUMNS

def evaluate_clusters(labels: np.ndarray, y_true, X_scaled: np.ndarray) -> dict:
    mask = labels >= 0

    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return {
            "silhouette": None,
            "adjusted_rand_index": None,
            "normalized_mutual_info": None,
            "homogeneity": None,
            "completeness": None,
            "v_measure": None,
            "calinski_harabasz": None,
        }

    y_eval = y_true[mask] if hasattr(y_true, "__getitem__") else y_true
    X_eval = X_scaled[mask]

    return {
        "silhouette": round(
            silhouette_score(X_eval, labels[mask]), 4
        ),
        "adjusted_rand_index": round(
            adjusted_rand_score(y_eval, labels[mask]), 4
        ),
        "normalized_mutual_info": round(
            normalized_mutual_info_score(
                y_eval,
                labels[mask]
            ),
            4,
        ),
        "homogeneity": round(
            homogeneity_score(
                y_eval,
                labels[mask]
            ),
            4,
        ),
        "completeness": round(
            completeness_score(
                y_eval,
                labels[mask]
            ),
            4,
        ),
        "v_measure": round(
            v_measure_score(
                y_eval,
                labels[mask]
            ),
            4,
        ),
        "calinski_harabasz": round(
            calinski_harabasz_score(
                X_eval,
                labels[mask]
            ),
            2,
        ),
    }