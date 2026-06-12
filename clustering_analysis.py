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