from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_utils import (
    CATEGORICAL_FEATURES,
    DEFAULT_DATA_PATH,
    NUMERIC_FEATURES,
    load_and_prepare,
)

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_PATH = DEFAULT_DATA_PATH
RESULTS_DIR = Path(__file__).parent / "results"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

NUMERIC_ALL = NUMERIC_FEATURES
CATEGORICAL = CATEGORICAL_FEATURES


FEATURE_SETS: dict[str, list[str]] = {
    "te_gjitha_veçorite": NUMERIC_ALL + CATEGORICAL,
    "vetem_numerike": NUMERIC_ALL.copy(),
    "cmime_dhe_zbritje": ["actual_price_num", "discounted_price_num", "discount_pct"],
    "pa_numrin_vleresimeve": [
        "actual_price_num",
        "discounted_price_num",
        "discount_pct",
        "main_category",
    ],
}


NN_ARCHITECTURES: dict[str, dict] = {
    "NN - 2 shtresa (64, 32)": {
        "hidden_layer_sizes": (64, 32),
        "activation": "relu",
        "description": (
            "Hyrje -> Shtresa e fshehur 1 (64 neurone, ReLU) -> "
            "Shtresa e fshehur 2 (32 neurone, ReLU) -> "
            "Dalje (1 neuron, Sigmoid)"
        ),
    },
    "NN - 3 shtresa (128, 64, 32)": {
        "hidden_layer_sizes": (128, 64, 32),
        "activation": "relu",
        "description": (
            "Hyrje -> Shtresa e fshehur 1 (128 neurone, ReLU) -> "
            "Shtresa e fshehur 2 (64 neurone, ReLU) -> "
            "Shtresa e fshehur 3 (32 neurone, ReLU) -> "
            "Dalje (1 neuron, Sigmoid)"
        ),
    },
}


def build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    numeric = [c for c in feature_cols if c in NUMERIC_ALL]
    categorical = [c for c in feature_cols if c in CATEGORICAL]

    transformers: list = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if categorical:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            )
        )
            return ColumnTransformer(transformers=transformers)


    return ColumnTransformer(transformers=transformers)

def build_reduction_step(method: str | None):
    if method == "select_k_best":
        return ("selector", SelectKBest(score_func=f_classif, k=10))
    if method == "pca":
        return ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE))
    return None


def make_pipeline(
    feature_cols: list[str],
    estimator,
    reduction: str | None = None,
) -> Pipeline:
    steps: list = [("preprocess", build_preprocessor(feature_cols))]
    reduction_step = build_reduction_step(reduction)
    if reduction_step:
        steps.append(reduction_step)
    steps.append(("model", estimator))
    return Pipeline(steps)


def get_base_estimators() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(kernel="rbf", random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(),
    }


def get_param_grids() -> dict[str, dict]:
    return {
        "Logistic Regression": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__solver": ["lbfgs", "liblinear"],
        },
        "Decision Tree": {
            "model__max_depth": [3, 5, 8, 12, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "Random Forest": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [5, 8, 12, None],
            "model__min_samples_split": [2, 5],
        },
        "SVM (RBF)": {
            "model__C": [0.1, 1.0, 10.0],
            "model__gamma": ["scale", "auto", 0.01, 0.1],
        },
        "K-Nearest Neighbors": {
            "model__n_neighbors": [3, 5, 7, 11, 15],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan"],
        },
        "Neural Network": {
            "model__hidden_layer_sizes": [(64, 32), (128, 64, 32), (256, 128)],
            "model__alpha": [0.0001, 0.001, 0.01],
            "model__learning_rate_init": [0.001, 0.01],
        },
    }

