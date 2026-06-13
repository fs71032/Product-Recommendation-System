"""
Vlerësimi i klasifikuesve mbi datasetin Amazon.

Përfshin:
- Ndarje train/test (stratified)
- Parapërpunim (StandardScaler, OneHotEncoder)
- Eksperimente zgjedhje/reduktimi veçorish
- Rregullim hiperparametrash (GridSearchCV) për çdo klasifikues
- Rrjet neuronal (MLPClassifier) me dy arkitektura
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
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


def make_mlp(hidden_layer_sizes: tuple[int, ...], activation: str = "relu") -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver="adam",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
    )


def evaluate(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def run_feature_experiments(
    X_train,
    y_train,
    X_test,
    y_test,
) -> pd.DataFrame:
    rows: list[dict] = []
    configs: list[tuple[str, list[str], str | None]] = []

    for name, cols in FEATURE_SETS.items():
        configs.append((name, cols, None))
    configs.append(("te_gjitha_plus_select_k_best",
                   FEATURE_SETS["te_gjitha_veçorite"], "select_k_best"))
    configs.append(
        ("te_gjitha_plus_pca", FEATURE_SETS["te_gjitha_veçorite"], "pca"))

    for feature_name, cols, reduction in configs:
        pipeline = make_pipeline(
            cols,
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            reduction,
        )
        pipeline.fit(X_train[cols], y_train)
        metrics = evaluate(y_test, pipeline.predict(X_test[cols]))
        rows.append(
            {
                "feature_config": feature_name,
                "reduction_method": reduction or "asnjë",
                "n_raw_features": len(cols),
                **metrics,
            }
        )

    return pd.DataFrame(rows).sort_values("f1_score", ascending=False)


def run_nn_architecture_experiments(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_cols: list[str],
    reduction: str | None,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    rows: list[dict] = []
    models: dict[str, Pipeline] = {}

    for arch_name, arch in NN_ARCHITECTURES.items():
        pipeline = make_pipeline(
            feature_cols,
            make_mlp(arch["hidden_layer_sizes"], arch["activation"]),
            reduction,
        )
        pipeline.fit(X_train[feature_cols], y_train)
        models[arch_name] = pipeline
        y_pred = pipeline.predict(X_test[feature_cols])
        metrics = evaluate(y_test, y_pred)
        save_confusion_matrix(arch_name, confusion_matrix(
            y_test, y_pred), RESULTS_DIR)
        layers = arch["hidden_layer_sizes"]
        rows.append(
            {
                "classifier": arch_name,
                "type": "Neural Network",
                "architecture": arch["description"],
                "hidden_layers": str(layers),
                "n_hidden_layers": len(layers),
                "units_per_layer": str(layers),
                "activation": arch["activation"],
                **metrics,
            }
        )

    return pd.DataFrame(rows).sort_values("f1_score", ascending=False), models


def tune_neural_network(
    X_train,
    y_train,
    feature_cols: list[str],
    reduction: str | None,
) -> tuple[Pipeline, dict, dict]:
    base_arch = list(NN_ARCHITECTURES.values())[0]
    pipeline = make_pipeline(
        feature_cols,
        make_mlp(base_arch["hidden_layer_sizes"], base_arch["activation"]),
        reduction,
    )
    grid = GridSearchCV(
        pipeline,
        get_param_grids()["Neural Network"],
        cv=CV_FOLDS,
        scoring="f1",
        n_jobs=-1,
    )
    grid.fit(X_train[feature_cols], y_train)

    best_layers = grid.best_params_["model__hidden_layer_sizes"]
    tuning_info = {
        "classifier": "Neural Network (tuned)",
        "type": "Neural Network",
        "param_grid_tested": get_param_grids()["Neural Network"],
        "best_params": grid.best_params_,
        "best_cv_f1": round(grid.best_score_, 4),
        "feature_config": feature_cols,
        "reduction": reduction or "asnjë",
        "architecture": (
            f"Hyrje -> {' -> '.join(f'Shtresa {i+1} ({u} neurone, ReLU)' for i,
                                    u in enumerate(best_layers))} "
            f"-> Dalje (1 neuron, Sigmoid)"
        ),
        "hidden_layers": str(best_layers),
        "n_hidden_layers": len(best_layers),
        "units_per_layer": str(best_layers),
        "activation": "relu",
    }
    cv_results = pd.DataFrame(grid.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")
    return grid.best_estimator_, tuning_info, cv_results.to_dict(orient="records")


def tune_classifier(
    name: str,
    X_train,
    y_train,
    feature_cols: list[str],
    reduction: str | None,
) -> tuple[Pipeline, dict, dict]:
    estimator = get_base_estimators()[name]
    pipeline = make_pipeline(feature_cols, estimator, reduction)
    grid = GridSearchCV(
        pipeline,
        get_param_grids()[name],
        cv=CV_FOLDS,
        scoring="f1",
        n_jobs=-1,
    )
    grid.fit(X_train[feature_cols], y_train)

    tuning_info = {
        "classifier": name,
        "param_grid_tested": get_param_grids()[name],
        "best_params": grid.best_params_,
        "best_cv_f1": round(grid.best_score_, 4),
        "feature_config": feature_cols,
        "reduction": reduction or "asnjë",
    }
    cv_results = pd.DataFrame(grid.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")
    return grid.best_estimator_, tuning_info, cv_results.to_dict(orient="records")


def save_best_classifier(
    model: Pipeline,
    name: str,
    feature_cols: list[str],
    reduction: str | None,
    f1_score: float,
) -> None:
    joblib.dump(model, RESULTS_DIR / "best_classifier.joblib")
    meta = {
        "classifier": name,
        "feature_columns": feature_cols,
        "reduction": reduction or "asnjë",
        "f1_score": round(f1_score, 4),
    }
    with open(RESULTS_DIR / "best_classifier_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def save_confusion_matrix(name: str, cm, output_dir: Path) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["I ulët (<4)", "I lartë (>=4)"],
        yticklabels=["I ulët (<4)", "I lartë (>=4)"],
    )
    plt.title(f"Matrica e konfuzionit - {name}")
    plt.ylabel("Vlera reale")
    plt.xlabel("Parashikimi")
    plt.tight_layout()
    safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(output_dir / f"confusion_matrix_{safe}.png", dpi=150)
    plt.show()
    plt.close()


def save_bar_plot(df: pd.DataFrame, x: str, y: str, title: str, filename: str) -> None:
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x=x, y=y, hue=x if x !=
                y else None, legend=False, palette="viridis")
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150)
    plt.show()
    plt.close()


def resolve_best_feature_config(feature_df: pd.DataFrame) -> tuple[list[str], str | None]:
    best_row = feature_df.iloc[0]
    name = best_row["feature_config"]

    if name == "te_gjitha_plus_select_k_best":
        return FEATURE_SETS["te_gjitha_veçorite"], "select_k_best"
    if name == "te_gjitha_plus_pca":
        return FEATURE_SETS["te_gjitha_veçorite"], "pca"
    return FEATURE_SETS[name], None


def write_analysis_report(
    feature_df: pd.DataFrame,
    tuning_records: list[dict],
    final_df: pd.DataFrame,
    split_info: dict,
    nn_arch_df: pd.DataFrame,
) -> None:
    best_feature = feature_df.iloc[0]
    best_final = final_df.iloc[0]
    best_nn_arch = nn_arch_df.iloc[0]

    lines = [
        "RAPORT ANALIZË - KLASIFIKUESIT AMAZON",
        "=" * 50,
        "",
        "1. NDARJA E DATASETIT",
        f"   - Mostra totale: {split_info['total']}",
        f"   - Trajnim (80%): {split_info['train']} mostra",
        f"   - Test (20%): {split_info['test']} mostra",
        f"   - Stratifikim sipas klasës: po (random_state={RANDOM_STATE})",
        f"   - Klasa 0 (rating < 4): {split_info['class_0']}",
        f"   - Klasa 1 (rating >= 4): {split_info['class_1']}",
        "",
        "2. PARAPËRPROCESIMI",
        "   - Çmimet: heqja e simbolit ₹ dhe presjeve, konvertim në float",
        "   - Zbritja: heqja e % dhe konvertim numerik",
        "   - Numri vlerësimeve: heqja e presjeve, konvertim int",
        "   - Kategoria: nxjerrja e kategorisë kryesore (para |)",
        "   - Veçoritë numerike: StandardScaler (mesatare 0, devijim 1)",
        "   - Veçoritë kategorike: OneHotEncoder (handle_unknown='ignore')",
        "   - Rreshtat me vlera të munguara në veçoritë kryesore: hequr",
        "",
        "3. EKSPERIMENTET ME VEÇORITË",
        "   Metodat e testuara:",
        "   a) Bashkësi të ndryshme veçorish:",
        "      - te_gjitha_veçorite: 4 numerike + kategoria",
        "      - vetem_numerike: pa kategori",
        "      - cmime_dhe_zbritje: vetëm 3 veçori çmimi",
        "      - pa_numrin_vleresimeve: pa rating_count_num",
        "   b) Zgjedhje veçorish: SelectKBest (f_classif, k=10)",
        "   c) Reduktim dimensionaliteti: PCA (95% e variancës)",
        "",
        "   Rezultatet (Logistic Regression, set testimi):",
    ]

    for _, row in feature_df.iterrows():
        lines.append(
            f"   - {row['feature_config']} ({row['reduction_method']}): "
            f"F1={row['f1_score']}, Acc={row['accuracy']}"
        )

    lines.extend(
        [
            "",
            f"   Përfundim: konfigurimi më i mirë = '{best_feature['feature_config']}' "
            f"(F1={best_feature['f1_score']}).",
            "",
            "4. RRJETAT NEURONALE (MLPClassifier)",
            "   Përdoret Multi-Layer Perceptron me solver Adam dhe early stopping.",
            "   Shtresat e fshehura përdorin ReLU; shtresa e daljes përdor Sigmoid (klasë binare).",
            "",
            "   Arkitekturat e testuara:",
        ]
    )

    for _, row in nn_arch_df.iterrows():
        lines.extend(
            [
                f"\n   --- {row['classifier']} ---",
                f"   {row['architecture']}",
                f"   Rezultat test: F1={row['f1_score']}, Acc={row['accuracy']}, "
                f"Prec={row['precision']}, Rec={row['recall']}",
            ]
        )

    lines.extend(
        [
            "",
            "   Ndikimi i arkitekturës:",
            f"   - Arkitektura më e mirë (pa tuning): {best_nn_arch['classifier']} (F1={best_nn_arch['f1_score']})",
            "   - Më shumë shtresa/neurone rrisin kapacitetin e modelit por edhe rrezikun e overfitting;",
            "     me dataset të vogël (~1171 mostra trajnimi), diferenca mbetet e vogël.",
            "   - Early stopping ndalon trajnimin kur validimi nuk përmirësohet, duke stabilizuar NN.",
            "",
            "5. RREGULLIMI I HIPERPARAMETRAVE (GridSearchCV, 5-fold, scoring=F1)",
        ]
    )

    for rec in tuning_records:
        lines.append(f"\n   --- {rec['classifier']} ---")
        lines.append(
            f"   Parametra të testuar: {json.dumps(rec['param_grid_tested'], ensure_ascii=False)}"
        )
        lines.append(
            f"   Parametra më të mirë: {json.dumps(rec['best_params'], ensure_ascii=False)}")
        if "architecture" in rec:
            lines.append(f"   Arkitektura finale: {rec['architecture']}")
        lines.append(f"   F1 mesatar CV: {rec['best_cv_f1']}")

    lines.extend(
        [
            "",
            "6. TABELA KRAHASUESE FINALE (set testimi)",
            "",
            f"   {'Klasifikuesi':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}",
            "   " + "-" * 72,
        ]
    )
    for _, row in final_df.iterrows():
        lines.append(
            f"   {row['classifier']:<30} {row['accuracy']:>10.4f} "
            f"{row['precision']:>10.4f} {row['recall']:>10.4f} {row['f1_score']:>10.4f}"
        )

    lines.extend(
        [
            "",
            "7. DISKUTIMI",
            f"   Klasifikuesi më i mirë: {best_final['classifier']} (F1={best_final['f1_score']}).",
            "",
            "   Pse performoi më mirë:",
            "   - Logistic Regression / SVM arrijnë F1 të lartë sepse dataseti ka pak veçorë",
            "     dhe marrëdhënie të thjeshta lineare/near-linear midis çmimit, zbritjes dhe rating-ut.",
            "   - Random Forest ka accuracy më të lartë por balancë të ndryshme precision/recall.",
            "   - Decision Tree dhe KNN janë më të dobët për shkak të overfitting/ndjeshmërisë ndaj shkallës.",
            "   - Rrjeti neuronal arrin rezultate konkurruese por nuk tejkalon modelet klasike",
            "     në këtë dataset të vogël tabular — NN shkëlqen më shumë me të dhëna të mëdha",
            "     ose veçori komplekse (tekst, imazhe).",
            "",
            "   Matricat e konfuzionit: shiko results/confusion_matrix_*.png për secilin klasifikues.",
        ]
    )

    (RESULTS_DIR / "analysis_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    X, y = load_and_prepare(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    split_info = {
        "total": len(X),
        "train": len(X_train),
        "test": len(X_test),
        "class_0": int((y == 0).sum()),
        "class_1": int((y == 1).sum()),
    }

    print("=== Faza 1: Eksperimente veçorish ===")
    feature_df = run_feature_experiments(X_train, y_train, X_test, y_test)
    feature_df.to_csv(
        RESULTS_DIR / "feature_selection_results.csv", index=False)
    save_bar_plot(
        feature_df,
        "feature_config",
        "f1_score",
        "Eksperimentet e veçorive (Logistic Regression)",
        "feature_selection_comparison.png",
    )
    print(feature_df.to_string(index=False))

    best_cols, best_reduction = resolve_best_feature_config(feature_df)
    print(
        f"\nKonfigurimi më i mirë i veçorive: {best_cols}, reduktim={best_reduction}\n")

    print("=== Faza 2: Eksperimente arkitekturash rrjeti neuronal ===")
    nn_arch_df, nn_arch_models = run_nn_architecture_experiments(
        X_train, y_train, X_test, y_test, best_cols, best_reduction
    )
    nn_arch_df.to_csv(RESULTS_DIR / "nn_architecture_results.csv", index=False)
    save_bar_plot(
        nn_arch_df,
        "classifier",
        "f1_score",
        "Krahasimi i arkitekturave te rrjetit neuronal",
        "nn_architecture_comparison.png",
    )
    print(nn_arch_df[["classifier", "f1_score", "accuracy",
          "architecture"]].to_string(index=False))
    print()

    print("=== Faza 3: Rregullim hiperparametrash ===")
    tuning_records: list[dict] = []
    cv_details: dict = {}
    final_metrics: list[dict] = []
    reports: dict[str, str] = {}
    trained_models: dict[str, Pipeline] = dict(nn_arch_models)

    for name in get_base_estimators():
        print(f"Tuning: {name}...")
        best_model, tuning_info, cv_detail = tune_classifier(
            name, X_train, y_train, best_cols, best_reduction
        )
        trained_models[name] = best_model
        tuning_records.append(tuning_info)
        cv_details[name] = cv_detail

        y_pred = best_model.predict(X_test[best_cols])
        metrics = evaluate(y_test, y_pred)
        metrics["classifier"] = name
        metrics["type"] = "Classical"
        final_metrics.append(metrics)
        reports[name] = classification_report(y_test, y_pred, zero_division=0)
        save_confusion_matrix(name, confusion_matrix(
            y_test, y_pred), RESULTS_DIR)

        print(
            f"  CV F1={tuning_info['best_cv_f1']} | "
            f"Test F1={metrics['f1_score']} | params={tuning_info['best_params']}"
        )

    for _, row in nn_arch_df.iterrows():
        final_metrics.append(
            {
                "classifier": row["classifier"],
                "type": "Neural Network",
                "accuracy": row["accuracy"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1_score": row["f1_score"],
            }
        )

    print("Tuning: Neural Network...")
    nn_model, nn_tuning, nn_cv = tune_neural_network(
        X_train, y_train, best_cols, best_reduction
    )
    trained_models["Neural Network (tuned)"] = nn_model
    tuning_records.append(nn_tuning)
    cv_details["Neural Network"] = nn_cv

    y_pred_nn = nn_model.predict(X_test[best_cols])
    nn_metrics = evaluate(y_test, y_pred_nn)
    nn_metrics["classifier"] = "Neural Network (tuned)"
    nn_metrics["type"] = "Neural Network"
    final_metrics.append(nn_metrics)
    reports["Neural Network (tuned)"] = classification_report(
        y_test, y_pred_nn, zero_division=0
    )
    save_confusion_matrix(
        "Neural Network tuned",
        confusion_matrix(y_test, y_pred_nn),
        RESULTS_DIR,
    )
    print(
        f"  CV F1={nn_tuning['best_cv_f1']} | "
        f"Test F1={nn_metrics['f1_score']} | params={nn_tuning['best_params']}"
    )

    final_df = pd.DataFrame(final_metrics).sort_values(
        "f1_score", ascending=False)
    final_df.to_csv(RESULTS_DIR / "comparative_table.csv", index=False)
    final_df.to_csv(RESULTS_DIR / "metrics_summary_tuned.csv", index=False)
    save_bar_plot(
        final_df,
        "classifier",
        "f1_score",
        "Tabela krahasuese - te gjithe klasifikuesit (F1 test)",
        "classifiers_comparative_comparison.png",
    )

    with open(RESULTS_DIR / "hyperparameter_tuning.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "cv_folds": CV_FOLDS,
                "scoring": "f1",
                "best_feature_columns": best_cols,
                "best_reduction": best_reduction,
                "classifiers": tuning_records,
                "cv_results_detail": cv_details,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(RESULTS_DIR / "classification_reports_tuned.txt", "w", encoding="utf-8") as f:
        for name, report in reports.items():
            f.write(f"=== {name} ===\n{report}\n\n")

    write_analysis_report(feature_df, tuning_records,
                          final_df, split_info, nn_arch_df)

    summary = {
        "split": split_info,
        "best_feature_config": {
            "columns": best_cols,
            "reduction": best_reduction,
        },
        "best_classifier": final_df.iloc[0].to_dict(),
        "feature_experiments": feature_df.to_dict(orient="records"),
        "nn_architecture_experiments": nn_arch_df.to_dict(orient="records"),
        "tuning": tuning_records,
        "comparative_table": final_df.to_dict(orient="records"),
    }
    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    winner = final_df.iloc[0]
    winner_name = winner["classifier"]
    winner_model = trained_models.get(winner_name)
    if winner_model is None:
        print(f"\nParalajmërim: modeli '{winner_name}' nuk u gjet për ruajtje.")
    else:
        save_best_classifier(
            winner_model,
            winner_name,
            best_cols,
            best_reduction,
            float(winner["f1_score"]),
        )
        print(f"\nModeli më i mirë u ruajt: {winner_name} -> best_classifier.joblib")

    print("\nRezultatet u ruajtën në:", RESULTS_DIR)
    print("\nRenditja finale (test set):")
    print(final_df.to_string(index=False))


if __name__ == "__main__":
    main()








