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

def run_kmeans_experiments(
    X_scaled: np.ndarray,
    y_true,
) -> tuple[pd.DataFrame, dict]:

    rows = []
    best_run = {"score": -1}

    for n_clusters in [2, 3, 4, 5, 6]:

        for init in ["k-means++", "random"]:

            model = KMeans(
                n_clusters=n_clusters,
                init=init,
                n_init=10,
                random_state=RANDOM_STATE,
            )

            labels = model.fit_predict(X_scaled)

            metrics = evaluate_clusters(
                labels,
                y_true,
                X_scaled,
            )

            rows.append(
                {
                    "algorithm": ALG_KMEANS,
                    "n_clusters": n_clusters,
                    "init": init,
                    "linkage": "-",
                    "inertia": round(model.inertia_, 2),
                    **metrics,
                }
            )

            if (
                metrics["silhouette"]
                and metrics["silhouette"]
                > best_run.get("score", -1)
            ):
                best_run = {
                    "model": model,
                    "labels": labels,
                    "score": metrics["silhouette"],
                    "config": (
                        f"{ALG_KMEANS} "
                        f"k={n_clusters}, "
                        f"init={init}"
                    ),
                }

    return pd.DataFrame(rows), best_run

def run_agglomerative_experiments(
    X_scaled: np.ndarray,
    y_true,
) -> tuple[pd.DataFrame, dict]:

    rows = []
    best_run = {"score": -1}

    for n_clusters in [2, 3, 4, 5]:

        for linkage in [
            "ward",
            "complete",
            "average",
        ]:

            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage,
            )

            labels = model.fit_predict(X_scaled)

            metrics = evaluate_clusters(
                labels,
                y_true,
                X_scaled,
            )

            rows.append(
                {
                    "algorithm": ALG_AGLOMERATIV,
                    "n_clusters": n_clusters,
                    "init": "-",
                    "linkage": linkage,
                    "inertia": None,
                    **metrics,
                }
            )

            if (
                metrics["silhouette"]
                and metrics["silhouette"]
                > best_run.get("score", -1)
            ):
                best_run = {
                    "model": model,
                    "labels": labels,
                    "score": metrics["silhouette"],
                    "config": (
                        f"{ALG_AGLOMERATIV} "
                        f"k={n_clusters}, "
                        f"linkage={linkage}"
                    ),
                }

    return pd.DataFrame(rows), best_run

def plot_elbow(
    X_scaled: np.ndarray,
    output_dir: Path,
) -> None:

    inertias = []

    ks = range(2, 11)

    for k in ks:

        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            random_state=RANDOM_STATE,
        )

        km.fit(X_scaled)

        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(list(ks), inertias, marker="o")
    plt.xlabel("Numri i grupeve (k)")
    plt.ylabel("Inertia")
    plt.title("Metoda Elbow - K-Means")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "elbow_plot.png", dpi=150)
    plt.close()


def plot_experiment_comparison(
    exp_df: pd.DataFrame,
    output_dir: Path,
) -> None:

    plot_df = exp_df.dropna(
        subset=["silhouette"]
    ).copy()

    plot_df["label"] = (
        plot_df["algorithm"]
        + " k="
        + plot_df["n_clusters"].astype(str)
        + " "
        + plot_df["init"].where(
            plot_df["init"] != "-",
            plot_df["linkage"],
        )
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
    )

    sns.barplot(
        data=plot_df.sort_values(
            "silhouette",
            ascending=False,
        ).head(10),
        x="silhouette",
        y="label",
        hue="algorithm",
        ax=axes[0],
        dodge=False,
    )

    sns.barplot(
        data=plot_df.sort_values(
            "adjusted_rand_index",
            ascending=False,
        ).head(10),
        x="adjusted_rand_index",
        y="label",
        hue="algorithm",
        ax=axes[1],
        dodge=False,
    )

    plt.tight_layout()
    plt.savefig(
        output_dir /
        "experiment_comparison.png",
        dpi=150,
    )
    plt.close()

def plot_pca_scatter(
    X_scaled: np.ndarray,
    cluster_labels: np.ndarray,
    y_true,
    title: str,
    filename: str,
    output_dir: Path,
) -> None:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    scatter0 = axes[0].scatter(
        X_2d[:, 0], X_2d[:, 1], c=cluster_labels, cmap="tab10", alpha=0.7, s=25
    )
    axes[0].set_title(f"Grupimet ({title})")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.colorbar(scatter0, ax=axes[0], label="Grup")

    scatter1 = axes[1].scatter(
        X_2d[:, 0], X_2d[:, 1], c=y_true, cmap="coolwarm", alpha=0.7, s=25
    )
    axes[1].set_title("Etiketat e verteta (high_rating)")
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.colorbar(scatter1, ax=axes[1], label="Klasa (0/1)")

    plt.suptitle("PCA 2D - Grupimet vs Etiketat Reale", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()


def plot_cluster_distribution(
    cluster_labels: np.ndarray, y_true, algorithm: str, output_dir: Path
) -> None:
    df = pd.DataFrame({"cluster": cluster_labels, "true_label": y_true})
    crosstab = pd.crosstab(df["cluster"], df["true_label"], normalize="index") * 100

    plt.figure(figsize=(8, 5))
    sns.heatmap(
        crosstab,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        cbar_kws={"label": "% brenda grupit"},
    )
    plt.title(f"Shperndarja e klasave reale brenda grupeve - {algorithm}")
    plt.xlabel("Etiketa e vertete (0=i ulet, 1=i larte)")
    plt.ylabel("Grupi")
    plt.tight_layout()
    safe = algorithm.lower().replace(" ", "_")
    plt.savefig(output_dir / f"crosstab_{safe}.png", dpi=150)
    plt.close()


def plot_experiment_comparison(exp_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = exp_df.dropna(subset=["silhouette"]).copy()
    plot_df["label"] = (
        plot_df["algorithm"]
        + " k="
        + plot_df["n_clusters"].astype(str)
        + " "
        + plot_df["init"].where(plot_df["init"] != "-", plot_df["linkage"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.barplot(
        data=plot_df.sort_values("silhouette", ascending=False).head(10),
        x="silhouette",
        y="label",
        hue="algorithm",
        ax=axes[0],
        dodge=False,
    )
    axes[0].set_title("Silhouette Score (me i larte = me mire)")
    axes[0].set_xlabel("Silhouette")

    sns.barplot(
        data=plot_df.sort_values("adjusted_rand_index", ascending=False).head(10),
        x="adjusted_rand_index",
        y="label",
        hue="algorithm",
        ax=axes[1],
        dodge=False,
    )
    axes[1].set_title("Adjusted Rand Index vs etiketat reale")
    axes[1].set_xlabel("ARI")

    plt.tight_layout()
    plt.savefig(output_dir / "experiment_comparison.png", dpi=150)
    plt.close()

    def analyze_cluster_profiles(
    X: pd.DataFrame, cluster_labels: np.ndarray, output_dir: Path
) -> pd.DataFrame:
    profile = X.copy()
    profile["cluster"] = cluster_labels
    summary = profile.groupby("cluster")[NUMERIC].mean().round(2)
    summary["n_produkte"] = profile.groupby("cluster").size()
    top_cats = (
        profile.groupby("cluster")["main_category"]
        .agg(lambda s: s.value_counts().index[0])
        .rename("kategoria_dominante")
    )
    summary = summary.join(top_cats)
    summary.to_csv(output_dir / "cluster_profiles.csv")
    return summary


def write_report(
    exp_df: pd.DataFrame,
    best_kmeans: dict,
    best_agg: dict,
    profiles_km: pd.DataFrame,
    n_samples: int,
) -> None:
    best_ari_row = exp_df.sort_values("adjusted_rand_index", ascending=False).iloc[0]
    best_sil_row = exp_df.sort_values("silhouette", ascending=False).iloc[0]

    lines = [
        "RAPORT GRUPIMI (CLUSTERING) - DATASET AMAZON",
        "=" * 55,
        "",
        "1. METODOLOGJIA",
        f"   - Mostra: {n_samples} produkte",
        "   - Etiketat (high_rating) u hoqen PARA grupimit",
        "   - Veçoritë: çmim, zbritje, rating_count, kategoria",
        "   - Parapërpunim: StandardScaler + OneHotEncoder",
        "",
        "2. ALGORITMET E PËRDORUR",
        "   a) K-Means: n_clusters=2..6, init=k-means++/random",
        "   b) Grupim aglomerativ: n_clusters=2..5, linkage=ward/complete/average",
        "",
        "3. PARAMETRAT E EKSPERIMENTUAR",
    ]

    for _, row in exp_df.head(15).iterrows():
        extra = row["init"] if row["init"] != "-" else row["linkage"]
        lines.append(
            f"   - {row['algorithm']} k={int(row['n_clusters'])} {extra}: "
            f"Silhouette={row['silhouette']}, ARI={row['adjusted_rand_index']}, "
            f"NMI={row['normalized_mutual_info']}"
        )

    lines.extend(
        [
            "",
            "4. EFEKTI I PARAMETRAVE",
            f"   - Silhouette më i mirë: {best_sil_row['algorithm']} k={int(best_sil_row['n_clusters'])} "
            f"(Silhouette={best_sil_row['silhouette']})",
            f"   - Përputhja më e mirë me klasa: {best_ari_row['algorithm']} k={int(best_ari_row['n_clusters'])} "
            f"(ARI={best_ari_row['adjusted_rand_index']}, NMI={best_ari_row['normalized_mutual_info']})",
            "   - init=k-means++ zakonisht jep grupe më të balancuara se random",
            "   - linkage=ward (K-Means / grupim aglomerativ) funksionon mire me veçori te shkallëzuara",
            "   - k=2 përputhet me klasat binare (rating i lartë/ulët) por grupimet",
            "     kapin edhe struktura çmimi/kategori, jo vetëm rating",
            "",
            "5. KRAHASIMI ME ETIKETAT E VËRTETA",
            "   Metrikat: ARI (1= përputhje perfekte), NMI, Homogeneity, Completeness",
            f"   ARI maksimal i arritur: {exp_df['adjusted_rand_index'].max():.4f}",
            f"   NMI maksimal: {exp_df['normalized_mutual_info'].max():.4f}",
            "   Përputhja me klasa NUK është e lartë — grupimet bazohen në çmim/kategori,",
            "   ndërsa etiketa varet nga rating. Kjo tregon që struktura e produkteve",
            "   nuk përputhet plotësisht me vlerësimin e lartë/ulët.",
            "",
            "6. PROFILI I GRUPEVE (K-Means më i mirë)",
        ]
    )

    for cluster_id, row in profiles_km.iterrows():
        lines.append(
            f"   Grupi {cluster_id}: n={int(row['n_produkte'])}, "
            f"cmim mesatar={row['actual_price_num']}, "
            f"zbritje mes={row['discount_pct']}%, "
            f"kategori={row['kategoria_dominante']}"
        )

    lines.extend(
        [
            "",
            "7. GJETJE DHE NJOHURI",
            "   - Grupimet zbulojnë segmente produktesh (kabllo, TV, HDMI) bazuar në kategori/çmim",
            "   - Produktet me rating të lartë janë të shpërndara në shumë grupe — rating",
            "     nuk është faktor dominues i grupimit",
            "   - Grup i papritur: produkte të shtrenjta dhe të lira mund të bien në të njëjtin",
            "     grup nëse kanë kategori të ngjashme dhe zbritje të ngjashme",
            "   - k=4..5 ndan më mirë kategori produktesh (TV vs kabllo vs aksesorë)",
            "",
            "8. VIZUALIZIMET",
            "   - elbow_plot.png: zgjedhja e k optimal",
            "   - pca_kmeans_best.png / pca_grupim_aglomerativ.png: grupimet vs etiketat",
            "   - crosstab_*.png: shpërndarja e klasave reale brenda grupeve",
            "   - experiment_comparison.png: krahasimi i parametrave",
        ]
    )

    (RESULTS_DIR / "clustering_report.txt").write_text("\n".join(lines), encoding="utf-8")
