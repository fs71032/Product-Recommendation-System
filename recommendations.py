"""
Rekomandime produktesh Amazon.

Metodat:
- Content-based: produkte të ngjashme (veçori: çmim, kategori, zbritje)
- Cluster-based: produkte nga i njëjti cluster K-Means
- Classifier-based: produkte të ngjashme që modeli parashikon si high_rating
"""

from __future__ import annotations

import json
from pathlib import Path

from plot_helpers import cleanup_matplotlib, configure_matplotlib, save_plot

configure_matplotlib()
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from data_utils import DEFAULT_DATA_PATH, FEATURE_COLUMNS, load_raw_dataframe, preprocess_features

DATA_PATH = DEFAULT_DATA_PATH
RESULTS_DIR = Path(__file__).parent / "results"
RECOMMENDATIONS_VIZ_DIR = RESULTS_DIR / "recommendations"
CLASSIFIER_MODEL_PATH = RESULTS_DIR / "best_classifier.joblib"
CLASSIFIER_META_PATH = RESULTS_DIR / "best_classifier_meta.json"
RANDOM_STATE = 42
DEFAULT_TOP_N = 5
DEFAULT_CANDIDATE_POOL = 20


def load_catalog(path: Path = DATA_PATH) -> pd.DataFrame:
    return load_raw_dataframe(path).reset_index(drop=True)


def load_best_cluster_count(default: int = 4) -> int:
    experiments = Path(__file__).parent / "results" / "clustering" / "clustering_experiments.csv"
    if not experiments.exists():
        return default

    df = pd.read_csv(experiments)
    if df.empty or "silhouette" not in df.columns:
        return default

    best = df.sort_values("silhouette", ascending=False).iloc[0]
    return int(best["n_clusters"])


def build_recommendation_context(catalog: pd.DataFrame) -> dict:
    features = catalog[FEATURE_COLUMNS]
    X_scaled = preprocess_features(features)

    n_clusters = load_best_cluster_count()
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_STATE)
    cluster_labels = kmeans.fit_predict(X_scaled)

    nn = NearestNeighbors(n_neighbors=min(DEFAULT_TOP_N + 1, len(catalog)), metric="cosine")
    nn.fit(X_scaled)

    return {
        "catalog": catalog,
        "X_scaled": X_scaled,
        "cluster_labels": cluster_labels,
        "nn": nn,
        "n_clusters": n_clusters,
    }


def _product_row(catalog: pd.DataFrame, product_id: str) -> int:
    matches = catalog.index[catalog["product_id"] == product_id].tolist()
    if not matches:
        raise ValueError(f"Produkti '{product_id}' nuk u gjet në dataset.")
    return matches[0]


def _format_recommendations(
    catalog: pd.DataFrame,
    indices: list[int],
    scores: list[float],
    method: str,
    source_product_id: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    source = catalog.loc[catalog["product_id"] == source_product_id].iloc[0]

    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        product = catalog.iloc[idx]
        rows.append(
            {
                "source_product_id": source_product_id,
                "source_product_name": source["product_name"],
                "rank": rank,
                "recommended_product_id": product["product_id"],
                "recommended_product_name": product["product_name"],
                "main_category": product["main_category"],
                "discounted_price": product["discounted_price"],
                "rating": product["rating_num"],
                "rating_count": product["rating_count_num"],
                "method": method,
                "score": round(score, 4),
            }
        )

    return pd.DataFrame(rows)


def recommend_similar(
    product_id: str,
    ctx: dict,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    catalog: pd.DataFrame = ctx["catalog"]
    idx = _product_row(catalog, product_id)

    distances, indices = ctx["nn"].kneighbors(ctx["X_scaled"][idx : idx + 1], n_neighbors=top_n + 1)
    neighbor_indices = indices[0].tolist()
    neighbor_distances = distances[0].tolist()

    filtered_indices: list[int] = []
    filtered_scores: list[float] = []
    for neighbor_idx, distance in zip(neighbor_indices, neighbor_distances):
        if neighbor_idx == idx:
            continue
        filtered_indices.append(neighbor_idx)
        filtered_scores.append(1 - distance)
        if len(filtered_indices) >= top_n:
            break

    return _format_recommendations(
        catalog,
        filtered_indices,
        filtered_scores,
        method="content_based",
        source_product_id=product_id,
    )

def recommend_from_cluster(
    product_id: str,
    ctx: dict,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    catalog: pd.DataFrame = ctx["catalog"]
    idx = _product_row(catalog, product_id)
    cluster_id = ctx["cluster_labels"][idx]

    same_cluster = [
        i
        for i, label in enumerate(ctx["cluster_labels"])
        if label == cluster_id and i != idx
    ]
    same_cluster.sort(
        key=lambda i: (catalog.iloc[i]["rating_num"], catalog.iloc[i]["rating_count_num"]),
        reverse=True,
    )
    selected = same_cluster[:top_n]
    scores = [1.0 - (rank / (top_n + 1)) for rank in range(1, len(selected) + 1)]

    df = _format_recommendations(
        catalog,
        selected,
        scores,
        method=f"cluster_k{ctx['n_clusters']}",
        source_product_id=product_id,
    )
    df["cluster_id"] = cluster_id
    return df


def classifier_model_available() -> bool:
    return CLASSIFIER_MODEL_PATH.exists() and CLASSIFIER_META_PATH.exists()


def load_classifier() -> tuple[object, dict]:
    if not classifier_model_available():
        raise FileNotFoundError(
            "Modeli i klasifikuesit nuk u gjet. Ekzekuto së pari classifiers_evaluation.py."
        )
    with open(CLASSIFIER_META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    return joblib.load(CLASSIFIER_MODEL_PATH), meta


def _classifier_scores(model, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    predictions = model.predict(features)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(features)[:, 1]
    elif hasattr(model, "decision_function"):
        decisions = model.decision_function(features)
        scores = 1.0 / (1.0 + np.exp(-decisions))
    else:
        scores = predictions.astype(float)
    return predictions.astype(int), scores.astype(float)


def recommend_with_classifier(
    product_id: str,
    ctx: dict,
    top_n: int = DEFAULT_TOP_N,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    min_probability: float = 0.5,
) -> pd.DataFrame:
    model, meta = load_classifier()
    feature_cols = meta["feature_columns"]
    classifier_name = meta["classifier"]

    catalog: pd.DataFrame = ctx["catalog"]
    idx = _product_row(catalog, product_id)

    pool_size = min(candidate_pool + 1, len(catalog))
    nn = NearestNeighbors(n_neighbors=pool_size, metric="cosine")
    nn.fit(ctx["X_scaled"])
    _, indices = nn.kneighbors(ctx["X_scaled"][idx : idx + 1])
    candidate_indices = [i for i in indices[0].tolist() if i != idx]

    candidate_features = catalog.iloc[candidate_indices][feature_cols]
    predictions, scores = _classifier_scores(model, candidate_features)

    ranked = [
        (candidate_idx, float(score), int(pred))
        for candidate_idx, score, pred in zip(candidate_indices, scores, predictions)
        if pred == 1 and score >= min_probability
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)

    if len(ranked) < top_n:
        fallback = sorted(
            zip(candidate_indices, scores, predictions),
            key=lambda item: item[1],
            reverse=True,
        )
        seen = {item[0] for item in ranked}
        for candidate_idx, score, pred in fallback:
            if candidate_idx in seen:
                continue
            ranked.append((candidate_idx, float(score), int(pred)))
            seen.add(candidate_idx)
            if len(ranked) >= top_n:
                break

    ranked = ranked[:top_n]
    source = catalog.loc[catalog["product_id"] == product_id].iloc[0]
    rows: list[dict] = []
    for rank, (candidate_idx, score, pred) in enumerate(ranked, start=1):
        product = catalog.iloc[candidate_idx]
        rows.append(
            {
                "source_product_id": product_id,
                "source_product_name": source["product_name"],
                "rank": rank,
                "recommended_product_id": product["product_id"],
                "recommended_product_name": product["product_name"],
                "main_category": product["main_category"],
                "discounted_price": product["discounted_price"],
                "rating": product["rating_num"],
                "rating_count": product["rating_count_num"],
                "method": "classifier_filtered",
                "score": round(score, 4),
                "predicted_high_rating": pred,
                "classifier": classifier_name,
            }
        )

    return pd.DataFrame(rows)


def recommend_top_in_category(
    category: str,
    catalog: pd.DataFrame,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    subset = catalog[catalog["main_category"].str.contains(category, case=False, na=False)].copy()
    if subset.empty:
        raise ValueError(f"Nuk u gjet asnjë produkt për kategorinë '{category}'.")

    subset = subset.sort_values(["rating_num", "rating_count_num"], ascending=False).head(top_n)
    rows: list[dict] = []
    for rank, (_, product) in enumerate(subset.iterrows(), start=1):
        rows.append(
            {
                "source_product_id": "-",
                "source_product_name": f"Kategoria: {category}",
                "rank": rank,
                "recommended_product_id": product["product_id"],
                "recommended_product_name": product["product_name"],
                "main_category": product["main_category"],
                "discounted_price": product["discounted_price"],
                "rating": product["rating_num"],
                "rating_count": product["rating_count_num"],
                "method": "top_rated_category",
                "score": round(product["rating_num"], 4),
            }
        )
    return pd.DataFrame(rows)


def _short_label(name: str, max_len: int = 42) -> str:
    text = str(name).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def plot_ratings_by_method(combined: pd.DataFrame, output_dir: Path) -> None:
    methods = combined["method"].unique().tolist()
    n_methods = len(methods)
    n_cols = min(2, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(10 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )
    flat_axes = axes.flatten()

    for ax, method in zip(flat_axes, methods):
        subset = combined[combined["method"] == method].sort_values("rank").copy()
        subset["product_label"] = subset.apply(
            lambda row: f"#{int(row['rank'])} {_short_label(row['recommended_product_name'])}",
            axis=1,
        )

        sns.barplot(
            data=subset,
            y="product_label",
            x="rating",
            ax=ax,
            color="steelblue",
            order=subset["product_label"].tolist(),
        )

        x_min = max(0, subset["rating"].min() - 0.4)
        x_max = min(5.5, subset["rating"].max() + 0.5)
        if x_max - x_min < 0.6:
            x_min = max(0, x_min - 0.2)
            x_max = min(5.5, x_max + 0.2)
        ax.set_xlim(x_min, x_max)

        for bar, (_, row) in zip(ax.patches, subset.iterrows()):
            ax.text(
                bar.get_width() + 0.03,
                bar.get_y() + bar.get_height() / 2,
                f"{row['rating']:.1f}",
                va="center",
                ha="left",
                fontsize=8,
            )

        ax.set_title(method.replace("_", " "), fontsize=11)
        ax.set_xlabel("Rating")
        ax.set_ylabel("Produkti i rekomanduar")
        ax.tick_params(axis="y", labelsize=7)

    for ax in flat_axes[n_methods:]:
        ax.axis("off")

    plt.suptitle("Rating i produkteve te rekomanduara (sipas metodes)", fontsize=13)
    plt.tight_layout()
    save_plot(output_dir / "recommendation_ratings_by_method.png")


def plot_method_comparison(combined: pd.DataFrame, output_dir: Path) -> None:
    summary = (
        combined.groupby("method", as_index=False)
        .agg(avg_rating=("rating", "mean"), avg_score=("score", "mean"))
        .sort_values("avg_rating", ascending=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.barplot(
        data=summary,
        x="method",
        y="avg_rating",
        ax=axes[0],
        hue="method",
        palette="viridis",
        legend=False,
    )
    axes[0].set_title("Rating mesatar i rekomandimeve")
    axes[0].set_xlabel("Metoda")
    axes[0].set_ylabel("Rating mesatar")
    axes[0].set_ylim(0, 5.5)
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(
        data=summary,
        x="method",
        y="avg_score",
        ax=axes[1],
        hue="method",
        palette="magma",
        legend=False,
    )
    axes[1].set_title("Score mesatar i rekomandimeve")
    axes[1].set_xlabel("Metoda")
    axes[1].set_ylabel("Score mesatar")
    axes[1].tick_params(axis="x", rotation=20)

    plt.suptitle("Krahasimi i metodave te rekomandimit", fontsize=13)
    plt.tight_layout()
    save_plot(output_dir / "recommendation_method_comparison.png")


def plot_classifier_scores(classifier_df: pd.DataFrame, output_dir: Path) -> None:
    if classifier_df.empty or "score" not in classifier_df.columns:
        return

    plot_df = classifier_df.sort_values("rank").copy()
    plot_df["rank_label"] = plot_df["rank"].apply(lambda r: f"#{int(r)}")

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=plot_df,
        x="rank_label",
        y="score",
        hue="predicted_high_rating",
        palette={0: "lightcoral", 1: "seagreen"},
        dodge=False,
        order=plot_df["rank_label"].tolist(),
    )
    plt.title("Rekomandime me klasifikues: probabiliteti i high_rating")
    plt.xlabel("Renditja e rekomandimit")
    plt.ylabel("Score / probabilitet")
    plt.tight_layout()
    save_plot(output_dir / "recommendation_classifier_scores.png")


def save_recommendation_plots(
    combined: pd.DataFrame,
    classifier_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if combined.empty:
        return

    plot_ratings_by_method(combined, output_dir)
    plot_method_comparison(combined, output_dir)
    plot_classifier_scores(classifier_df, output_dir)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    catalog = load_catalog()
    ctx = build_recommendation_context(catalog)

    example_id = catalog.iloc[0]["product_id"]
    example_name = catalog.iloc[0]["product_name"]
    example_category = catalog.iloc[0]["main_category"]

    print(f"Mostra produktesh: {len(catalog)}")
    print(f"Cluster-e K-Means: {ctx['n_clusters']}\n")

    print(f"=== Produkti hyrës ===")
    print(f"ID: {example_id}")
    print(f"Emri: {example_name}\n")

    similar_df = recommend_similar(example_id, ctx)
    cluster_df = recommend_from_cluster(example_id, ctx)
    category_df = recommend_top_in_category(example_category, catalog)

    classifier_df = pd.DataFrame()
    if classifier_model_available():
        classifier_df = recommend_with_classifier(example_id, ctx)
    else:
        print(
            "Paralajmërim: modeli i klasifikuesit mungon. "
            "Ekzekuto classifiers_evaluation.py për rekomandime me classifier.\n"
        )

    print("=== Rekomandime të ngjashme (content-based) ===")
    print(
        similar_df[
            ["rank", "recommended_product_name", "rating", "score"]
        ].to_string(index=False)
    )

    print("\n=== Rekomandime nga i njëjti cluster ===")
    print(
        cluster_df[
            ["rank", "recommended_product_name", "rating", "cluster_id"]
        ].to_string(index=False)
    )

    print(f"\n=== Top produkte në kategori: {example_category} ===")
    print(
        category_df[
            ["rank", "recommended_product_name", "rating"]
        ].to_string(index=False)
    )

    if not classifier_df.empty:
        print("\n=== Rekomandime të filtruara me klasifikuesin ===")
        print(
            classifier_df[
                [
                    "rank",
                    "recommended_product_name",
                    "rating",
                    "score",
                    "predicted_high_rating",
                    "classifier",
                ]
            ].to_string(index=False)
        )

    combined_parts = [similar_df, cluster_df, category_df]
    methods = ["content_based", f"cluster_k{ctx['n_clusters']}", "top_rated_category"]
    if not classifier_df.empty:
        combined_parts.append(classifier_df)
        methods.append("classifier_filtered")

    combined = pd.concat(combined_parts, ignore_index=True)
    combined.to_csv(RESULTS_DIR / "recommendations.csv", index=False)

    summary = {
        "n_products": len(catalog),
        "n_clusters": ctx["n_clusters"],
        "example_product_id": example_id,
        "example_product_name": example_name,
        "methods": methods,
        "recommendations": combined.to_dict(orient="records"),
    }
    with open(RESULTS_DIR / "recommendations_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    save_recommendation_plots(combined, classifier_df, RECOMMENDATIONS_VIZ_DIR)

    print(f"\nRezultatet u ruajtën në: {RESULTS_DIR / 'recommendations.csv'}")
    print(f"Vizualizimet u ruajtën në: {RECOMMENDATIONS_VIZ_DIR}")
    cleanup_matplotlib()


if __name__ == "__main__":
    main()





