"""
Script d'entraînement du modèle de segmentation Olist.

Pipeline :
    load_olist() → build_features() → StandardScaler → KMeans → mlflow.log

Usage :
    python src/models/train.py
    python src/models/train.py --n-clusters 4 --data-dir data/raw
    python src/models/train.py --n-clusters 5 --run-name my_run

Outputs :
    - Modèle et scaler loggés dans MLflow
    - data/processed/rfm_features.csv
    - models/kmeans_model.pkl
    - models/scaler.pkl
"""

import argparse
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.data.load_data import load_olist
from src.features.build_features import build_features

# ── Constantes ────────────────────────────────────────────────────────────────

FEATURES = [
    "recency",
    "monetary_log",
    "review_score_mean",
    "delivery_delay_mean",
    "frequency_log",
]

DEFAULT_N_CLUSTERS = 4
DEFAULT_DATA_DIR = "data/raw"
DEFAULT_RUN_NAME = "kmeans_final_k4"


# ── Fonctions ─────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Entraîne le modèle de segmentation Olist"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=DEFAULT_N_CLUSTERS,
        help=f"Nombre de clusters K-Means (défaut: {DEFAULT_N_CLUSTERS})",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Dossier des données brutes (défaut: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=DEFAULT_RUN_NAME,
        help=f"Nom du run MLflow (défaut: {DEFAULT_RUN_NAME})",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state pour la reproductibilité (défaut: 42)",
    )
    return parser.parse_args()


def train(
    data_dir: str = DEFAULT_DATA_DIR,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    run_name: str = DEFAULT_RUN_NAME,
    random_state: int = 42,
) -> dict:
    """
    Pipeline complet d'entraînement du modèle de segmentation.

    Args:
        data_dir      : chemin vers les données brutes
        n_clusters    : nombre de clusters K-Means
        run_name      : nom du run MLflow
        random_state  : graine aléatoire

    Returns:
        Dictionnaire avec les métriques et chemins des artefacts
    """
    # ── Setup MLflow ──────────────────────────────────────────────────────
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("olist-segmentation")

    with mlflow.start_run(run_name=run_name):

        # ── Étape 1 : Chargement des données ─────────────────────────────
        print("\n[1/5] Chargement des données...")
        dfs = load_olist(data_dir)

        # ── Étape 2 : Feature engineering ────────────────────────────────
        print("\n[2/5] Feature engineering...")
        df_features = build_features(dfs)
        print(f"      {len(df_features)} clients · {len(FEATURES)} features")

        # Sauvegarder les features
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        features_path = "data/processed/rfm_features.csv"
        df_features.to_csv(features_path, index=False)
        print(f"      Features sauvegardées : {features_path}")

        # ── Étape 3 : Scaling ─────────────────────────────────────────────
        print("\n[3/5] Scaling (StandardScaler)...")
        X = df_features[FEATURES].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ── Étape 4 : Entraînement K-Means ───────────────────────────────
        print(f"\n[4/5] Entraînement K-Means (K={n_clusters})...")
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(X_scaled)

        # ── Métriques ─────────────────────────────────────────────────────
        silhouette = silhouette_score(X_scaled, labels)
        db_score = davies_bouldin_score(X_scaled, labels)
        inertia = kmeans.inertia_

        print(f"      Silhouette Score  : {silhouette:.4f}")
        print(f"      Davies-Bouldin    : {db_score:.4f}")
        print(f"      Inertie           : {inertia:.0f}")

        # Distribution des clusters
        df_features["cluster"] = labels
        cluster_dist = df_features["cluster"].value_counts().sort_index()
        print("\n      Distribution des clusters :")
        for cid, count in cluster_dist.items():
            pct = count / len(df_features) * 100
            print(f"      Cluster {cid} : {count:>6} clients ({pct:.1f}%)")

        # ── Étape 5 : MLflow logging ──────────────────────────────────────
        print("\n[5/5] Logging MLflow...")

        # Paramètres
        mlflow.log_param("algorithm", "kmeans")
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("features", str(FEATURES))
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_param("n_clients", len(df_features))
        mlflow.log_param("data_dir", data_dir)

        # Métriques
        mlflow.log_metric("silhouette_score", silhouette)
        mlflow.log_metric("davies_bouldin", db_score)
        mlflow.log_metric("inertia", inertia)

        # Distribution des clusters comme métriques
        for cid, count in cluster_dist.items():
            mlflow.log_metric(f"cluster_{cid}_size", count)
            mlflow.log_metric(
                f"cluster_{cid}_pct", round(count / len(df_features) * 100, 2)
            )

        # Sauvegarder modèles dans MLflow
        mlflow.sklearn.log_model(kmeans, "kmeans_model")
        mlflow.sklearn.log_model(scaler, "scaler")

        # Sauvegarder les features comme artefact
        mlflow.log_artifact(features_path)

        # Sauvegarder les modèles localement
        Path("models").mkdir(parents=True, exist_ok=True)
        kmeans_path = "models/kmeans_model.pkl"
        scaler_path = "models/scaler.pkl"

        with open(kmeans_path, "wb") as f:
            pickle.dump(kmeans, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        print(f"      Modèle sauvegardé : {kmeans_path}")
        print(f"      Scaler sauvegardé : {scaler_path}")

        run_id = mlflow.active_run().info.run_id
        print(f"\nRun MLflow : {run_id}")
        print(f"UI         : {tracking_uri}/#/experiments/")

        return {
            "run_id": run_id,
            "silhouette": silhouette,
            "davies_bouldin": db_score,
            "inertia": inertia,
            "n_clients": len(df_features),
            "kmeans_path": kmeans_path,
            "scaler_path": scaler_path,
            "features_path": features_path,
        }


# ── Point d'entrée ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    print("=" * 55)
    print("  Pipeline entraînement — Segmentation Olist")
    print("=" * 55)
    print(f"  data_dir     : {args.data_dir}")
    print(f"  n_clusters   : {args.n_clusters}")
    print(f"  run_name     : {args.run_name}")
    print(f"  random_state : {args.random_state}")
    print("=" * 55)

    metrics = train(
        data_dir=args.data_dir,
        n_clusters=args.n_clusters,
        run_name=args.run_name,
        random_state=args.random_state,
    )

    print("\n" + "=" * 55)
    print("  Entraînement terminé")
    print(f"  Silhouette : {metrics['silhouette']:.4f}")
    print(f"  Clients    : {metrics['n_clients']}")
    print("=" * 55)
