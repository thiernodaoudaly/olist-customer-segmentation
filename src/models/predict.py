"""
Module de prédiction du segment client.

Usage :
    from src.models.predict import predict_segment
    segment = predict_segment(client_features)
"""

import numpy as np
import pandas as pd

SEGMENT_NAMES = {
    0: "Nouveaux satisfaits",
    1: "Clients dormants",
    2: "Clients insatisfaits",
    3: "VIP multi-acheteurs",
}

SEGMENT_STRATEGIES = {
    0: "Fidélisation — offres exclusives pour 2ème achat",
    1: "Réactivation — offre de retour personnalisée",
    2: "Récupération — enquête + geste commercial",
    3: "Programme VIP — service premium",
}

FEATURES = [
    "recency",
    "monetary_log",
    "review_score_mean",
    "delivery_delay_mean",
    "frequency_log",
]


def predict_segment(
    client_features: dict,
    kmeans_model,
    scaler,
) -> dict:
    """
    Prédit le segment d'un client à partir de ses features.

    Args:
        client_features : dictionnaire avec les features brutes du client
                         {recency, monetary, review_score_mean,
                          delivery_delay_mean, frequency}
        kmeans_model    : modèle K-Means chargé depuis MLflow
        scaler          : StandardScaler chargé depuis MLflow

    Returns:
        Dictionnaire avec segment_id, segment_name, strategy
    """
    # 1. Créer un DataFrame d'une ligne
    df = pd.DataFrame([client_features])

    # 2. Appliquer log1p sur frequency et monetary
    df["frequency_log"] = np.log1p(df["frequency"])
    df["monetary_log"] = np.log1p(df["monetary"])

    # 3. Sélectionner les features dans le bon ordre
    X = df[FEATURES].values

    # 4. Scaler les features
    X_scaled = scaler.transform(
        X
    )  # scaler.transform(X) et non scaler.fit_transform(X) — on transforme avec le scaler déjà entraîné sur les données d'entraînement. fit_transform recalculerait mean/std sur ce seul client — résultat absurde.

    # 5. Prédire le cluster
    cluster_id = int(kmeans_model.predict(X_scaled)[0])

    return {
        "segment_id": cluster_id,
        "segment_name": SEGMENT_NAMES[cluster_id],
        "strategy": SEGMENT_STRATEGIES[cluster_id],
    }
