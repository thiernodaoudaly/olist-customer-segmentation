"""
Feature engineering pour la segmentation clients Olist.

Construit les features RFM enrichies à partir des tables brutes :
    - Recency   : jours depuis le dernier achat
    - Frequency : nombre de commandes
    - Monetary  : montant total dépensé
    - Satisfaction : score moyen des avis
    - Delivery  : délai moyen de livraison

Usage :
    from src.features.build_features import build_features
    from src.data.load_data import load_olist

    dfs = load_olist("data/raw")
    df_features = build_features(dfs)
"""

import numpy as np
import pandas as pd

# ── Constantes ────────────────────────────────────────────────────────────────

# Date de référence RFM — lendemain de la dernière commande du dataset
SNAPSHOT_DATE = pd.Timestamp("2018-09-01")

# Seuils pour le clipping du délai de livraison
DELAY_CLIP_MIN = -30
DELAY_CLIP_MAX = 30


# ── Fonctions par feature ─────────────────────────────────────────────────────


def compute_recency(
    df_orders: pd.DataFrame,
    df_customers: pd.DataFrame,
    snapshot_date: pd.Timestamp = SNAPSHOT_DATE,
) -> pd.DataFrame:
    """
    Calcule la récence : nb de jours entre le dernier achat et snapshot_date.

    Un client récent a une petite valeur de récence.

    Args:
        df_orders    : table orders (filtrée sur delivered)
        df_customers : table customers
        snapshot_date: date de référence pour le calcul

    Returns:
        DataFrame avec colonnes [customer_unique_id, recency]
    """
    # Convertir la date d'achat en datetime si pas encore fait
    df = df_orders.copy()
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])

    # Joindre avec customers pour avoir customer_unique_id
    df = df.merge(
        df_customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="left",
    )

    # Dernière date d'achat par client unique
    recency = (
        df.groupby("customer_unique_id")["order_purchase_timestamp"].max().reset_index()
    )

    # Récence = nb de jours depuis le dernier achat
    recency["recency"] = (snapshot_date - recency["order_purchase_timestamp"]).dt.days

    return recency[["customer_unique_id", "recency"]]


def compute_frequency(
    df_orders: pd.DataFrame,
    df_customers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcule la fréquence : nombre de commandes par client unique.

    Args:
        df_orders    : table orders (filtrée sur delivered)
        df_customers : table customers

    Returns:
        DataFrame avec colonnes [customer_unique_id, frequency]
    """
    df = df_orders.merge(
        df_customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="left",
    )

    frequency = (
        df.groupby("customer_unique_id")["order_id"]
        .nunique()
        .reset_index(name="frequency")
    )

    return frequency


def compute_monetary(
    df_orders: pd.DataFrame,
    df_customers: pd.DataFrame,
    df_items: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcule le montant total dépensé par client (prix + frais de port).

    Args:
        df_orders    : table orders (filtrée sur delivered)
        df_customers : table customers
        df_items     : table order_items

    Returns:
        DataFrame avec colonnes [customer_unique_id, monetary]
    """
    # Montant total par commande
    items_agg = (
        df_items.groupby("order_id")
        .agg(order_value=("price", "sum"), freight_value=("freight_value", "sum"))
        .reset_index()
    )
    items_agg["total_value"] = items_agg["order_value"] + items_agg["freight_value"]

    # Joindre orders + customers + montants
    df = (
        df_orders[["order_id", "customer_id"]]
        .merge(
            df_customers[["customer_id", "customer_unique_id"]],
            on="customer_id",
            how="left",
        )
        .merge(items_agg[["order_id", "total_value"]], on="order_id", how="left")
    )

    # Montant total par client unique
    monetary = (
        df.groupby("customer_unique_id")["total_value"]
        .sum()
        .reset_index(name="monetary")
    )

    return monetary


def compute_satisfaction(
    df_orders: pd.DataFrame,
    df_customers: pd.DataFrame,
    df_reviews: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcule le score de satisfaction moyen par client.

    Args:
        df_orders    : table orders (filtrée sur delivered)
        df_customers : table customers
        df_reviews   : table order_reviews

    Returns:
        DataFrame avec colonnes [customer_unique_id, review_score_mean]
    """
    df = (
        df_orders[["order_id", "customer_id"]]
        .merge(
            df_customers[["customer_id", "customer_unique_id"]],
            on="customer_id",
            how="left",
        )
        .merge(df_reviews[["order_id", "review_score"]], on="order_id", how="left")
    )

    satisfaction = (
        df.groupby("customer_unique_id")["review_score"]
        .mean()
        .reset_index(name="review_score_mean")
    )

    return satisfaction


def compute_delivery(
    df_orders: pd.DataFrame,
    df_customers: pd.DataFrame,
    clip_min: int = DELAY_CLIP_MIN,
    clip_max: int = DELAY_CLIP_MAX,
) -> pd.DataFrame:
    """
    Calcule le délai moyen de livraison (réel vs estimé) par client.

    Valeur négative = livré en avance
    Valeur positive = livré en retard

    Les outliers sont clippés entre clip_min et clip_max.

    Args:
        df_orders : table orders (filtrée sur delivered)
        df_customers : table customers
        clip_min  : borne inférieure du clipping (défaut -30 jours)
        clip_max  : borne supérieure du clipping (défaut +30 jours)

    Returns:
        DataFrame avec colonnes [customer_unique_id, delivery_delay_mean]
    """
    df = df_orders.copy()

    # Convertir les dates
    df["order_delivered_customer_date"] = pd.to_datetime(
        df["order_delivered_customer_date"]
    )
    df["order_estimated_delivery_date"] = pd.to_datetime(
        df["order_estimated_delivery_date"]
    )

    # Calcul du délai en jours
    df["delivery_delay"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.days

    # Clipping des outliers
    df["delivery_delay"] = df["delivery_delay"].clip(clip_min, clip_max)

    # Joindre avec customers
    df = df.merge(
        df_customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="left",
    )

    # Délai moyen par client
    delivery = (
        df.groupby("customer_unique_id")["delivery_delay"]
        .mean()
        .reset_index(name="delivery_delay_mean")
    )

    return delivery


# ── Fonction principale ───────────────────────────────────────────────────────


def build_features(
    dfs: dict,
    snapshot_date: pd.Timestamp = SNAPSHOT_DATE,
    apply_log: bool = True,
) -> pd.DataFrame:
    """
    Assemble toutes les features RFM enrichies en un seul DataFrame.

    Pipeline :
        1. Filtre les commandes livrées
        2. Calcule chaque feature indépendamment
        3. Merge tout sur customer_unique_id
        4. Impute les valeurs manquantes
        5. Applique les transformations (log1p)

    Args:
        dfs          : dictionnaire de DataFrames chargé par load_olist()
        snapshot_date: date de référence pour la récence
        apply_log    : si True, applique log1p sur frequency et monetary

    Returns:
        DataFrame avec une ligne par client et les colonnes :
        [customer_unique_id, recency, frequency, monetary,
         review_score_mean, delivery_delay_mean]
    """
    print("Building features...")

    # ── Étape 1 : filtrer les commandes livrées ───────────────────────────
    orders_clean = dfs["orders"][dfs["orders"]["order_status"] == "delivered"].copy()
    orders_clean["order_purchase_timestamp"] = pd.to_datetime(
        orders_clean["order_purchase_timestamp"]
    )
    print(f"  Commandes livrées    : {len(orders_clean)}")

    # ── Étape 2 : calculer chaque feature ────────────────────────────────
    print("  Calcul recency...")
    recency = compute_recency(orders_clean, dfs["customers"], snapshot_date)

    print("  Calcul frequency...")
    frequency = compute_frequency(orders_clean, dfs["customers"])

    print("  Calcul monetary...")
    monetary = compute_monetary(orders_clean, dfs["customers"], dfs["order_items"])

    print("  Calcul satisfaction...")
    satisfaction = compute_satisfaction(
        orders_clean, dfs["customers"], dfs["order_reviews"]
    )

    print("  Calcul delivery delay...")
    delivery = compute_delivery(orders_clean, dfs["customers"])

    # ── Étape 3 : merger toutes les features ─────────────────────────────
    df = recency.copy()
    for other in [frequency, monetary, satisfaction, delivery]:
        df = df.merge(other, on="customer_unique_id", how="left")

    print(f"  Shape avant imputation : {df.shape}")

    # ── Étape 4 : imputer les valeurs manquantes ──────────────────────────
    # review_score_mean : NaN si le client n'a laissé aucun avis
    # → on impute avec la médiane (plus robuste que la moyenne)
    median_score = df["review_score_mean"].median()
    df["review_score_mean"] = df["review_score_mean"].fillna(median_score)

    # delivery_delay_mean : NaN si dates manquantes
    median_delay = df["delivery_delay_mean"].median()
    df["delivery_delay_mean"] = df["delivery_delay_mean"].fillna(median_delay)

    # Supprimer les lignes avec monetary = 0 ou NaN (pas de transaction réelle)
    df = df[df["monetary"] > 0].dropna(subset=["monetary"])

    print(f"  Shape après imputation : {df.shape}")

    # ── Étape 5 : transformations ─────────────────────────────────────────
    if apply_log:
        # log1p sur frequency et monetary pour réduire l'asymétrie
        df["frequency_log"] = np.log1p(df["frequency"])
        df["monetary_log"] = np.log1p(df["monetary"])
        print("  Log1p appliqué sur frequency et monetary")

    print(f"\nFeatures finales : {list(df.columns)}")
    print(f"Nb clients       : {len(df)}")

    return df
