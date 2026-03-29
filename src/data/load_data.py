"""
Module de chargement des données Olist.

Fournit une fonction unique load_olist() qui charge les tables
nécessaires à la segmentation client depuis data/raw/.

Usage :
    from src.data.load_data import load_olist
    dfs = load_olist()
    orders = dfs["orders"]
"""
import os
import pandas as pd


# ── Constantes ────────────────────────────────────────────────────────────────

# Tables nécessaires pour la segmentation
REQUIRED_FILES = [
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_customers_dataset.csv",
]


# ── Fonction principale ───────────────────────────────────────────────────────

def load_olist(data_dir: str = "data/raw") -> dict[str, pd.DataFrame]:
    """
    Charge les tables Olist depuis le répertoire data/raw.

    Chaque CSV est chargé dans un DataFrame indépendant.
    Le nom de la clé est simplifié :
        "olist_orders_dataset.csv" -> "orders"
        "olist_order_items_dataset.csv" -> "order_items"

    Args:
        data_dir : chemin vers le dossier contenant les CSV Olist.
                   Relatif à la racine du projet.

    Returns:
        Dictionnaire {nom_table: DataFrame}
        Exemple : {"orders": df_orders, "order_items": df_items, ...}

    Raises:
        FileNotFoundError : si un fichier requis est absent de data_dir.

    Example:
        >>> dfs = load_olist("data/raw")
        >>> dfs["orders"].shape
        (99441, 8)
    """
    dfs = {}

    for filename in REQUIRED_FILES:

        # Construire le chemin complet vers le fichier
        # os.path.join gere les separateurs Windows/Linux automatiquement
        filepath = os.path.join(data_dir, filename)

        # Verifier que le fichier existe avant de l'ouvrir
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Fichier manquant : {filepath}\n"
                f"Telecharge le dataset depuis :\n"
                f"https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce\n"
                f"Et place les CSV dans : {data_dir}/"
            )

        # Extraire le nom court pour la cle du dictionnaire
        # "olist_orders_dataset.csv" -> "orders"
        # "olist_order_items_dataset.csv" -> "order_items"
        table_name = filename.replace("olist_", "").replace("_dataset.csv", "")

        # Charger le CSV dans un DataFrame
        dfs[table_name] = pd.read_csv(filepath)

        # Afficher un resume pour confirmer le chargement
        shape = dfs[table_name].shape
        print(f"OK {table_name:<25} {shape[0]:>7} lignes x {shape[1]:>2} colonnes")

    return dfs


# ── Fonction utilitaire ───────────────────────────────────────────────────────

def get_merged_dataset(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Charge et merge toutes les tables en un seul DataFrame.

    Effectue les jointures dans cet ordre :
        orders
        -> order_items      (sur order_id)
        -> order_payments   (sur order_id)
        -> order_reviews    (sur order_id)
        -> customers        (sur customer_id)

    Args:
        data_dir : chemin vers le dossier contenant les CSV Olist.

    Returns:
        DataFrame merge avec toutes les informations par commande.

    Note:
        Utilise un LEFT JOIN pour conserver toutes les commandes
        meme si certaines n'ont pas de paiement ou d'avis associe.
    """
    print("Chargement des tables...")
    dfs = load_olist(data_dir)

    print("\nMerge des tables...")

    # orders -> order_items
    df = dfs["orders"].merge(
        dfs["order_items"],
        on="order_id",
        how="left"
    )
    print(f"  orders + order_items     : {df.shape}")

    # -> order_payments (agrege par order_id pour eviter les doublons)
    payments_agg = (
        dfs["order_payments"]
        .groupby("order_id")
        .agg(
            payment_type=("payment_type", "first"),
            payment_installments=("payment_installments", "max"),
            payment_value=("payment_value", "sum"),
        )
        .reset_index()
    )
    df = df.merge(payments_agg, on="order_id", how="left")
    print(f"  + order_payments         : {df.shape}")

    # -> order_reviews (agrege par order_id — garde la moyenne des avis)
    reviews_agg = (
        dfs["order_reviews"]
        .groupby("order_id")
        .agg(review_score=("review_score", "mean"))
        .reset_index()
    )
    df = df.merge(reviews_agg, on="order_id", how="left")
    print(f"  + order_reviews          : {df.shape}")

    # -> customers
    df = df.merge(
        dfs["customers"],
        on="customer_id",
        how="left"
    )
    print(f"  + customers              : {df.shape}")

    print(f"\nDataset merge final     : {df.shape}")
    return df