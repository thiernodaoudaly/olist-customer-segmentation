"""
Tests unitaires pour src/data/load_data.py

On teste la logique des fonctions sans charger les vrais CSV.
"""
import os
import pytest
import pandas as pd

from src.data.load_data import load_olist, REQUIRED_FILES


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_data_dir(tmp_path):
    """
    Crée un répertoire temporaire avec des CSV minimaux.

    tmp_path est une fixture Pytest native qui crée un dossier
    temporaire unique pour chaque test — supprimé automatiquement après.
    """
    for filename in REQUIRED_FILES:
        # Créer un CSV minimal avec les colonnes essentielles
        if "orders" in filename and "items" not in filename:
            df = pd.DataFrame({
                "order_id":                    ["o1", "o2", "o3"],
                "customer_id":                 ["c1", "c2", "c3"],
                "order_status":                ["delivered", "delivered", "canceled"],
                "order_purchase_timestamp":    ["2018-01-01", "2018-03-01", "2018-02-01"],
                "order_approved_at":           ["2018-01-02", "2018-03-02", None],
                "order_delivered_carrier_date":["2018-01-05", "2018-03-05", None],
                "order_delivered_customer_date":["2018-01-10", "2018-03-10", None],
                "order_estimated_delivery_date":["2018-01-20", "2018-03-15", "2018-02-20"],
            })
        elif "order_items" in filename:
            df = pd.DataFrame({
                "order_id":           ["o1", "o2", "o3"],
                "order_item_id":      [1, 1, 1],
                "product_id":         ["p1", "p2", "p3"],
                "seller_id":          ["s1", "s2", "s3"],
                "shipping_limit_date":["2018-01-05", "2018-03-05", "2018-02-05"],
                "price":              [100.0, 200.0, 150.0],
                "freight_value":      [10.0, 20.0, 15.0],
            })
        elif "order_payments" in filename:
            df = pd.DataFrame({
                "order_id":             ["o1", "o2", "o3"],
                "payment_sequential":   [1, 1, 1],
                "payment_type":         ["credit_card", "boleto", "credit_card"],
                "payment_installments": [1, 1, 3],
                "payment_value":        [110.0, 220.0, 165.0],
            })
        elif "order_reviews" in filename:
            df = pd.DataFrame({
                "review_id":              ["r1", "r2", "r3"],
                "order_id":              ["o1", "o2", "o3"],
                "review_score":          [5, 4, 1],
                "review_comment_title":  [None, None, "Problème"],
                "review_comment_message":[None, "Bien", "Très déçu"],
                "review_creation_date":  ["2018-01-15", "2018-03-15", "2018-02-15"],
                "review_answer_timestamp":["2018-01-16", "2018-03-16", "2018-02-16"],
            })
        elif "customers" in filename:
            df = pd.DataFrame({
                "customer_id":              ["c1", "c2", "c3"],
                "customer_unique_id":       ["uc1", "uc2", "uc3"],
                "customer_zip_code_prefix": [12345, 23456, 34567],
                "customer_city":            ["São Paulo", "Rio", "Belo Horizonte"],
                "customer_state":           ["SP", "RJ", "MG"],
            })

        # Sauvegarder dans le dossier temporaire
        df.to_csv(tmp_path / filename, index=False)

    return str(tmp_path)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestLoadOlist:
    """Tests pour la fonction load_olist()."""

    def test_returns_dict(self, tmp_data_dir):
        """load_olist() doit retourner un dictionnaire."""
        result = load_olist(tmp_data_dir)
        assert isinstance(result, dict)

    def test_returns_all_tables(self, tmp_data_dir):
        """load_olist() doit retourner les 5 tables attendues."""
        result = load_olist(tmp_data_dir)
        expected_keys = {"orders", "order_items", "order_payments",
                         "order_reviews", "customers"}
        assert set(result.keys()) == expected_keys

    def test_returns_dataframes(self, tmp_data_dir):
        """Chaque valeur du dictionnaire doit être un DataFrame."""
        result = load_olist(tmp_data_dir)
        for name, df in result.items():
            assert isinstance(df, pd.DataFrame), \
                f"Table '{name}' n'est pas un DataFrame"

    def test_dataframes_not_empty(self, tmp_data_dir):
        """Chaque DataFrame doit avoir au moins une ligne."""
        result = load_olist(tmp_data_dir)
        for name, df in result.items():
            assert len(df) > 0, f"Table '{name}' est vide"

    def test_orders_has_required_columns(self, tmp_data_dir):
        """La table orders doit avoir les colonnes essentielles."""
        result = load_olist(tmp_data_dir)
        required_cols = [
            "order_id", "customer_id", "order_status",
            "order_purchase_timestamp"
        ]
        for col in required_cols:
            assert col in result["orders"].columns, \
                f"Colonne '{col}' manquante dans orders"

    def test_raises_if_file_missing(self, tmp_path):
        """load_olist() doit lever FileNotFoundError si un fichier manque."""
        # Dossier vide — aucun CSV présent
        with pytest.raises(FileNotFoundError):
            load_olist(str(tmp_path))

    def test_table_names_simplified(self, tmp_data_dir):
        """Les clés doivent être simplifiées (sans 'olist_' ni '_dataset')."""
        result = load_olist(tmp_data_dir)
        for key in result.keys():
            assert not key.startswith("olist_"), \
                f"Clé '{key}' non simplifiée"
            assert not key.endswith("_dataset"), \
                f"Clé '{key}' non simplifiée"
    
class TestGetMergedDataset:
    """Tests pour la fonction get_merged_dataset()."""

    def test_returns_dataframe(self, tmp_data_dir):
        """get_merged_dataset() doit retourner un DataFrame."""
        from src.data.load_data import get_merged_dataset
        result = get_merged_dataset(tmp_data_dir)
        assert isinstance(result, pd.DataFrame)

    def test_has_order_and_customer_columns(self, tmp_data_dir):
        """Le DataFrame mergé doit avoir des colonnes des deux tables."""
        from src.data.load_data import get_merged_dataset
        result = get_merged_dataset(tmp_data_dir)
        assert "order_id" in result.columns
        assert "customer_id" in result.columns

    def test_no_duplicate_order_ids(self, tmp_data_dir):
        """Pas de doublons sur order_id après merge."""
        from src.data.load_data import get_merged_dataset
        result = get_merged_dataset(tmp_data_dir)
        assert result["order_id"].nunique() == len(result)