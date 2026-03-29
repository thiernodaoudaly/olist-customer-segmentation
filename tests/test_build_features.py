"""
Tests unitaires pour src/features/build_features.py

On teste chaque fonction de feature engineering indépendamment.
"""

import pytest
import pandas as pd

from src.features.build_features import (
    compute_recency,
    compute_frequency,
    compute_monetary,
    compute_satisfaction,
    compute_delivery,
    build_features,
    SNAPSHOT_DATE,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_orders():
    """Commandes minimales pour les tests."""
    return pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4"],
            "customer_id": ["c1", "c2", "c3", "c1"],  # c1 a 2 commandes
            "order_status": ["delivered", "delivered", "delivered", "delivered"],
            "order_purchase_timestamp": [
                "2018-01-01",
                "2018-03-01",
                "2018-06-01",
                "2018-07-01",
            ],
            "order_delivered_customer_date": [
                "2018-01-10",
                "2018-03-10",
                "2018-06-10",
                "2018-07-10",
            ],
            "order_estimated_delivery_date": [
                "2018-01-20",
                "2018-03-05",
                "2018-06-25",
                "2018-07-20",
            ],
        }
    )


@pytest.fixture
def sample_customers():
    """Clients minimaux pour les tests."""
    return pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c3"],
            "customer_unique_id": ["uc1", "uc2", "uc3"],
            "customer_state": ["SP", "RJ", "MG"],
        }
    )


@pytest.fixture
def sample_items():
    """Articles de commande minimaux pour les tests."""
    return pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4"],
            "price": [100.0, 200.0, 150.0, 300.0],
            "freight_value": [10.0, 20.0, 15.0, 30.0],
        }
    )


@pytest.fixture
def sample_reviews():
    """Avis minimaux pour les tests."""
    return pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4"],
            "review_score": [5, 4, 1, 5],
        }
    )


@pytest.fixture
def sample_dfs(sample_orders, sample_customers, sample_items, sample_reviews):
    """Dictionnaire complet simulant load_olist()."""
    return {
        "orders": sample_orders,
        "customers": sample_customers,
        "order_items": sample_items,
        "order_reviews": sample_reviews,
    }


# ── Tests compute_recency ─────────────────────────────────────────────────────


class TestComputeRecency:

    def test_returns_dataframe(self, sample_orders, sample_customers):
        result = compute_recency(sample_orders, sample_customers)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, sample_orders, sample_customers):
        result = compute_recency(sample_orders, sample_customers)
        assert "customer_unique_id" in result.columns
        assert "recency" in result.columns

    def test_recency_is_positive(self, sample_orders, sample_customers):
        """La récence doit toujours être positive (snapshot > dernière commande)."""
        result = compute_recency(sample_orders, sample_customers)
        assert (result["recency"] >= 0).all()

    def test_one_row_per_customer(self, sample_orders, sample_customers):
        """Un client avec 2 commandes doit avoir 1 seule ligne."""
        result = compute_recency(sample_orders, sample_customers)
        # c1 a 2 commandes (o1 et o4) mais doit apparaître une seule fois
        assert result["customer_unique_id"].nunique() == len(result)

    def test_recency_uses_latest_order(self, sample_orders, sample_customers):
        """La récence doit être calculée depuis la DERNIÈRE commande."""
        result = compute_recency(sample_orders, sample_customers)
        # uc1 (c1) a commandé le 2018-01-01 et 2018-07-01
        # Sa récence doit être basée sur 2018-07-01 (la plus récente)
        uc1_recency = result[result["customer_unique_id"] == "uc1"]["recency"].values[0]
        expected = (SNAPSHOT_DATE - pd.Timestamp("2018-07-01")).days
        assert uc1_recency == expected


# ── Tests compute_frequency ───────────────────────────────────────────────────


class TestComputeFrequency:

    def test_returns_dataframe(self, sample_orders, sample_customers):
        result = compute_frequency(sample_orders, sample_customers)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, sample_orders, sample_customers):
        result = compute_frequency(sample_orders, sample_customers)
        assert "customer_unique_id" in result.columns
        assert "frequency" in result.columns

    def test_frequency_minimum_one(self, sample_orders, sample_customers):
        """Tout client a au moins 1 commande."""
        result = compute_frequency(sample_orders, sample_customers)
        assert (result["frequency"] >= 1).all()

    def test_multi_order_customer(self, sample_orders, sample_customers):
        """Un client avec 2 commandes doit avoir frequency=2."""
        result = compute_frequency(sample_orders, sample_customers)
        uc1_freq = result[result["customer_unique_id"] == "uc1"]["frequency"].values[0]
        assert uc1_freq == 2


# ── Tests compute_monetary ────────────────────────────────────────────────────


class TestComputeMonetary:

    def test_returns_dataframe(self, sample_orders, sample_customers, sample_items):
        result = compute_monetary(sample_orders, sample_customers, sample_items)
        assert isinstance(result, pd.DataFrame)

    def test_monetary_is_positive(self, sample_orders, sample_customers, sample_items):
        """Le montant total doit être positif."""
        result = compute_monetary(sample_orders, sample_customers, sample_items)
        assert (result["monetary"] > 0).all()

    def test_monetary_includes_freight(
        self, sample_orders, sample_customers, sample_items
    ):
        """Le montant doit inclure les frais de port."""
        result = compute_monetary(sample_orders, sample_customers, sample_items)
        # uc1 a 2 commandes : o1 (100+10) + o4 (300+30) = 440
        uc1_monetary = result[result["customer_unique_id"] == "uc1"]["monetary"].values[
            0
        ]
        assert uc1_monetary == pytest.approx(440.0)


# ── Tests compute_satisfaction ────────────────────────────────────────────────


class TestComputeSatisfaction:

    def test_returns_dataframe(self, sample_orders, sample_customers, sample_reviews):
        result = compute_satisfaction(sample_orders, sample_customers, sample_reviews)
        assert isinstance(result, pd.DataFrame)

    def test_score_between_1_and_5(
        self, sample_orders, sample_customers, sample_reviews
    ):
        """Le score moyen doit être entre 1 et 5."""
        result = compute_satisfaction(sample_orders, sample_customers, sample_reviews)
        assert (result["review_score_mean"] >= 1).all()
        assert (result["review_score_mean"] <= 5).all()


# ── Tests compute_delivery ────────────────────────────────────────────────────


class TestComputeDelivery:

    def test_returns_dataframe(self, sample_orders, sample_customers):
        result = compute_delivery(sample_orders, sample_customers)
        assert isinstance(result, pd.DataFrame)

    def test_delay_is_clipped(self, sample_orders, sample_customers):
        """Le délai doit être clippé entre -30 et +30."""
        result = compute_delivery(
            sample_orders, sample_customers, clip_min=-30, clip_max=30
        )
        assert (result["delivery_delay_mean"] >= -30).all()
        assert (result["delivery_delay_mean"] <= 30).all()


# ── Tests build_features ──────────────────────────────────────────────────────


class TestBuildFeatures:

    def test_returns_dataframe(self, sample_dfs):
        result = build_features(sample_dfs)
        assert isinstance(result, pd.DataFrame)

    def test_has_all_feature_columns(self, sample_dfs):
        """Le DataFrame final doit avoir toutes les features."""
        result = build_features(sample_dfs)
        expected_cols = [
            "recency",
            "frequency",
            "monetary",
            "review_score_mean",
            "delivery_delay_mean",
            "frequency_log",
            "monetary_log",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Colonne '{col}' manquante"

    def test_no_missing_values(self, sample_dfs):
        """Aucune valeur manquante après build_features()."""
        result = build_features(sample_dfs)
        assert result.isnull().sum().sum() == 0

    def test_log_features_positive(self, sample_dfs):
        """Les features log-transformées doivent être positives."""
        result = build_features(sample_dfs)
        assert (result["frequency_log"] >= 0).all()
        assert (result["monetary_log"] >= 0).all()

    def test_one_row_per_customer(self, sample_dfs):
        """Une ligne par client unique."""
        result = build_features(sample_dfs)
        assert result["customer_unique_id"].nunique() == len(result)
