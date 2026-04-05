"""
Tests unitaires pour src/models/predict.py
"""
import numpy as np
import pytest
from unittest.mock import MagicMock

from src.models.predict import (
    SEGMENT_NAMES,
    SEGMENT_STRATEGIES,
    FEATURES,
    predict_segment,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_scaler():
    """
    Scaler factice qui retourne les valeurs telles quelles.
    On teste la logique, pas le scaling réel.
    """
    scaler = MagicMock()
    scaler.transform.side_effect = lambda x: x
    return scaler


@pytest.fixture
def mock_kmeans_cluster0():
    """K-Means factice qui retourne toujours le cluster 0."""
    model = MagicMock()
    model.predict.return_value = np.array([0])
    return model


@pytest.fixture
def mock_kmeans_cluster3():
    """K-Means factice qui retourne toujours le cluster 3 (VIP)."""
    model = MagicMock()
    model.predict.return_value = np.array([3])
    return model


@pytest.fixture
def sample_client():
    """Client type — Nouveau satisfait."""
    return {
        "recency":             135,
        "frequency":           1,
        "monetary":            147.0,
        "review_score_mean":   4.5,
        "delivery_delay_mean": -12.0,
    }


@pytest.fixture
def vip_client():
    """Client type — VIP multi-acheteurs."""
    return {
        "recency":             222,
        "frequency":           3,
        "monetary":            309.0,
        "review_score_mean":   4.2,
        "delivery_delay_mean": -12.0,
    }


# ── Tests constants ───────────────────────────────────────────────────────────


class TestConstants:

    def test_segment_names_has_4_entries(self):
        """SEGMENT_NAMES doit avoir exactement 4 segments."""
        assert len(SEGMENT_NAMES) == 4

    def test_segment_strategies_has_4_entries(self):
        """SEGMENT_STRATEGIES doit avoir exactement 4 stratégies."""
        assert len(SEGMENT_STRATEGIES) == 4

    def test_segment_ids_are_0_to_3(self):
        """Les IDs de segments doivent être 0, 1, 2, 3."""
        assert set(SEGMENT_NAMES.keys()) == {0, 1, 2, 3}
        assert set(SEGMENT_STRATEGIES.keys()) == {0, 1, 2, 3}

    def test_features_list_not_empty(self):
        """La liste des features ne doit pas être vide."""
        assert len(FEATURES) > 0

    def test_features_contains_required(self):
        """Les features essentielles doivent être présentes."""
        required = {"recency", "monetary_log", "review_score_mean"}
        assert required.issubset(set(FEATURES))


# ── Tests predict_segment ─────────────────────────────────────────────────────


class TestPredictSegment:

    def test_returns_dict(
        self, sample_client, mock_kmeans_cluster0, mock_scaler
    ):
        """predict_segment() doit retourner un dictionnaire."""
        result = predict_segment(
            sample_client, mock_kmeans_cluster0, mock_scaler
        )
        assert isinstance(result, dict)

    def test_has_required_keys(
        self, sample_client, mock_kmeans_cluster0, mock_scaler
    ):
        """Le résultat doit avoir les clés segment_id, segment_name, strategy."""
        result = predict_segment(
            sample_client, mock_kmeans_cluster0, mock_scaler
        )
        assert "segment_id"   in result
        assert "segment_name" in result
        assert "strategy"     in result

    def test_segment_id_is_int(
        self, sample_client, mock_kmeans_cluster0, mock_scaler
    ):
        """segment_id doit être un entier."""
        result = predict_segment(
            sample_client, mock_kmeans_cluster0, mock_scaler
        )
        assert isinstance(result["segment_id"], int)

    def test_segment_id_in_valid_range(
        self, sample_client, mock_kmeans_cluster0, mock_scaler
    ):
        """segment_id doit être entre 0 et 3."""
        result = predict_segment(
            sample_client, mock_kmeans_cluster0, mock_scaler
        )
        assert 0 <= result["segment_id"] <= 3

    def test_cluster0_returns_correct_name(
        self, sample_client, mock_kmeans_cluster0, mock_scaler
    ):
        """Cluster 0 doit retourner 'Nouveaux satisfaits'."""
        result = predict_segment(
            sample_client, mock_kmeans_cluster0, mock_scaler
        )
        assert result["segment_id"]   == 0
        assert result["segment_name"] == SEGMENT_NAMES[0]
        assert result["strategy"]     == SEGMENT_STRATEGIES[0]

    def test_cluster3_returns_vip(
        self, vip_client, mock_kmeans_cluster3, mock_scaler
    ):
        """Cluster 3 doit retourner 'VIP multi-acheteurs'."""
        result = predict_segment(
            vip_client, mock_kmeans_cluster3, mock_scaler
        )
        assert result["segment_id"]   == 3
        assert result["segment_name"] == "VIP multi-acheteurs"

    def test_log1p_applied_to_frequency(
        self, sample_client, mock_scaler
    ):
        """
        Vérifie que log1p est appliqué sur frequency et monetary
        avant le scaling.
        """
        captured_input = {}

        def capture_transform(x):
            captured_input["X"] = x.copy()
            return x

        mock_scaler.transform.side_effect = capture_transform

        model = MagicMock()
        model.predict.return_value = np.array([0])

        predict_segment(sample_client, model, mock_scaler)

        # Vérifier que frequency_log = log1p(1) ≈ 0.693
        freq_idx = FEATURES.index("frequency_log")
        expected_freq_log = np.log1p(sample_client["frequency"])
        assert abs(
            captured_input["X"][0][freq_idx] - expected_freq_log
        ) < 1e-6

    def test_scaler_transform_called_once(
        self, sample_client, mock_kmeans_cluster0, mock_scaler
    ):
        """Le scaler doit être appelé exactement une fois."""
        predict_segment(sample_client, mock_kmeans_cluster0, mock_scaler)
        mock_scaler.transform.assert_called_once()

    def test_model_predict_called_once(
        self, sample_client, mock_kmeans_cluster0, mock_scaler
    ):
        """Le modèle doit être appelé exactement une fois."""
        predict_segment(sample_client, mock_kmeans_cluster0, mock_scaler)
        mock_kmeans_cluster0.predict.assert_called_once()

    def test_all_segments_reachable(self, sample_client, mock_scaler):
        """Chaque segment (0-3) doit être atteignable."""
        for seg_id in range(4):
            model = MagicMock()
            model.predict.return_value = np.array([seg_id])
            result = predict_segment(sample_client, model, mock_scaler)
            assert result["segment_id"] == seg_id
            assert result["segment_name"] == SEGMENT_NAMES[seg_id]