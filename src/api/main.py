"""
API FastAPI pour la segmentation clients Olist.

Endpoints :
    GET  /              → health check
    GET  /segments      → liste des segments disponibles
    POST /predict       → prédit le segment d'un client

Usage :
    uvicorn src.api.main:app --reload --port 8000
    Documentation : http://localhost:8000/docs
"""
import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.predict import (
    SEGMENT_NAMES,
    SEGMENT_STRATEGIES,
    predict_segment,
)


# ── Modèles Pydantic ──────────────────────────────────────────────────────────


class ClientFeatures(BaseModel):
    """Features brutes d'un client pour la prédiction."""

    recency: float = Field(
        ...,
        ge=0,
        description="Nb de jours depuis le dernier achat",
        example=135,
    )
    frequency: float = Field(
        ...,
        ge=1,
        description="Nb de commandes total",
        example=1,
    )
    monetary: float = Field(
        ...,
        gt=0,
        description="Montant total dépensé en BRL",
        example=147.0,
    )
    review_score_mean: float = Field(
        ...,
        ge=1,
        le=5,
        description="Score moyen des avis (1-5)",
        example=4.5,
    )
    delivery_delay_mean: float = Field(
        ...,
        ge=-30,
        le=30,
        description="Délai moyen livraison en jours (négatif=avance)",
        example=-12.0,
    )


class PredictionResponse(BaseModel):
    """Réponse de l'endpoint /predict."""

    segment_id: int
    segment_name: str
    strategy: str
    input_features: dict


class SegmentInfo(BaseModel):
    """Informations sur un segment."""

    segment_id: int
    segment_name: str
    strategy: str


# ── Chargement du modèle au démarrage ─────────────────────────────────────────

# Variables globales pour stocker le modèle et le scaler
_kmeans_model = None
_scaler = None


def load_models():
    """
    Charge le modèle K-Means et le scaler depuis MLflow.

    Cherche le run MLflow nommé 'kmeans_final_k4' dans
    l'expérience 'olist-segmentation'.
    """
    global _kmeans_model, _scaler

    # URI MLflow — local par défaut
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    try:
        # Chercher le run kmeans_final_k4
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("olist-segmentation")

        if experiment is None:
            raise ValueError("Expérience 'olist-segmentation' introuvable")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.mlflow.runName = 'kmeans_final_k4'",
            max_results=1,
        )

        if not runs:
            raise ValueError("Run 'kmeans_final_k4' introuvable")

        run_id = runs[0].info.run_id

        # Charger le modèle et le scaler
        _kmeans_model = mlflow.sklearn.load_model(
            f"runs:/{run_id}/kmeans_model"
        )
        _scaler = mlflow.sklearn.load_model(
            f"runs:/{run_id}/scaler"
        )

        print(f"Modèle chargé depuis run : {run_id}")
        print(f"K-Means clusters : {_kmeans_model.n_clusters}")

    except Exception as e:
        print(f"Erreur chargement modèle : {e}")
        print("API démarrée sans modèle — /predict indisponible")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage de l'API."""
    load_models()
    yield


# ── Application FastAPI ───────────────────────────────────────────────────────

app = FastAPI(
    title="Olist Customer Segmentation API",
    description=(
        "API de segmentation clients Olist.\n\n"
        "Prédit le segment marketing d'un client "
        "à partir de ses données comportementales (RFM enrichi)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/", tags=["Health"])
def health_check():
    """Vérifie que l'API est opérationnelle."""
    return {
        "status": "ok",
        "model_loaded": _kmeans_model is not None,
        "scaler_loaded": _scaler is not None,
        "version": "1.0.0",
    }


@app.get("/segments", response_model=list[SegmentInfo], tags=["Segments"])
def get_segments():
    """Retourne la liste de tous les segments disponibles."""
    return [
        SegmentInfo(
            segment_id=sid,
            segment_name=name,
            strategy=SEGMENT_STRATEGIES[sid],
        )
        for sid, name in SEGMENT_NAMES.items()
    ]


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: ClientFeatures):
    """
    Prédit le segment marketing d'un client.

    Fournit les features comportementales du client et récupère
    son segment avec la stratégie marketing associée.
    """
    # Vérifier que le modèle est chargé
    if _kmeans_model is None or _scaler is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Modèle non disponible. "
                "Vérifiez que MLflow contient le run 'kmeans_final_k4'."
            ),
        )

    # Prédire le segment
    client_dict = features.model_dump()
    result = predict_segment(client_dict, _kmeans_model, _scaler)

    return PredictionResponse(
        segment_id=result["segment_id"],
        segment_name=result["segment_name"],
        strategy=result["strategy"],
        input_features=client_dict,
    )