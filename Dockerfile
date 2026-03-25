# ── Image de base ──────────────────────────────────────────────
# python:3.9-slim = Python 3.9 sur Debian minimal (~50Mo vs ~900Mo pour python:3.9)
FROM python:3.9-slim

# ── Métadonnées ────────────────────────────────────────────────
LABEL maintainer="Thierno Daouda LY"
LABEL description="Olist customer segmentation API"

# ── Variables d'environnement ──────────────────────────────────
# Empêche Python de créer des fichiers .pyc dans le container
ENV PYTHONDONTWRITEBYTECODE=1
# Force les logs Python à apparaître immédiatement (pas de buffer)
ENV PYTHONUNBUFFERED=1
# Ajoute src/ au PYTHONPATH pour que les imports fonctionnent
ENV PYTHONPATH=/app

# ── Répertoire de travail ──────────────────────────────────────
WORKDIR /app

# ── Dépendances système ────────────────────────────────────────
# Nécessaires pour compiler certains packages Python (umap-learn, scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*   # nettoie le cache apt → réduit la taille de l'image

# ── Dépendances Python ─────────────────────────────────────────
# IMPORTANT : on copie requirements.txt SEUL avant le reste du code
# → Docker cache cette couche tant que requirements.txt ne change pas
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Code source ────────────────────────────────────────────────
# Copié APRÈS les dépendances → le cache des dépendances est préservé
# si on modifie uniquement le code
COPY . .

# ── Installation du package src/ ──────────────────────────────
RUN pip install --no-cache-dir -e .

# ── Port exposé ────────────────────────────────────────────────
EXPOSE 8000

# ── Commande de démarrage ──────────────────────────────────────
# Lance l'API FastAPI via uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]