"""
Vérification des installations du requirements.txt.
Usage : python scripts/check_env.py
"""
import sys
import importlib


PACKAGES = {
    # (nom_import, nom_affichage)
    "numpy":      "numpy",
    "pandas":     "pandas",
    "scipy":      "scipy",
    "matplotlib": "matplotlib",
    "seaborn":    "seaborn",
    "plotly":     "plotly",
    "sklearn":    "scikit-learn",
    "mlflow":     "mlflow",
    "dvc":        "dvc",
    "fastapi":    "fastapi",
    "uvicorn":    "uvicorn",
    "pydantic":   "pydantic",
    "black":      "black",
    "flake8":     "flake8",
    "isort":      "isort",
    "pytest":     "pytest",
    "pytest_cov": "pytest-cov",
}


def check_packages() -> list[str]:
    """
    Tente d'importer chaque package et affiche son statut.

    Returns:
        Liste des packages manquants.
    """
    failures = []

    print(f"Python : {sys.version.split()[0]}\n")
    print(f"{'Package':<20} {'Statut':<8} {'Version'}")
    print("-" * 45)

    for import_name, display_name in PACKAGES.items():
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "?")
            print(f"{display_name:<20} {'OK':<8} {version}")
        except ImportError:
            print(f"{display_name:<20} {'MANQUANT':<8}")
            failures.append(display_name)

    return failures


def check_umap() -> None:
    """Vérifie umap-learn séparément car optionnel sur Python 3.13."""
    print("\numap-learn :")
    print("-" * 45)
    try:
        import umap
        version = getattr(umap, "__version__", "?")
        print(f"{'umap-learn':<20} {'OK':<8} {version}")
    except ImportError:
        print(f"{'umap-learn':<20} {'ABSENT':<8} fallback -> PCA")


def main() -> None:
    print("=" * 45)
    print("  Verification environnement MLOps Olist")
    print("=" * 45 + "\n")

    failures = check_packages()
    check_umap()

    print("\n" + "=" * 45)
    if not failures:
        print("  Environnement OK — tous les packages sont installes")
    else:
        print("  Packages manquants :")
        for pkg in failures:
            print(f"    -> pip install {pkg}")
        print("\n  Ou relance : pip install -r requirements.txt")
    print("=" * 45)


if __name__ == "__main__":
    main()