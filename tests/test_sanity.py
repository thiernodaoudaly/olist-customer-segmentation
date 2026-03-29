"""Test de sanité — vérifie que l'environnement de test fonctionne."""


def test_python_works():
    """1 + 1 = 2."""
    assert 1 + 1 == 2


def test_imports_work():
    """Les packages essentiels sont importables."""
    import pandas as pd
    import numpy as np
    import sklearn
    assert True


def test_src_importable():
    """Le package src/ est bien installé."""
    from src.data.load_data import load_olist
    from src.features.build_features import build_features
    assert callable(load_olist)
    assert callable(build_features)