import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
sys.path.append(str(NOTEBOOKS))

from notebooks.train_model_tuning import ThresholdOptimizerEnsemble as Optimizer


@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        "f1": np.random.randn(200),
        "f2": np.random.randn(200),
        "f3": np.random.randn(200)
    })
    y = (X["f1"] + X["f2"] * 0.3 + np.random.randn(200) * 0.5 > 0).astype(int)
    return X, y


def test_optimizer_initialization(sample_data):
    X, y = sample_data
    opt = Optimizer(models_dir="models_test")
    assert opt is not None


def test_get_model(sample_data):
    """
    Este test reemplaza el antiguo best_model() para pruebas.
    """
    X, y = sample_data
    opt = Optimizer(models_dir="models_test")

    # Construir y entrenar un modelo base
    model = opt.get_model("lr", X, y)
    assert hasattr(model, "predict_proba"), "El modelo debe tener predict_proba()"

    model = opt.get_model("lgbm", X, y)
    assert hasattr(model, "predict_proba"), "El modelo debe tener predict_proba()"


def test_threshold_search(sample_data):
    X, y = sample_data
    opt = Optimizer(models_dir="models_test")

    # Usamos get_model en lugar de best_model
    model = opt.get_model("lr", X, y)

    best_t, df, strategy = opt.find_best_threshold_cv(
        model, X, y, thresholds=np.linspace(0.1, 0.9, 5), cv_splits=3
    )

    assert 0 <= best_t <= 1, "El threshold debe estar entre 0 y 1"
    assert len(df) > 0, "Debe devolver un dataframe con resultados"


def test_full_pipeline(sample_data):
    X, y = sample_data
    X_train, X_test = X.iloc[:150], X.iloc[150:]
    y_train, y_test = y.iloc[:150], y.iloc[150:]

    opt = Optimizer(models_dir="models_test")

    results = opt.run_pipeline(
        X_train, y_train, X_test, y_test,
        calibrate=False,
        cv_splits=3,
        min_recall=0.5
    )

    assert isinstance(results, dict), "Los resultados deben venir en un diccionario"
    assert opt.best_model_name in results, "Debe devolver el modelo ganador"
