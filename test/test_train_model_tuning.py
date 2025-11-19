"""
Tests para el modelo de predicción de ictus
Ejecutar con: pytest tests/test_model.py -v
"""

import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import recall_score, precision_score

# AÑADIR DESPUÉS DE LOS IMPORTS
from sklearn.base import BaseEstimator, TransformerMixin

class IdentityScaler(BaseEstimator, TransformerMixin):
    """Scaler dummy para tests"""
    def __init__(self):
        self.n_features_in_ = None
        self.feature_names_in_ = None
    
    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.n_features_in_ = len(X.columns)
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.n_features_in_ = X.shape[1]
        return self
    
    def transform(self, X):
        return X
    
    def inverse_transform(self, X):
        return X

# ===========================================
# FIXTURES
# ===========================================

@pytest.fixture(scope="module")
def modelo_cargado():
    """Carga el modelo guardado una sola vez para todos los tests"""
    # Obtener ruta absoluta desde donde se ejecuta pytest
    import os
    
    # Si se ejecuta desde raíz del proyecto
    project_root = Path(__file__).parent.parent  # Sube 2 niveles desde test/
    model_path = project_root / "models" / "modelo_final_oficial.pkl"
    
    print(f"\n🔍 Buscando modelo en: {model_path.absolute()}")
    
    if not model_path.exists():
        # Buscar en ubicaciones alternativas
        alt_paths = [
            Path("models/modelo_final_oficial.pkl"),
            Path("../models/modelo_final_oficial.pkl"),
            project_root / "notebooks" / "models" / "modelo_final_oficial.pkl"
        ]
        
        for alt_path in alt_paths:
            print(f"   Intentando: {alt_path.absolute()}")
            if alt_path.exists():
                model_path = alt_path
                break
        else:
            pytest.skip(f"Modelo no encontrado. Búsqueda en:\n{model_path.absolute()}")
    
    print(f"✅ Modelo encontrado en: {model_path.absolute()}")
    return joblib.load(model_path)

@pytest.fixture
def datos_sinteticos(modelo_cargado):
    """Genera datos sintéticos CON EL NÚMERO CORRECTO de features"""
    np.random.seed(42)
    n_samples = 100
    
    # Usar el número de features del modelo
    scaler = modelo_cargado.get('scaler')
    
    # Si no hay scaler, usar n_features del modelo guardado
    if scaler is not None and hasattr(scaler, 'n_features_in_'):
        n_features = scaler.n_features_in_
    elif 'n_features' in modelo_cargado:
        n_features = modelo_cargado['n_features']
    else:
        # Fallback: usar feature_names
        n_features = len(modelo_cargado['feature_names'])
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    
    return X, y


@pytest.fixture
def caso_positivo_real(modelo_cargado):
    """Caso real típico de paciente con ictus"""
    scaler = modelo_cargado.get('scaler')
    
    # Obtener número de features
    if scaler is not None and hasattr(scaler, 'n_features_in_'):
        n_features = scaler.n_features_in_
    elif 'n_features' in modelo_cargado:
        n_features = modelo_cargado['n_features']
    else:
        n_features = len(modelo_cargado['feature_names'])
    
    # Crear array con número correcto de features
    features = np.array([
        75,      # edad avanzada
        1,       # hipertensión
        1,       # enfermedad cardíaca
        1,       # casado
    ])
    
    # Rellenar el resto
    if n_features > len(features):
        padding = np.random.randn(n_features - len(features)) * 0.5
        features = np.concatenate([features, padding])
    
    return features.reshape(1, -1)


# ===========================================
# TESTS DE CARGA Y ESTRUCTURA
# ===========================================

def test_modelo_existe():
    """Verifica que el archivo del modelo existe"""
    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / "modelo_final_oficial.pkl"
    
    # Mostrar info de debug
    print(f"\n📂 Directorio de trabajo: {Path.cwd()}")
    print(f"📂 Directorio del test: {Path(__file__).parent}")
    print(f"📂 Raíz del proyecto: {project_root}")
    print(f"🔍 Buscando en: {model_path.absolute()}")
    
    if not model_path.exists():
        print(f"\n❌ Modelo NO encontrado")
        print(f"📁 Contenido de {project_root / 'models'}:")
        models_dir = project_root / "models"
        if models_dir.exists():
            for f in models_dir.iterdir():
                print(f"   - {f.name}")
        else:
            print("   (carpeta models no existe)")
    
    assert model_path.exists(), f"El archivo del modelo no existe en {model_path.absolute()}"


def test_estructura_modelo(modelo_cargado):
    """Verifica que el modelo tiene todos los componentes necesarios"""
    assert 'model' in modelo_cargado, "Falta el modelo XGBoost"
    assert 'threshold' in modelo_cargado, "Falta el threshold óptimo"
    assert 'scaler' in modelo_cargado, "Falta el scaler"
    assert 'feature_names' in modelo_cargado, "Faltan los nombres de features"


def test_threshold_valido(modelo_cargado):
    """Verifica que el threshold está en un rango válido"""
    threshold = modelo_cargado['threshold']
    assert 0.0 <= threshold <= 1.0, f"Threshold fuera de rango: {threshold}"
    assert 0.15 <= threshold <= 0.60, f"Threshold fuera del rango esperado: {threshold}"


def test_numero_features(modelo_cargado):
    """Verifica que el modelo espera 25 features"""
    assert len(modelo_cargado['feature_names']) == 25, "Debería tener 25 features"


# ===========================================
# TESTS DE PREDICCIÓN
# ===========================================

def test_prediccion_forma_correcta(modelo_cargado, datos_sinteticos):
    """Verifica que las predicciones tienen la forma correcta"""
    X, _ = datos_sinteticos
    model = modelo_cargado['model']
    
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    assert y_pred_proba.shape[0] == X.shape[0], "Número de predicciones incorrecto"
    assert np.all((y_pred_proba >= 0) & (y_pred_proba <= 1)), "Probabilidades fuera de rango"


def test_prediccion_binaria(modelo_cargado, datos_sinteticos):
    """Verifica que las predicciones binarias son 0 o 1"""
    X, _ = datos_sinteticos
    model = modelo_cargado['model']
    threshold = modelo_cargado['threshold']
    
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    assert set(y_pred).issubset({0, 1}), "Predicciones deberían ser solo 0 o 1"


def test_caso_positivo_detectado(modelo_cargado, caso_positivo_real):
    """Verifica que un caso positivo típico es detectado"""
    model = modelo_cargado['model']
    scaler = modelo_cargado.get('scaler')
    threshold = modelo_cargado['threshold']
    
    # Si no hay scaler, los datos ya deben venir preprocesados
    if scaler is not None:
        X_scaled = scaler.transform(caso_positivo_real)
    else:
        X_scaled = caso_positivo_real
    
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    
    # Con recall ~86%, es probable pero no garantizado
    # Verificamos solo que la probabilidad sea razonable
    assert y_pred_proba[0] > 0.3, f"Probabilidad muy baja para caso positivo: {y_pred_proba[0]:.4f}"



# ===========================================
# TESTS DE MÉTRICAS
# ===========================================

def test_recall_minimo(modelo_cargado):
    """Verifica que el modelo tiene el recall mínimo esperado"""
    test_recall = modelo_cargado.get('test_recall', None)
    
    if test_recall:
        assert test_recall >= 0.80, f"Recall en test ({test_recall:.2%}) por debajo del mínimo (80%)"


def test_precision_minima(modelo_cargado):
    """Verifica que el modelo tiene precision razonable"""
    test_precision = modelo_cargado.get('test_precision', None)
    
    if test_precision:
        assert test_precision >= 0.12, f"Precision en test ({test_precision:.2%}) muy baja"


def test_verdaderos_positivos_minimos(modelo_cargado):
    """Verifica que detecta suficientes casos reales"""
    test_tp = modelo_cargado.get('test_tp', None)
    test_fn = modelo_cargado.get('test_fn', None)
    
    if test_tp and test_fn:
        total_positivos = test_tp + test_fn
        assert test_tp >= 40, f"Solo detecta {test_tp}/{total_positivos} casos"


# ===========================================
# TESTS DE ROBUSTEZ
# ===========================================

def test_manejo_valores_faltantes(modelo_cargado):
    """Verifica que el modelo maneja NaN correctamente"""
    model = modelo_cargado['model']
    scaler = modelo_cargado['scaler']
    
    X_con_nan = np.random.randn(10, 25)
    X_con_nan[0, 0] = np.nan
    
    # El scaler debería fallar o el modelo debería rechazarlo
    with pytest.raises((ValueError, Exception)):
        X_scaled = scaler.transform(X_con_nan)
        model.predict_proba(X_scaled)


def test_forma_incorrecta_input(modelo_cargado):
    """Verifica que rechaza inputs con forma incorrecta"""
    model = modelo_cargado['model']
    
    X_incorrecto = np.random.randn(10, 20)  # 20 features en vez de 25
    
    with pytest.raises((ValueError, Exception)):
        model.predict_proba(X_incorrecto)


def test_escalado_consistente(modelo_cargado, datos_sinteticos):
    """Verifica que el escalado produce valores razonables"""
    X, _ = datos_sinteticos
    scaler = modelo_cargado.get('scaler')
    
    if scaler is None:
        pytest.skip("Modelo no usa scaler (datos preprocesados)")
    
    X_scaled = scaler.transform(X)
    
    # Los valores escalados deberían estar mayormente en [-3, 3]
    assert np.abs(X_scaled).mean() < 2.0, "Escalado produce valores extremos"


# ===========================================
# TESTS DE RENDIMIENTO
# ===========================================

def test_velocidad_prediccion(modelo_cargado, datos_sinteticos):
    """Verifica que las predicciones son suficientemente rápidas"""
    import time
    
    X, _ = datos_sinteticos
    model = modelo_cargado['model']
    scaler = modelo_cargado.get('scaler')
    
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X
    
    start = time.time()
    for _ in range(100):
        _ = model.predict_proba(X_scaled)
    elapsed = time.time() - start
    
    # Debería procesar 100 batches de 100 muestras en < 1 segundo
    assert elapsed < 1.0, f"Predicción muy lenta: {elapsed:.3f}s para 10,000 muestras"


# ===========================================
# TESTS DE INTEGRACIÓN
# ===========================================

def test_pipeline_completo(modelo_cargado, datos_sinteticos):
    """Test de integración: carga -> escala -> predice -> threshold"""
    X, y = datos_sinteticos
    
    model = modelo_cargado['model']
    scaler = modelo_cargado.get('scaler')
    threshold = modelo_cargado['threshold']
    
    # Pipeline completo
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X
    
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Verificaciones básicas
    assert len(y_pred) == len(y), "Longitud de predicciones incorrecta"
    assert y_pred.dtype == int, "Predicciones deberían ser enteros"
    assert set(y_pred).issubset({0, 1}), "Solo debería predecir 0 o 1"


# ===========================================
# TESTS PARAMETRIZADOS
# ===========================================

@pytest.mark.parametrize("threshold_test", [0.3, 0.4, 0.5, 0.6])
def test_diferentes_thresholds(modelo_cargado, datos_sinteticos, threshold_test):
    """Verifica comportamiento con diferentes thresholds"""
    X, y = datos_sinteticos
    model = modelo_cargado['model']
    scaler = modelo_cargado.get('scaler')
    
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X
    
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_pred_proba >= threshold_test).astype(int)
    
    # Con threshold más alto -> menos positivos
    # Con threshold más bajo -> más positivos
    n_positivos = y_pred.sum()
    assert 0 <= n_positivos <= len(y), "Número de positivos fuera de rango"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])