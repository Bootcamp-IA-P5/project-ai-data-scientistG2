"""
Módulo de predicción para el modelo de detección de ictus.
Contiene funciones para cargar el modelo, preprocesar datos y realizar predicciones.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path


def preprocess_input(patient_data, scaler, feature_names):
    """
    Preprocesa los datos del paciente aplicando el MISMO pipeline que en entrenamiento.

    Args:
        patient_data: DataFrame con datos crudos del paciente
        scaler: StandardScaler usado en entrenamiento
        feature_names: Lista de nombres de características del modelo

    Returns:
        DataFrame preprocesado listo para predicción
    """
    df = patient_data.copy()

    # ===========================================
    # 1. FEATURE ENGINEERING (igual que Preprocessing.ipynb)
    # ===========================================

    # 1.1 Rangos de edad
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 18, 35, 50, 65, 100],
        labels=["0-18", "19-35", "36-50", "51-65", "65+"],
    )

    # 1.2 Factores de riesgo combinados
    df["risk_factors"] = df["hypertension"] + df["heart_disease"]

    # 1.3 Categorías de BMI
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    )

    # 1.4 Categorías de glucosa
    df["glucose_category"] = pd.cut(
        df["avg_glucose_level"],
        bins=[0, 100, 125, 300],
        labels=["Normal", "Prediabetes", "Diabetes"],
    )

    # 1.5 Interacción edad-riesgo
    df["age_risk_interaction"] = df["age"] * df["risk_factors"]

    # ===========================================
    # 2. CODIFICACIÓN DE VARIABLES CATEGÓRICAS
    # ===========================================

    # 2.1 Label Encoding para variables binarias
    binary_mappings = {
        "gender": {"Male": 1, "Female": 0, "Other": 2},
        "ever_married": {"Yes": 1, "No": 0},
        "Residence_type": {"Urban": 1, "Rural": 0},
    }

    for col, mapping in binary_mappings.items():
        df[f"{col}_encoded"] = df[col].map(mapping)

    # 2.2 One-Hot Encoding para variables con múltiples categorías
    categorical_cols = [
        "work_type",
        "smoking_status",
        "age_group",
        "bmi_category",
        "glucose_category",
    ]

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

    # ===========================================
    # 3. ELIMINAR COLUMNAS ORIGINALES
    # ===========================================
    cols_to_drop = ["gender", "ever_married", "Residence_type"]
    df = df.drop(
        columns=[col for col in cols_to_drop if col in df.columns], errors="ignore"
    )

    # ===========================================
    # 4. ASEGURAR QUE TENGA TODAS LAS CARACTERÍSTICAS DEL MODELO
    # ===========================================
    # Agregar columnas faltantes con valor 0
    for feat in feature_names:
        if feat not in df.columns:
            df[feat] = 0

    # Reordenar para que coincida con el entrenamiento
    df = df[feature_names]

    # ===========================================
    # 5. ESCALAR VARIABLES NUMÉRICAS
    # ===========================================
    numeric_features = ["age", "avg_glucose_level", "bmi", "age_risk_interaction"]
    df[numeric_features] = scaler.transform(df[numeric_features])

    return df


def make_prediction(model_artifacts, patient_data):
    """
    Realiza la predicción para un paciente.

    Args:
        model_artifacts: Diccionario con modelo, scaler, feature_names, etc.
        patient_data: DataFrame con datos del paciente

    Returns:
        dict con prediction, probability, risk_level, confidence
    """
    model = model_artifacts["model"]
    scaler = model_artifacts["scaler"]
    feature_names = model_artifacts["feature_names"]
    threshold = model_artifacts.get("optimal_threshold", 0.5)

    # Preprocesar
    X = preprocess_input(patient_data, scaler, feature_names)

    # Predecir probabilidad
    probability = model.predict_proba(X)[:, 1][0]

    # Aplicar threshold personalizado
    prediction = int(probability >= threshold)

    # Interpretar riesgo
    if probability >= 0.7:
        risk_level = "ALTO"
        confidence = "ALTA"
    elif probability >= 0.5:
        risk_level = "MODERADO-ALTO"
        confidence = "MODERADA-ALTA"
    elif probability >= 0.3:
        risk_level = "MODERADO"
        confidence = "MODERADA"
    else:
        risk_level = "BAJO"
        confidence = "ALTA"

    return {
        "prediction": prediction,
        "probability": float(probability),
        "risk_level": risk_level,
        "confidence": confidence,
        "threshold_used": threshold,
    }


def get_feature_contributions(model_artifacts, patient_data, top_n=5):
    """
    Calcula las características que más influyen en la predicción.

    Args:
        model_artifacts: Diccionario con modelo y artefactos
        patient_data: DataFrame con datos del paciente
        top_n: Número de características principales a retornar

    Returns:
        DataFrame con las características más influyentes
    """
    model = model_artifacts["model"]
    scaler = model_artifacts["scaler"]
    feature_names = model_artifacts["feature_names"]

    # Preprocesar
    X = preprocess_input(patient_data, scaler, feature_names)

    # Obtener importancia de características del modelo
    feature_importance = model.feature_importances_

    # Calcular contribución (importancia * valor)
    contributions = feature_importance * X.values[0]

    # Crear DataFrame
    contrib_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": feature_importance,
            "value": X.values[0],
            "contribution": contributions,
        }
    ).sort_values("contribution", ascending=False, key=abs)

    return contrib_df.head(top_n)


def load_model_artifacts(model_path=None):
    """
    Carga el modelo y artefactos necesarios.

    Args:
        model_path: Ruta al archivo del modelo. Si es None, busca en la ubicación por defecto.

    Returns:
        dict: Diccionario con modelo, scaler, feature_names, etc.
    """
    if model_path is None:
        # Buscar desde la raíz del proyecto (independiente de dónde se ejecute)
        current_file = Path(__file__).resolve()  # Ubicación de este archivo
        project_root = (
            current_file.parent.parent.parent
        )  # Subir 3 niveles (models -> src -> raíz)
        model_path = project_root / "artifacts" / "best_model_balanced.pkl"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"❌ No se encontró el modelo en: {model_path}\n"
            f"\n"
            f"🔧 Soluciones:\n"
            f"  1. Ejecuta el notebook 'notebooks/Modeling.ipynb' para entrenar el modelo\n"
            f"  2. Asegúrate de que el archivo 'artifacts/best_model_balanced.pkl' existe\n"
            f"  3. Verifica que estás ejecutando desde la raíz del proyecto\n"
        )

    with open(model_path, "rb") as f:
        artifacts = pickle.load(f)

    # Si no tiene scaler, cargarlo desde preprocessed_data.pkl
    if "scaler" not in artifacts:
        project_root = Path(__file__).resolve().parent.parent.parent
        preprocessed_path = project_root / "data" / "preprocessed_data.pkl"

        if preprocessed_path.exists():
            with open(preprocessed_path, "rb") as f:
                preprocessed_data = pickle.load(f)
                artifacts["scaler"] = preprocessed_data["scaler"]
                # También cargar feature_names si no los tiene
                if "feature_names" not in artifacts:
                    artifacts["feature_names"] = preprocessed_data["feature_names"]
        else:
            raise FileNotFoundError(
                f"❌ El modelo no contiene 'scaler' y no se encontró 'preprocessed_data.pkl'\n"
                f"Ejecuta el notebook 'notebooks/Preprocessing.ipynb' primero."
            )

    print(f"✅ Modelo cargado desde: {model_path}")
    return artifacts


def validate_input_data(patient_data):
    """
    Valida que los datos del paciente sean correctos.

    Args:
        patient_data: DataFrame con datos del paciente

    Returns:
        tuple (is_valid, error_messages)
    """
    errors = []

    # Validar campos requeridos
    required_fields = [
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "avg_glucose_level",
        "bmi",
        "smoking_status",
    ]

    for field in required_fields:
        if field not in patient_data.columns:
            errors.append(f"Campo faltante: {field}")

    if len(errors) > 0:
        return False, errors

    # Validar rangos
    if not (0 < patient_data["age"].iloc[0] <= 120):
        errors.append("Edad debe estar entre 1 y 120 años")

    if not (0 < patient_data["avg_glucose_level"].iloc[0] <= 300):
        errors.append("Nivel de glucosa debe estar entre 1 y 300 mg/dL")

    if not (10 < patient_data["bmi"].iloc[0] <= 100):
        errors.append("BMI debe estar entre 10 y 100")

    # Validar categorías
    valid_categories = {
        "gender": ["Male", "Female", "Other"],
        "hypertension": [0, 1],
        "heart_disease": [0, 1],
        "ever_married": ["Yes", "No"],
        "work_type": [
            "Private",
            "Self-employed",
            "Govt_job",
            "children",
            "Never_worked",
        ],
        "Residence_type": ["Urban", "Rural"],
        "smoking_status": ["formerly smoked", "never smoked", "smokes", "Unknown"],
    }

    for field, valid_values in valid_categories.items():
        if patient_data[field].iloc[0] not in valid_values:
            errors.append(f"{field} debe ser uno de: {valid_values}")

    return len(errors) == 0, errors


# ===========================================
# EJEMPLO DE USO
# ===========================================
if __name__ == "__main__":
    # Ejemplo de uso del módulo
    print("=" * 60)
    print("🧪 EJEMPLO DE USO - MÓDULO DE PREDICCIÓN")
    print("=" * 60)

    # 1. Cargar modelo
    try:
        artifacts = load_model_artifacts()  # Ahora busca automáticamente
        print(f"\n✅ Modelo cargado correctamente")
        print(f"   Threshold óptimo: {artifacts['optimal_threshold']:.2f}")
        print(f"   Métricas de test:")
        print(f"     - Accuracy: {artifacts['test_metrics']['accuracy']:.4f}")
        print(f"     - Recall: {artifacts['test_metrics']['recall']:.4f}")
        print(f"     - F1-Score: {artifacts['test_metrics']['f1']:.4f}")
    except FileNotFoundError as e:
        print(f"\n{e}")
        exit(1)

    # 2. Datos de ejemplo de un paciente
    patient_example = pd.DataFrame(
        [
            {
                "gender": "Male",
                "age": 67.0,
                "hypertension": 0,
                "heart_disease": 1,
                "ever_married": "Yes",
                "work_type": "Private",
                "Residence_type": "Urban",
                "avg_glucose_level": 228.69,
                "bmi": 36.6,
                "smoking_status": "formerly smoked",
            }
        ]
    )

    print("\n📋 Datos del paciente:")
    for key, value in patient_example.iloc[0].items():
        print(f"   {key}: {value}")

    # 3. Validar datos
    is_valid, error_msg = validate_input_data(patient_example)
    if not is_valid:
        print(f"\n❌ Error en validación: {error_msg}")
        exit(1)

    print("\n✅ Datos validados correctamente")

    # 4. Realizar predicción
    result = make_prediction(artifacts, patient_example)

    print("\n" + "=" * 60)
    print("🎯 RESULTADO DE LA PREDICCIÓN")
    print("=" * 60)
    print(
        f"\n{'🔴' if result['prediction'] == 1 else '🟢'} Predicción: {'ICTUS' if result['prediction'] == 1 else 'SIN ICTUS'}"
    )
    print(f"   Probabilidad: {result['probability']:.2%}")
    print(f"   Nivel de riesgo: {result['risk_level']}")
    print(f"   Confianza: {result['confidence']}")
    print(f"   Threshold usado: {result['threshold_used']:.2f}")

    # 5. Características más importantes
    contributions = get_feature_contributions(artifacts, patient_example, top_n=5)
    print(f"\n📊 Top 5 características más importantes:")
    for idx, row in contributions.iterrows():
        print(
            f"   {row['feature']}: {row['value']:.4f} (importancia: {row['importance']:.4f})"
        )

    print("\n" + "=" * 60)
    print("⚠️ DISCLAIMER: Este es un resultado de CRIBA PREVIA.")
    print("   Consulte siempre a un profesional de la salud.")
    print("=" * 60)
