# Aplicación Streamlit/API

import streamlit as st
import pandas as pd
import pickle
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os 

# 1. --- Configuración y Título de la Página ---
st.set_page_config(
    page_title="Riesgo de Ictus | Hospital F5",
    layout="wide",
    initial_sidebar_state="auto"
)

@st.cache_resource
def load_model_and_metadata():
    """Cargar el modelo MLP, scaler y metadatos guardados"""
    try:
        # Cargar modelo desde pickle
        model_path = "data/mlp_model.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Cargar scaler
        scaler_path = "data/scaler.pkl"
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Cargar metadatos
        metadata_path = "data/modelo_info.pkl"
        metadata = joblib.load(metadata_path)
        
        st.success("✅ Modelo MLP y scaler cargados correctamente")
        return model, scaler, metadata
    except Exception as e:
        st.error(f"❌ Error cargando el modelo: {str(e)}")
        return None, None, None

def preprocess_input(data):
    """Preprocesar datos de entrada para el modelo"""
    df = data.copy()
    
    # Mapear variables categóricas (como se hizo en el entrenamiento)
    gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
    ever_married_map = {'No': 0, 'Yes': 1}
    work_type_map = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4}
    residence_map = {'Rural': 0, 'Urban': 1}
    smoking_map = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3}
    
    # Aplicar transformaciones
    df['gender'] = df['gender'].map(gender_map)
    df['ever_married'] = df['ever_married'].map(ever_married_map)
    df['work_type'] = df['work_type'].map(work_type_map)
    df['Residence_type'] = df['Residence_type'].map(residence_map)
    df['smoking_status'] = df['smoking_status'].map(smoking_map)
    
    return df

def make_prediction(model, scaler, data, metadata):
    """Hacer predicción con el modelo MLP"""
    try:
        # Preprocesar datos
        processed_data = preprocess_input(data)
        
        # Escalar datos usando el scaler entrenado
        X_scaled = scaler.transform(processed_data)
        
        # Hacer predicción
        probability = model.predict(X_scaled)[0][0]
        
        # Usar los umbrales del entrenamiento
        best_threshold_f1 = metadata.get('best_threshold_f1', 0.5)
        best_threshold_precision = metadata.get('best_threshold_precision', 0.5)
        
        # Clasificación binaria usando umbral F1
        prediction = 1 if probability >= best_threshold_f1 else 0
        
        return prediction, probability, best_threshold_f1
    
    except Exception as e:
        st.error(f"Error en predicción: {str(e)}")
        return None, None, None

def display_mock_prediction(data):
    """
    Función de MOCK para simular la predicción sin cargar el modelo.
    Se reemplazará con la lógica real del modelo más tarde.
    """
    st.subheader("Resultado de la Evaluación (MOCK)")
    
    # Simulación simple basada en edad como ejemplo
    if data['age'] > 60:
        st.error("🔴 ALERTA: Alto Riesgo de Ictus (Simulación)")
        st.markdown("**Probabilidad Estimada:** `75.45%`")
        st.write("Motivo simulado: La edad avanzada aumenta significativamente el riesgo.")
    elif data['avg_glucose_level'] > 180:
        st.warning("🟠 Riesgo Moderado de Ictus (Simulación)")
        st.markdown("**Probabilidad Estimada:** `40.12%`")
        st.write("Motivo simulado: Nivel de glucosa elevado.")
    else:
        st.success("🟢 Bajo Riesgo de Ictus (Simulación)")
        st.markdown("**Probabilidad Estimada:** `8.99%`")

    st.info("⚠️ **Descargo de Responsabilidad:** Esta es una criba previa de IA. Consulte a un doctor para un diagnóstico definitivo.")

def display_real_prediction(prediction, probability, threshold, metadata):
    """Mostrar resultados de la predicción real"""
    st.subheader("🤖 Resultado de la Evaluación con IA")
    
    # Mostrar métricas del modelo
    if metadata:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ROC-AUC del Modelo", f"{metadata.get('roc_auc', 0):.3f}")
        with col2:
            st.metric("PR-AUC del Modelo", f"{metadata.get('pr_auc', 0):.3f}")
    
    # Resultado principal
    probability_percent = probability * 100
    
    if prediction == 1:
        st.error(f"� ALERTA: Alto Riesgo de Ictus")
        st.markdown(f"**Probabilidad:** `{probability_percent:.2f}%`")
        st.markdown(f"**Umbral usado:** `{threshold:.3f}`")
        
        # Recomendaciones
        st.markdown("### 🚨 Recomendaciones:")
        st.markdown("- Consulte a un médico especialista inmediatamente")
        st.markdown("- Controle regularmente presión arterial y glucosa")
        st.markdown("- Evite factores de riesgo (tabaco, sedentarismo)")
        
    else:
        st.success(f"🟢 Bajo Riesgo de Ictus")
        st.markdown(f"**Probabilidad:** `{probability_percent:.2f}%`")
        st.markdown(f"**Umbral usado:** `{threshold:.3f}`")
        
        # Recomendaciones preventivas
        st.markdown("### 💚 Recomendaciones Preventivas:")
        st.markdown("- Mantenga hábitos de vida saludables")
        st.markdown("- Controles médicos regulares")
        st.markdown("- Ejercicio regular y dieta equilibrada")
    
    # Barra de progreso visual
    st.markdown("### 📊 Análisis de Riesgo:")
    # Convertir probability a float nativo de Python para evitar error con float32
    probability_float = float(probability)
    st.progress(probability_float, text=f"Probabilidad de Riesgo: {probability_percent:.2f}%")
    
    st.info("⚠️ **Descargo:** Este es un sistema de apoyo diagnóstico. Consulte siempre a un profesional médico.")


def main_app():
    """Define la estructura de la aplicación Streamlit."""
    
    st.title("🧠 Evaluación de Riesgo de Ictus (Stroke)")
    st.markdown("Hospital F5 - Herramienta de Criba Previa con Inteligencia Artificial.")
    
    st.markdown("---")
    
    # Selector de tipo de predicción
    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Configuración")
    prediction_type = st.sidebar.radio(
        "Tipo de Predicción:",
        ("🤖 Modelo MLP Real", "🎭 Simulación MOCK"),
        help="Elige si usar el modelo MLP entrenado o una simulación básica"
    )
    
    # Cargar modelo solo si se selecciona el modelo real
    model, scaler, metadata = None, None, None
    if prediction_type == "🤖 Modelo MLP Real":
        model, scaler, metadata = load_model_and_metadata()
        
        if model is None or scaler is None:
            st.error("❌ No se pudo cargar el modelo o scaler. Usando simulación MOCK.")
            prediction_type = "🎭 Simulación MOCK"
    
    # 2. --- Widgets de Entrada de Datos ---
    
    # Uso de la barra lateral para un diseño más limpio
    st.sidebar.header("Datos del Paciente")
    
    # CARACTERÍSTICAS CATEGÓRICAS
    gender = st.sidebar.selectbox("Género", ('Female', 'Male', 'Other'))
    smoking_status = st.sidebar.selectbox("Estado de Fumador", 
                                          ('never smoked', 'formerly smoked', 'smokes', 'Unknown'))
    work_type = st.sidebar.selectbox("Tipo de Trabajo", 
                                     ('Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'))
    residence_type = st.sidebar.selectbox("Tipo de Residencia", ('Urban', 'Rural'))

    # CARACTERÍSTICAS BINARIAS
    hypertension = st.sidebar.radio("Hipertensión", (0, 1), format_func=lambda x: 'Sí' if x == 1 else 'No')
    heart_disease = st.sidebar.radio("Enfermedad Cardíaca", (0, 1), format_func=lambda x: 'Sí' if x == 1 else 'No')

    # CARACTERÍSTICAS NUMÉRICAS
    # Usamos un slider para edad y una entrada numérica para los demás
    age = st.sidebar.slider("Edad", min_value=0.0, max_value=100.0, value=45.0, step=0.1)
    
    avg_glucose_level = st.sidebar.number_input("Nivel Promedio de Glucosa (mg/dL)", 
                                                 min_value=50.0, max_value=350.0, value=95.0, step=0.1)
    
    bmi = st.sidebar.number_input("Índice de Masa Corporal (IMC)", 
                                  min_value=10.0, max_value=60.0, value=25.0, step=0.1)

    # 3. --- Almacenamiento y Botón de Predicción ---
    
    # Recolectar datos en un diccionario/DataFrame (listo para el modelo)
    input_data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': 'Yes',  # Valor fijo (campo removido de la interfaz por bajo impacto)
        'work_type': work_type,
        'Residence_type': residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }
    
    # Convertir a DataFrame (para que el modelo lo procese)
    input_df = pd.DataFrame([input_data])
    
    # Mostrar información del modelo solo si está cargado
    if prediction_type == "🤖 Modelo MLP Real" and metadata:
        with st.expander("ℹ️ Información del Modelo MLP"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("ROC-AUC", f"{metadata.get('roc_auc', 0):.3f}")
                st.metric("Umbral F1", f"{metadata.get('best_threshold_f1', 0):.3f}")
            with col2:
                st.metric("PR-AUC", f"{metadata.get('pr_auc', 0):.3f}")
                st.metric("Umbral Precisión", f"{metadata.get('best_threshold_precision', 0):.3f}")
    
    # Botón de acción
    st.markdown("---")
    button_text = "🤖 Evaluar con Modelo MLP" if prediction_type == "🤖 Modelo MLP Real" else "🎭 Evaluar con Simulación"
    if st.button(button_text, type="primary"):
        # 4. --- Lógica de Predicción ---
        
        if prediction_type == "🤖 Modelo MLP Real":
            with st.spinner('Evaluando datos con IA real...'):
                # Hacer predicción real con el modelo MLP
                prediction, probability, threshold = make_prediction(model, scaler, input_df, metadata)
                
                if prediction is not None:
                    display_real_prediction(prediction, probability, threshold, metadata)
                else:
                    st.error("❌ Error al procesar la predicción")
        else:
            with st.spinner('Evaluando datos con simulación...'):
                # Usar la función de MOCK
                display_mock_prediction(input_df.iloc[0])
    
    # Información adicional
    st.markdown("---")
    st.markdown("### 📖 Sobre este Sistema")
    
    if prediction_type == "🤖 Modelo MLP Real":
        st.markdown("""
        Este sistema utiliza un **Perceptrón Multicapa (MLP)** entrenado con técnicas de balanceado de clases 
        para evaluar el riesgo de ictus. El modelo ha sido entrenado con datos clínicos y utiliza múltiples 
        factores de riesgo para generar una predicción.
        
        **Características del modelo:**
        - Arquitectura: Red Neuronal con 128 y 64 neuronas ocultas
        - Regularización: L2 y Dropout para evitar sobreajuste
        - Balanceado: Class weights para manejar datos desequilibrados
        - Optimización: Early stopping basado en pérdida de validación
        """)
    else:
        st.markdown("""
        **Modo Simulación MOCK:** Este modo utiliza reglas simples basadas en edad y glucosa 
        para simular una predicción. Es útil para demostración y pruebas del sistema.
        
        **Reglas de simulación:**
        - Edad > 60 años → Alto riesgo (75%)
        - Glucosa > 180 mg/dL → Riesgo moderado (40%)
        - Otros casos → Bajo riesgo (9%)
        """)


if __name__ == "__main__":
    main_app()