# Aplicación Streamlit/API

import streamlit as st
import pandas as pd
# Importarías las funciones del modelo más tarde, por ahora se quedan comentadas:
# from src.models.predict_model import load_model, make_prediction 

# 1. --- Configuración y Título de la Página ---
st.set_page_config(
    page_title="Riesgo de Ictus | Hospital F5",
    layout="wide",
    initial_sidebar_state="auto"
)

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


def main_app():
    """Define la estructura de la aplicación Streamlit."""
    
    st.title("🧠 Evaluación de Riesgo de Ictus (Stroke)")
    st.markdown("Hospital F5 - Herramienta de Criba Previa con Inteligencia Artificial.")
    
    st.markdown("---")
    
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
    ever_married = st.sidebar.radio("¿Alguna vez Casado?", ('Yes', 'No'))

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
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }
    
    # Convertir a DataFrame (para que el MOCK o el modelo lo procese)
    input_df = pd.DataFrame([input_data])
    
    # Botón de acción
    st.markdown("---")
    if st.button("Evaluar Riesgo de Ictus", type="primary"):
        # 4. --- Lógica de Predicción (Actual o MOCK) ---
        
        with st.spinner('Evaluando datos...'):
            # ESTE CÓDIGO SE REEMPLAZARÁ LUEGO CON EL MODELO REAL
            # model = load_model(ARTIFACTS_PATH / MODEL_FILENAME)
            # prediction, probability = make_prediction(model, input_df)
            # display_real_prediction(prediction, probability)
            
            # Por ahora, usamos la función de MOCK
            display_mock_prediction(input_df.iloc[0])


if __name__ == "__main__":
    main_app()