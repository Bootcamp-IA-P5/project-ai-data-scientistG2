"""
Aplicación Web Streamlit para predicción de riesgo de ictus.
Hospital F5 - Herramienta de Criba Previa con Inteligencia Artificial.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Añadir el directorio raíz al path para importar módulos
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.predict_model import (
    load_model_artifacts,
    make_prediction,
    validate_input_data,
    get_feature_contributions,
)


# ===========================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ===========================================
st.set_page_config(
    page_title="Riesgo de Ictus | Hospital F5",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===========================================
# 2. FUNCIONES AUXILIARES
# ===========================================


@st.cache_resource
def load_model():
    """Carga el modelo (con caché para eficiencia)."""
    model_path = Path(__file__).parent.parent / "artifacts" / "best_model_balanced.pkl"
    try:
        artifacts = load_model_artifacts(str(model_path))
        return artifacts
    except FileNotFoundError:
        return None


def display_prediction_result(
    prediction, probability, risk_level, patient_data, contributions
):
    """
    Muestra el resultado de la predicción de forma visual.

    Args:
        prediction: Predicción (0 o 1)
        probability: Probabilidad de ictus
        risk_level: Nivel de riesgo (BAJO, MODERADO, ALTO)
        patient_data: Datos del paciente
        contributions: DataFrame con características importantes
    """
    st.markdown("---")
    st.subheader("🎯 Resultado de la Evaluación")

    # Determinar el mensaje según el nivel de riesgo
    if prediction == 1:
        st.error("🔴 **ALERTA: Alto Riesgo de Ictus**")
    else:
        st.success("🟢 **Bajo Riesgo de Ictus**")

    # Mostrar métricas principales
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Predicción", value="ICTUS" if prediction == 1 else "SIN ICTUS")

    with col2:
        st.metric(label="Probabilidad", value=f"{probability:.1%}")

    with col3:
        risk_color = {"ALTO": "🔴", "MODERADO": "🟠", "BAJO": "🟢"}
        st.metric(
            label="Nivel de Riesgo",
            value=f"{risk_color.get(risk_level, '')} {risk_level}",
        )

    # Barra de progreso de probabilidad
    st.markdown("### 📊 Probabilidad de Ictus")
    st.progress(float(probability))

    # Interpretación del resultado
    st.markdown("### 💡 Interpretación")

    if risk_level == "ALTO":
        st.warning(
            """
        **Recomendaciones:**
        - ⚠️ El paciente presenta múltiples factores de riesgo
        - 🏥 Se recomienda **evaluación médica inmediata**
        - 🔬 Considerar estudios complementarios (neuroimagen, doppler carotídeo)
        - 💊 Revisar medicación y tratamiento preventivo
        """
        )
    elif risk_level == "MODERADO":
        st.info(
            """
        **Recomendaciones:**
        - 👨‍⚕️ El paciente presenta algunos factores de riesgo
        - 📅 Se recomienda **seguimiento médico regular**
        - 🥗 Implementar medidas preventivas (dieta, ejercicio)
        - 💊 Controlar factores de riesgo modificables
        """
        )
    else:
        st.success(
            """
        **Recomendaciones:**
        - ✅ El paciente presenta bajo riesgo de ictus
        - 🏃 Mantener hábitos de vida saludables
        - 📅 Realizar chequeos médicos periódicos
        - 🥗 Continuar con estilo de vida saludable
        """
        )

    # Factores más influyentes
    st.markdown("### 📈 Top 5 Factores Más Influyentes")

    if contributions is not None and not contributions.empty:
        # Crear un gráfico de barras horizontal
        st.bar_chart(contributions.set_index("feature")["importance"], height=300)

        # Mostrar tabla detallada
        with st.expander("Ver detalles de los factores"):
            st.dataframe(
                contributions[["feature", "importance", "value"]],
                use_container_width=True,
                hide_index=True,
            )

    # Resumen del paciente
    with st.expander("👤 Ver Resumen del Paciente"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Datos Demográficos:**")
            st.write(f"- Edad: {patient_data['age'].iloc[0]:.0f} años")
            st.write(f"- Género: {patient_data['gender'].iloc[0]}")
            st.write(f"- Estado civil: {patient_data['ever_married'].iloc[0]}")
            st.write(f"- Tipo de trabajo: {patient_data['work_type'].iloc[0]}")
            st.write(f"- Residencia: {patient_data['Residence_type'].iloc[0]}")

        with col2:
            st.markdown("**Datos de Salud:**")
            st.write(f"- IMC: {patient_data['bmi'].iloc[0]:.1f}")
            st.write(
                f"- Glucosa: {patient_data['avg_glucose_level'].iloc[0]:.1f} mg/dL"
            )
            st.write(
                f"- Hipertensión: {'Sí' if patient_data['hypertension'].iloc[0] == 1 else 'No'}"
            )
            st.write(
                f"- Enfermedad Cardíaca: {'Sí' if patient_data['heart_disease'].iloc[0] == 1 else 'No'}"
            )
            st.write(f"- Estado fumador: {patient_data['smoking_status'].iloc[0]}")

    # Disclaimer
    st.markdown("---")
    st.info(
        """
    ⚠️ **DESCARGO DE RESPONSABILIDAD:**
    
    Esta es una herramienta de **CRIBA PREVIA** basada en Inteligencia Artificial.
    **NO reemplaza** el diagnóstico médico profesional.
    
    Consulte **SIEMPRE** a un profesional de la salud para una evaluación completa.
    """
    )


# ===========================================
# 3. APLICACIÓN PRINCIPAL
# ===========================================


def main_app():
    """Define la estructura de la aplicación Streamlit."""

    # Título y descripción
    st.title("🧠 Evaluación de Riesgo de Ictus (Stroke)")
    st.markdown(
        """
    **Hospital F5** - Herramienta de Criba Previa con Inteligencia Artificial
    
    Esta aplicación utiliza Machine Learning para evaluar el riesgo de sufrir un ictus
    basándose en datos clínicos del paciente.
    """
    )

    # Cargar modelo
    artifacts = load_model()

    if artifacts is None:
        st.error(
            """
        ❌ **Error: Modelo no encontrado**
        
        Por favor, ejecuta primero el notebook `Modeling.ipynb` para entrenar
        y guardar el modelo antes de usar esta aplicación.
        """
        )
        st.stop()

    # Mostrar información del modelo
    with st.expander("ℹ️ Información del Modelo"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Modelo", "XGBoost")
        with col2:
            st.metric("Threshold", f"{artifacts['optimal_threshold']:.2f}")
        with col3:
            # Verificar si tiene test_metrics o solo metrics
            if "test_metrics" in artifacts and "auc_roc" in artifacts["test_metrics"]:
                st.metric("AUC-ROC", f"{artifacts['test_metrics']['auc_roc']:.3f}")
            elif "strategy" in artifacts:
                # Modelo balanceado
                st.metric("Estrategia", artifacts["strategy"].split("(")[0][:15])
            else:
                st.metric("F1-Score", f"{artifacts.get('metrics', 0.0):.3f}")

    st.markdown("---")

    # ===========================================
    # 4. FORMULARIO DE ENTRADA DE DATOS
    # ===========================================

    st.sidebar.header("📋 Datos del Paciente")
    st.sidebar.markdown("Ingrese los datos del paciente para evaluar su riesgo.")

    # DATOS DEMOGRÁFICOS
    st.sidebar.subheader("👤 Datos Demográficos")

    gender = st.sidebar.selectbox(
        "Género", options=["Male", "Female", "Other"], help="Género del paciente"
    )

    age = st.sidebar.slider(
        "Edad",
        min_value=0.0,
        max_value=100.0,
        value=45.0,
        step=0.1,
        help="Edad del paciente en años",
    )

    ever_married = st.sidebar.radio(
        "¿Alguna vez casado?", options=["Yes", "No"], help="Estado civil del paciente"
    )

    work_type = st.sidebar.selectbox(
        "Tipo de Trabajo",
        options=["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
        help="Ocupación del paciente",
    )

    residence_type = st.sidebar.selectbox(
        "Tipo de Residencia",
        options=["Urban", "Rural"],
        help="Tipo de área donde vive el paciente",
    )

    # DATOS DE SALUD
    st.sidebar.subheader("🏥 Datos de Salud")

    hypertension = st.sidebar.radio(
        "Hipertensión",
        options=[0, 1],
        format_func=lambda x: "Sí" if x == 1 else "No",
        help="¿El paciente tiene hipertensión?",
    )

    heart_disease = st.sidebar.radio(
        "Enfermedad Cardíaca",
        options=[0, 1],
        format_func=lambda x: "Sí" if x == 1 else "No",
        help="¿El paciente tiene alguna enfermedad cardíaca?",
    )

    avg_glucose_level = st.sidebar.number_input(
        "Nivel Promedio de Glucosa (mg/dL)",
        min_value=50.0,
        max_value=350.0,
        value=95.0,
        step=0.1,
        help="Nivel promedio de glucosa en sangre",
    )

    bmi = st.sidebar.number_input(
        "Índice de Masa Corporal (IMC)",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1,
        help="IMC = peso (kg) / altura² (m²)",
    )

    # ESTILO DE VIDA
    st.sidebar.subheader("🚬 Estilo de Vida")

    smoking_status = st.sidebar.selectbox(
        "Estado de Fumador",
        options=["never smoked", "formerly smoked", "smokes", "Unknown"],
        help="Historial de tabaquismo del paciente",
    )

    # ===========================================
    # 5. BOTÓN DE PREDICCIÓN
    # ===========================================

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🔍 Evaluar Riesgo de Ictus", type="primary", use_container_width=True
        )

    # ===========================================
    # 6. REALIZAR PREDICCIÓN
    # ===========================================

    if predict_button:
        # Recolectar datos en un diccionario
        input_data = {
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "ever_married": ever_married,
            "work_type": work_type,
            "Residence_type": residence_type,
            "avg_glucose_level": avg_glucose_level,
            "bmi": bmi,
            "smoking_status": smoking_status,
        }

        # Convertir a DataFrame
        input_df = pd.DataFrame([input_data])

        # Validar datos
        with st.spinner("🔍 Validando datos..."):
            is_valid, error_msg = validate_input_data(input_df)

            if not is_valid:
                st.error(f"❌ Error en validación: {error_msg}")
                st.stop()

        # Realizar predicción
        with st.spinner("🤖 Realizando predicción..."):
            try:
                result = make_prediction(artifacts, input_df)
                prediction = result["prediction"]
                probability = result["probability"]
                risk_level = result["risk_level"]

                contributions = get_feature_contributions(artifacts, input_df, top_n=5)

                # Mostrar resultado
                display_prediction_result(
                    prediction, probability, risk_level, input_df, contributions
                )

            except Exception as e:
                st.error(f"❌ Error al realizar la predicción: {e}")
                st.exception(e)

    # ===========================================
    # 7. INFORMACIÓN ADICIONAL
    # ===========================================

    st.markdown("---")

    # Pestañas con información adicional
    tab1, tab2, tab3 = st.tabs(
        ["📊 Sobre el Modelo", "❓ Preguntas Frecuentes", "👨‍⚕️ Contacto"]
    )

    with tab1:
        st.markdown(
            """
        ### Sobre el Modelo de Predicción
        
        Este modelo fue entrenado usando **XGBoost** con las siguientes características:
        
        - **Algoritmo**: XGBoost optimizado con Optuna
        - **Dataset**: 5,110 registros de pacientes
        - **Estrategia de balanceo**: {strategy}
        - **Métricas de rendimiento**:
          - Accuracy: {accuracy:.1%}
          - Precision: {precision:.1%}
          - Recall: {recall:.1%}
          - F1-Score: {f1:.1%}
          - G-Mean: {gmean:.3f}
        
        El modelo ha sido optimizado para **maximizar la detección de casos positivos**,
        minimizando los falsos negativos (casos de ictus no detectados).
        """.format(
                strategy=artifacts.get("strategy", "Balanceo avanzado"),
                accuracy=artifacts["test_metrics"]["accuracy"],
                precision=artifacts["test_metrics"]["precision"],
                recall=artifacts["test_metrics"]["recall"],
                f1=artifacts["test_metrics"]["f1"],
                gmean=artifacts["test_metrics"]["g_mean"],
            )
        )

    with tab2:
        st.markdown(
            """
        ### Preguntas Frecuentes
        
        **¿Qué es un ictus?**
        
        Un ictus (o accidente cerebrovascular) ocurre cuando el flujo sanguíneo al cerebro
        se interrumpe, causando daño a las células cerebrales.
        
        **¿Qué factores de riesgo considera el modelo?**
        
        El modelo considera múltiples factores:
        - Edad
        - Nivel de glucosa en sangre
        - Índice de Masa Corporal (IMC)
        - Hipertensión
        - Enfermedades cardíacas
        - Historial de tabaquismo
        
        **¿Qué significa cada nivel de riesgo?**
        
        - 🟢 **BAJO**: Probabilidad < 40%
        - 🟠 **MODERADO**: Probabilidad 40-70%
        - 🔴 **ALTO**: Probabilidad > 70%
        
        **¿Es 100% preciso?**
        
        No. Este es un modelo de Machine Learning con aproximadamente {:.1%} de precisión.
        Es una **herramienta de apoyo**, no un diagnóstico definitivo.
        """.format(
                artifacts["test_metrics"]["accuracy"]
            )
        )

    with tab3:
        st.markdown(
            """
        ### Contacto
        
        **Hospital F5 - Departamento de Innovación**
        
        - 📧 Email: innovacion@hospitalf5.es
        - 📞 Teléfono: +34 900 123 456
        - 🌐 Web: www.hospitalf5.es
        
        Para consultas médicas, contacte con su médico de cabecera
        o acuda al servicio de urgencias.
        """
        )


# ===========================================
# 8. EJECUTAR APLICACIÓN
# ===========================================

if __name__ == "__main__":
    main_app()
