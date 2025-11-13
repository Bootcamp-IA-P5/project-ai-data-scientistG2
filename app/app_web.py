# Aplicación Streamlit/API

import streamlit as st
import pandas as pd
import pickle
import os 

# 1. --- Configuración y Título de la Página ---
st.set_page_config(
    page_title="Riesgo de Ictus | Hospital F5",
    layout="wide",
    initial_sidebar_state="auto"
)

@st.cache_resource
def load_model_and_metadata():
    """Cargar el modelo con configuración Ultra optimizada"""
    try:
        config_path = "data/final_optimal_recall80_config.pkl"
        
        # Verificar si existe la configuración Ultra optimizada
        if os.path.exists(config_path):
            # Cargar configuración optimizada
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
            st.success("✅ Configuración Ultra optimizada cargada")
            
            # Determinar qué modelo usar según el ranking de la configuración
            best_model = config['all_models_ranking'][0]  # El mejor modelo del ranking
            model_name = best_model['model']
            
            # Mapear nombres de modelos a archivos disponibles
            if model_name == "Modelo Ultra":
                # El modelo Ultra usa la misma base que el original pero con parámetros optimizados
                model_path = "data/mlp_model.pkl"  # Usar modelo original
                scaler_path = "data/scaler.pkl" 
            elif model_name == "Modelo Original":
                model_path = "data/mlp_model.pkl"
                scaler_path = "data/scaler.pkl"
            elif model_name == "Modelo Precision v1":
                model_path = "data/mlp_model_precision.pkl"
                scaler_path = "data/scaler_precision.pkl"
            else:
                # Fallback al modelo original
                model_path = "data/mlp_model.pkl"
                scaler_path = "data/scaler.pkl"
            
            # Cargar modelo
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                st.success(f"✅ {model_name} cargado correctamente")
            except Exception as e:
                st.error(f"Error cargando modelo: {e}")
                return None, None, None
            
            # Cargar scaler
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            except Exception as e:
                st.error(f"Error cargando scaler: {e}")
                return None, None, None
            
            # Crear metadatos optimizados usando la configuración Ultra
            metadata = {
                'model_type': 'ultra_optimized',
                'model_name': model_name,
                'architecture': best_model['description'],
                'optimal_threshold': config['optimal_threshold'],
                'precision': config['metrics']['precision'],
                'recall': config['metrics']['recall'],
                'f1_score': config['metrics']['f1_score'],
                'specificity': config['metrics']['specificity'],
                'accuracy': config['metrics']['accuracy'],
                'npv': config['metrics']['npv'],
                'description': 'Configuración Ultra optimizada - máxima precisión con Recall ≥80%',
                'clinical_interpretation': config['clinical_interpretation']
            }
            
            st.info(f"🎯 {model_name} con configuración Ultra - Precisión: {metadata['precision']*100:.1f}%, Recall: {metadata['recall']*100:.1f}%")
            return model, scaler, metadata
            
        else:
            # Fallback a modelo original si no existe configuración optimizada
            st.warning("⚠️ Configuración optimizada no encontrada, usando modelo original")
            
            model_path = "data/mlp_model.pkl"
            scaler_path = "data/scaler.pkl" 
            metadata_path = "data/modelo_info.pkl"
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            # Cargar metadatos originales si existen
            try:
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
            except Exception:
                # Crear metadatos básicos si no existen
                metadata = {
                    'model_type': 'original',
                    'optimal_threshold': 0.5,
                    'description': 'Modelo original sin optimización'
                }
            
            st.success("✅ Modelo original cargado correctamente")
            return model, scaler, metadata
            
    except Exception as e:
        st.error(f"❌ Error cargando el modelo: {str(e)}")
        st.error("💡 Asegúrate de que los archivos del modelo estén en la carpeta 'data/'")
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
    """Hacer predicción con el modelo MLP optimizado"""
    try:
        # Preprocesar datos
        processed_data = preprocess_input(data)
        
        # Escalar datos usando el scaler entrenado
        X_scaled = scaler.transform(processed_data)
        
        # Hacer predicción
        if hasattr(model, 'predict'):
            # Modelo de TensorFlow/Keras
            prediction_raw = model.predict(X_scaled, verbose=0)
            probability = float(prediction_raw[0][0])
        else:
            # Modelo de sklearn
            probability = float(model.predict_proba(X_scaled)[0][1])
        
        # Usar el umbral optimizado de la configuración Ultra
        optimal_threshold = metadata.get('optimal_threshold', 0.5)  # Fallback a 0.5 si no está configurado
        
        # Clasificación binaria usando umbral optimizado
        prediction = 1 if probability >= optimal_threshold else 0
        
        return prediction, probability, optimal_threshold
    
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
    """Mostrar resultados con modelo Ultra optimizado"""
    model_name = metadata.get('model_name', 'Modelo Optimizado') if metadata else 'Modelo'
    st.subheader(f"🤖 Evaluación con IA - {model_name}")
    
    # Mostrar métricas del modelo optimizado
    if metadata and metadata.get('model_type') == 'ultra_optimized':
        st.markdown("#### 📊 Rendimiento del Modelo Ultra Optimizado:")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            precision = metadata.get('precision', 0) * 100
            st.metric("🎯 Precisión", f"{precision:.1f}%", 
                     help="De cada 100 alertas positivas, aproximadamente 14 son casos reales")
        
        with col2:
            recall = metadata.get('recall', 0) * 100
            st.metric("🔍 Detección (Recall)", f"{recall:.1f}%", 
                     help="Detecta 80 de cada 100 casos reales de ictus")
        
        with col3:
            specificity = metadata.get('specificity', 0) * 100
            st.metric("✅ Especificidad", f"{specificity:.1f}%", 
                     help="Identifica correctamente 73 de cada 100 personas sanas")
        
        with col4:
            accuracy = metadata.get('accuracy', 0) * 100
            st.metric("⚖️ Exactitud", f"{accuracy:.1f}%", 
                     help="Precisión general del modelo en todos los casos")
        
        # Mostrar información clínica adicional
        clinical = metadata.get('clinical_interpretation', {})
        if clinical:
            st.markdown("#### 🏥 Interpretación Clínica:")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                detection_rate = clinical.get('detection_rate', 0) * 100
                st.metric("🎯 Tasa de Detección", f"{detection_rate:.0f}%",
                         help="Porcentaje de casos de ictus detectados")
            
            with col2:
                false_alarm_rate = clinical.get('false_alarm_rate', 0) * 100
                st.metric("⚠️ Tasa de Falsa Alarma", f"{false_alarm_rate:.1f}%",
                         help="Porcentaje de falsas alarmas en personas sanas")
            
            with col3:
                patients_to_evaluate = clinical.get('patients_to_evaluate', 0)
                st.metric("👥 Pacientes a Evaluar", f"{patients_to_evaluate}",
                         help="De cada 1000 pacientes, cuántos necesitan evaluación adicional")
    
    elif metadata:
        st.markdown("#### 📊 Rendimiento del Modelo:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            precision = metadata.get('precision', 0) * 100 if 'precision' in metadata else 'N/A'
            st.metric("🎯 Precisión", f"{precision:.1f}%" if isinstance(precision, (int, float)) else precision)
        
        with col2:
            recall = metadata.get('recall', 0) * 100 if 'recall' in metadata else 'N/A'
            st.metric("🔍 Recall", f"{recall:.1f}%" if isinstance(recall, (int, float)) else recall)
        
        with col3:
            accuracy = metadata.get('accuracy', 0) * 100 if 'accuracy' in metadata else 'N/A'
            st.metric("⚖️ Exactitud", f"{accuracy:.1f}%" if isinstance(accuracy, (int, float)) else accuracy)
    
    # Resultado principal
    probability_percent = probability * 100
    
    if prediction == 1:
        st.error("🚨 ALERTA: Alto Riesgo de Ictus")
        st.markdown(f"**Probabilidad:** `{probability_percent:.2f}%`")
        st.markdown(f"**Umbral usado:** `{threshold:.4f}` (optimizado)")
        
        # Recomendaciones
        st.markdown("### 🚨 Recomendaciones:")
        st.markdown("- Consulte a un médico especialista inmediatamente")
        st.markdown("- Controle regularmente presión arterial y glucosa")
        st.markdown("- Evite factores de riesgo (tabaco, sedentarismo)")
        
    else:
        st.success("🟢 Bajo Riesgo de Ictus")
        st.markdown(f"**Probabilidad:** `{probability_percent:.2f}%`")
        st.markdown(f"**Umbral usado:** `{threshold:.4f}` (optimizado)")
        
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
    
    # Mostrar información del modelo optimizado si está cargado
    if prediction_type == "🤖 Modelo MLP Real" and metadata:
        model_name = metadata.get('model_name', 'Modelo Optimizado')
        with st.expander(f"ℹ️ Información del {model_name}"):
            st.markdown(f"**🏗️ Arquitectura:** {metadata.get('architecture', 'MLP')}")
            st.markdown(f"**🎯 Descripción:** {metadata.get('description', 'Modelo optimizado')}")
            
            if metadata.get('model_type') == 'ultra_optimized':
                col1, col2, col3 = st.columns(3)
                with col1:
                    precision = metadata.get('precision', 0) * 100
                    st.metric("🎯 Precisión", f"{precision:.1f}%")
                    recall = metadata.get('recall', 0) * 100
                    st.metric("🔍 Recall", f"{recall:.1f}%")
                with col2:
                    f1 = metadata.get('f1_score', 0)
                    st.metric("⚖️ F1-Score", f"{f1:.3f}")
                    threshold = metadata.get('optimal_threshold', 0.5)
                    st.metric("🎯 Umbral Óptimo", f"{threshold:.4f}")
                with col3:
                    specificity = metadata.get('specificity', 0) * 100
                    st.metric("✅ Especificidad", f"{specificity:.1f}%")
                    accuracy = metadata.get('accuracy', 0) * 100
                    st.metric("📊 Exactitud", f"{accuracy:.1f}%")
                
                # Mostrar información clínica
                clinical = metadata.get('clinical_interpretation', {})
                if clinical:
                    st.markdown("**🏥 Interpretación Clínica:**")
                    workload_ratio = clinical.get('workload_ratio', 0)
                    cases_missed = clinical.get('cases_missed', 0)
                    st.markdown(f"• Cada paciente positivo requiere evaluar {workload_ratio:.1f} pacientes adicionales")
                    st.markdown(f"• Se estima que {cases_missed} casos podrían no detectarse de cada 50")
            
            else:
                # Mostrar información básica para modelos no ultra-optimizados
                col1, col2 = st.columns(2)
                with col1:
                    threshold = metadata.get('optimal_threshold', 0.5)
                    st.metric("🎯 Umbral", f"{threshold:.4f}")
                with col2:
                    if 'accuracy' in metadata:
                        accuracy = metadata.get('accuracy', 0) * 100
                        st.metric("📊 Exactitud", f"{accuracy:.1f}%")
    
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
        if metadata and metadata.get('model_type') == 'ultra_optimized':
            st.markdown(f"""
            Este sistema utiliza el **{metadata.get('model_name', 'Modelo Ultra')}** basado en Perceptrón Multicapa (MLP) 
            específicamente configurado para **máxima precisión manteniendo 80% de detección** de casos de ictus.
            
            **🏆 Características del Modelo Ultra:**
            - **Arquitectura:** {metadata.get('architecture', 'Red Neuronal optimizada')}
            - **Optimización:** Umbral {metadata.get('optimal_threshold', 0.276):.4f} (búsqueda exhaustiva sobre 1,980 configuraciones)
            - **Rendimiento:** {metadata.get('precision', 0)*100:.1f}% precisión / {metadata.get('recall', 0)*100:.1f}% recall / {metadata.get('accuracy', 0)*100:.1f}% exactitud
            - **Regularización:** L2 + Dropout + BatchNormalization + Early Stopping
            - **Balanceado:** Class weights optimizados para clases desbalanceadas
            
            **🎯 Interpretación Clínica:**
            - Detecta **{metadata.get('recall', 0)*100:.0f} de cada 100 casos reales** de ictus ({metadata.get('recall', 0)*100:.0f}% recall)
            - De cada **100 alertas, ~{metadata.get('precision', 0)*100:.0f} son casos reales** ({metadata.get('precision', 0)*100:.1f}% precisión)
            - Optimizado para **screening médico inicial**
            - Minimiza casos perdidos priorizando la detección temprana
            
            **⚙️ Proceso de Optimización:**
            - Entrenamiento de 5 arquitecturas diferentes
            - Búsqueda exhaustiva de umbrales (1,980 combinaciones probadas)
            - Selección basada en balance precisión-recall para uso médico
            """)
        else:
            st.markdown("""
            Este sistema utiliza un **Modelo de Perceptrón Multicapa (MLP)** entrenado para la detección de riesgo de ictus.
            
            **🏆 Características del Modelo:**
            - **Arquitectura:** Red Neuronal con capas densas
            - **Entrenamiento:** Basado en datos clínicos reales
            - **Objetivo:** Detección temprana de riesgo de ictus
            
            **🎯 Uso Clínico:**
            - Herramienta de apoyo para screening inicial
            - No reemplaza el diagnóstico médico profesional
            - Optimizado para detección temprana
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