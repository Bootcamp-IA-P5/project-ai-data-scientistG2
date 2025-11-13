# IMPORTACIÓN DE LIBRERÍAS
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, roc_auc_score, accuracy_score # Se añaden métricas de clasificación
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from pathlib import Path

# --- CONFIGURACIÓN INICIAL ---
# Configuración de la página principal
st.set_page_config(
    page_title="Stroke Risk Prediction System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CONFIGURACIÓN DE RUTAS DEL SISTEMA
try:
    # Intenta obtener el directorio del script
    SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    # Fallback si __file__ no está definido (e.g., en un entorno interactivo)
    SCRIPT_DIR = os.path.dirname(os.getcwd())

# INTERFAZ DE USUARIO - ENCABEZADO PRINCIPAL
st.title("🧠 Stroke Risk Prediction System")
st.markdown("---")

# CONFIGURACIÓN DE MODELOS DISPONIBLES
# Asegúrate de que los archivos .pkl en 'data/results/' existan
AVAILABLE_MODELS = {
    "MLP Neural Network": os.path.join(SCRIPT_DIR, "data", "results", "mlp_model.pkl")
}

# --- FUNCIONES DE CARGA Y VERIFICACIÓN ---

@st.cache_data
def check_available_models():
    """Verifica qué modelos existen en el disco."""
    available = {}
    for name, path in AVAILABLE_MODELS.items():
        if os.path.exists(path):
            available[name] = path
    return available

@st.cache_resource
def load_pretrained_model(model_path):
    """Carga un modelo pre-entrenado desde disco."""
    try:
        loaded_data = joblib.load(model_path)
        # Asume que el modelo puede ser un diccionario o el modelo directamente
        if isinstance(loaded_data, dict) and 'model' in loaded_data:
            model = loaded_data['model']
            scaler = loaded_data.get('scaler', None)
            feature_names = loaded_data.get('feature_names', None)
        else:
            model = loaded_data
            scaler = None
            feature_names = None
        return model, scaler, feature_names, True
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None, None, None, False

@st.cache_data
def load_data():
    """Carga los datasets de ictus."""
    try:
        # ⚠️ REEMPLAZA ESTOS NOMBRES CON LOS DE TUS ARCHIVOS DE DATOS DE ICTUS
        train_path = os.path.join(SCRIPT_DIR, "data", "processed", "stroke_data_processed.csv") 
        test_path = os.path.join(SCRIPT_DIR, "data", "processed", "stroke_data_processed_test.csv")
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
        # Asume que la columna objetivo es 'stroke'
        if 'stroke' not in df_train.columns or 'stroke' not in df_test.columns:
            st.error("Error: La columna objetivo 'stroke' (0 o 1) no se encontró en los archivos de datos.")
            return None, None, False
        
        return df_train, df_test, True
    except FileNotFoundError as e:
        st.error(f"Error: No se encontraron los archivos de datos: {str(e)}")
        st.error("Asegúrate de que existan los archivos:")
        st.code(f"""
        {train_path}
        {test_path}
        """)
        return None, None, False

# --- SIDEBAR - PANEL DE CONFIGURACIÓN ---
st.sidebar.header("🔧 Configuración del Modelo")
available_models = check_available_models()

# Manejo de error si no hay modelos
if not available_models:
    st.sidebar.error("❌ No se encontraron modelos entrenados")
    st.stop()

available_names = list(available_models.keys())
model_type = st.sidebar.selectbox(
    "Selecciona el Algoritmo de Clasificación:",
    available_names,
    index=0
)

st.sidebar.subheader("📊 Estado de Modelos")
for name, path in AVAILABLE_MODELS.items():
    if name in available_models:
        st.sidebar.success(f"✅ {name}")
    else:
        st.sidebar.error(f"❌ {name}")

# CARGA DE DATOS
df_train, df_test, data_loaded = load_data()
if not data_loaded:
    st.stop()

# CONFIGURACIÓN DE DATASET Y MODELO
dataset_option = st.sidebar.selectbox(
    "Selecciona el Dataset para Evaluación:",
    ["Dataset de Entrenamiento", "Dataset de Test", "Ambos Datasets"],
    index=1
)

st.sidebar.info(f"🤖 Modelo cargado: {model_type}")
model_path = available_models[model_type]
loaded_model, scaler, feature_names, model_loaded_successfully = load_pretrained_model(model_path)

st.session_state.model_name = model_type

if not model_loaded_successfully:
    st.error(f"No se pudo cargar el modelo: {model_type}")
    st.stop()

if dataset_option == "Dataset de Entrenamiento":
    df = df_train.copy()
    st.info(f"📊 Usando Dataset de Entrenamiento ({len(df)} registros)")
elif dataset_option == "Dataset de Test":
    df = df_test.copy()
    st.info(f"📊 Usando Dataset de Test ({len(df)} registros)")
else:
    df = pd.concat([df_train, df_test], ignore_index=True)
    st.info(f"📊 Usando Ambos Datasets ({len(df)} registros: {len(df_train)} entrenamiento + {len(df_test)} test)")

# --- INTERFAZ PRINCIPAL - SISTEMA DE PESTAÑAS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Exploración de Datos (EDA)", "🔮 Evaluación", "📈 Métricas del Modelo", "🎯 Predicción Individual"])

# ----------------------------------------------------
# TAB 1: EXPLORACIÓN DE DATOS (EDA)
# ----------------------------------------------------
with tab1:
    st.header("🧠 Exploración de Datos de Ictus (EDA)")
    
    st.warning("⚠️ **Nota:** Esta pestaña muestra un ejemplo de EDA. **Asegúrate de que las columnas `gender`, `age` y `stroke` existan en tu dataset.**")
    st.markdown("Si tu dataset tiene otras columnas, adapta los gráficos o utiliza `st.dataframe(df)` para ver la estructura.")
    st.markdown(f"**Dataset Actual:** {dataset_option} ({len(df)} registros)")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Información General")
        st.write(f"**Forma del dataset:** {df.shape}")
        st.write("**Primeras 5 filas:**")
        st.dataframe(df.head())
        st.write("**Estadísticas descriptivas:**")
        st.dataframe(df.describe())

    with col2:
        st.subheader("Distribución de la Variable Objetivo ('stroke')")
        if 'stroke' in df.columns:
            stroke_counts = df['stroke'].value_counts(normalize=True).mul(100).rename({0: 'No Ictus (0)', 1: 'Ictus (1)'})
            fig = px.bar(stroke_counts, x=stroke_counts.index, y=stroke_counts.values,
                         labels={'x': 'Clase', 'y': 'Porcentaje'},
                         title="Distribución del Ictus (0 vs 1)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Columna 'stroke' no encontrada para la distribución.")
            
        st.subheader("Riesgo por Edad")
        if 'age' in df.columns and 'stroke' in df.columns:
            fig2 = px.histogram(df, x='age', color='stroke', barmode='overlay',
                                title="Distribución de Edad por Riesgo de Ictus")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No se puede mostrar el gráfico de riesgo por edad (columnas no encontradas).")
    
    st.subheader("Matriz de Correlación")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_columns].corr()
    fig3 = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                    title="Matriz de Correlación de Características Numéricas")
    st.plotly_chart(fig3, use_container_width=True)

# ----------------------------------------------------
# TAB 2: EVALUACIÓN DE MODELOS
# ----------------------------------------------------
with tab2:
    st.header("🔮 Evaluación del Modelo Pre-entrenado")
    
    if st.button("🔍 Evaluar Modelo en el Dataset Seleccionado", type="primary"):
        with st.spinner(f"Evaluando modelo {model_type}..."):
            try:
                # Separar características (X) y variable objetivo (y)
                TARGET_COL = 'stroke' # Columna objetivo para la predicción de ictus
                X = df.drop(TARGET_COL, axis=1, errors='ignore') # Usamos errors='ignore' por si la columna ya fue eliminada o renombrada
                y = df[TARGET_COL]

                # Verificar y escalar datos si es necesario
                # Si el modelo espera un conjunto específico de características
                if feature_names is not None and len(feature_names) > 0:
                    X_processed = X[feature_names].copy()
                    if scaler:
                         X_processed_array = scaler.transform(X_processed)
                    else:
                         X_processed_array = X_processed.values
                else:
                    X_processed_array = X.values # Si no hay 'feature_names', usa todo el dataframe

                # Predecir: usamos .predict() para clasificación (0 o 1) y .predict_proba() para probabilidad
                y_pred_class = loaded_model.predict(X_processed_array)
                
                # Intentar obtener probabilidades para AUC
                if hasattr(loaded_model, "predict_proba"):
                    y_pred_proba = loaded_model.predict_proba(X_processed_array)[:, 1] # Probabilidad de la clase positiva (ictus=1)
                else:
                    y_pred_proba = None
                
                # ALMACENAMIENTO EN SESSION STATE
                st.session_state.model = loaded_model
                st.session_state.scaler = scaler
                st.session_state.feature_names = feature_names
                st.session_state.X = X_processed_array
                st.session_state.y = y
                st.session_state.y_pred_class = y_pred_class
                st.session_state.y_pred_proba = y_pred_proba
                st.session_state.model_name = model_type
                
                st.success("✅ Modelo evaluado exitosamente!")
                
            except Exception as e:
                st.error(f"Error al hacer predicciones: {str(e)}")
                st.info("💡 Posibles causas:")
                st.write("- El modelo no es compatible con la estructura actual de datos.")
                st.write("- Faltan o sobran columnas o el preprocesamiento es diferente.")
        
        # VISUALIZACIONES POST-EVALUACIÓN
        if 'y_pred_class' in st.session_state:
            y = st.session_state.y
            y_pred_class = st.session_state.y_pred_class
            
            # Matriz de Confusión simple
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y, y_pred_class)
            
            st.subheader("Matriz de Confusión")
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['No Ictus (0)', 'Ictus (1)'],
                        yticklabels=['No Ictus (0)', 'Ictus (1)'], ax=ax)
            ax.set_title("Matriz de Confusión")
            ax.set_xlabel("Predicción")
            ax.set_ylabel("Valor Real")
            st.pyplot(fig)
    
    else:
        st.info("👆 Haz clic en 'Evaluar Modelo' para ver el rendimiento del modelo pre-entrenado en el dataset seleccionado.")

# ----------------------------------------------------
# TAB 3: MÉTRICAS DEL MODELO
# ----------------------------------------------------
with tab3:
    st.header("📈 Métricas del Modelo de Clasificación")
    
    if 'y_pred_class' in st.session_state:
        y = st.session_state.y
        y_pred_class = st.session_state.y_pred_class
        y_pred_proba = st.session_state.y_pred_proba
        model_name = st.session_state.model_name
        
        # Métricas de Clasificación
        acc = accuracy_score(y, y_pred_class)
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y, y_pred_class, zero_division=0)
        recall = recall_score(y, y_pred_class, zero_division=0)
        f1 = f1_score(y, y_pred_class, zero_division=0)
        
        auc = roc_auc_score(y, y_pred_proba) if y_pred_proba is not None else "N/A"
        
        st.subheader(f"📊 Métricas de Clasificación: {model_name}")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy (Precisión Global)", f"{acc:.4f}")
        with col2:
            st.metric("Precision", f"{precision:.4f}")
        with col3:
            st.metric("Recall (Sensibilidad)", f"{recall:.4f}")
        with col4:
            st.metric("F1-Score", f"{f1:.4f}")
        with col5:
            st.metric("AUC-ROC", f"{auc:.4f}" if auc != "N/A" else "N/A")
        
        st.subheader("📋 Interpretación de Métricas Clave")
        st.markdown(f"**Accuracy:** {acc*100:.2f}% de los casos fueron clasificados correctamente.")
        st.markdown(f"**Recall (Sensibilidad):** {recall*100:.2f}% de los casos de **Ictus (1)** fueron detectados correctamente por el modelo. **(Clave para riesgo médico)**")
        st.markdown(f"**Precision:** {precision*100:.2f}% de las predicciones de **Ictus (1)** fueron correctas.")
        
        if y_pred_proba is not None:
             st.subheader("📉 Distribución de Probabilidades Predichas (Clase 1)")
             fig = px.histogram(x=y_pred_proba, color=y.astype(str), nbins=50,
                                title="Probabilidad de Ictus (Clase 1)")
             fig.update_layout(xaxis_title="Probabilidad Predicha", legend_title="Clase Real")
             st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("El modelo no proporciona probabilidades (`predict_proba`) para el cálculo de AUC y el gráfico de distribución de probabilidad.")

    else:
        st.info("👆 Primero evalúa el modelo en la pestaña 'Evaluación'")

# ----------------------------------------------------
# TAB 4: PREDICCIÓN INDIVIDUAL CON RECOMENDACIÓN
# ----------------------------------------------------
with tab4:
    st.header("🎯 Predicción Individual de Riesgo de Ictus")
    
    if 'model' in st.session_state:
        st.subheader("📝 Ingresa los Factores de Riesgo para la Predicción:")
        
        # ⚠️ Nota: Las características deben coincidir con las usadas para entrenar tu modelo.
        # Aquí se usa una suposición basada en un dataset común de ictus.
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Ejemplo de característica continua/numérica
            age = st.number_input("Edad del Paciente", min_value=1, max_value=120, value=50, step=1)
            # Ejemplo de característica categórica
            hypertension = st.selectbox("Hipertensión", [0, 1], format_func=lambda x: 'Sí' if x == 1 else 'No', index=0)
            heart_disease = st.selectbox("Enfermedad Cardíaca", [0, 1], format_func=lambda x: 'Sí' if x == 1 else 'No', index=0)
        
        with col2:
            avg_glucose_level = st.number_input("Nivel Promedio de Glucosa", min_value=50.0, max_value=300.0, value=90.0)
            bmi = st.number_input("Índice de Masa Corporal (IMC)", min_value=10.0, max_value=60.0, value=25.0)
            smoking_status = st.selectbox("Estado de Tabaquismo", ["nunca fumó", "anteriormente fumó", "fuma"])
        
        with col3:
            gender = st.selectbox("Género", ["Male", "Female", "Other"])
            ever_married = st.selectbox("Casado/a Anteriormente", ["Yes", "No"])
            work_type = st.selectbox("Tipo de Trabajo", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
            residence_type = st.selectbox("Tipo de Residencia", ["Urban", "Rural"])
        
        
        st.markdown("---")
        
        # Umbral para la recomendación (puedes ajustarlo)
        RISK_THRESHOLD = st.slider("Umbral de Riesgo para Recomendación (Probabilidad)", 0.0, 1.0, 0.5, 0.05)


        if st.button("🔮 **Predecir Riesgo de Ictus**", type="primary"):
            try:
                # 1. CREAR EL DATAFRAME DE ENTRADA (con OHE manual o similar)
                input_data = pd.DataFrame({
                    'gender': [gender],
                    'age': [age],
                    'hypertension': [hypertension],
                    'heart_disease': [heart_disease],
                    'ever_married': [ever_married],
                    'work_type': [work_type],
                    'Residence_type': [residence_type],
                    'avg_glucose_level': [avg_glucose_level],
                    'bmi': [bmi],
                    'smoking_status': [smoking_status]
                })

                # Aplica One-Hot Encoding (OHE) a las columnas categóricas como si fuera un preprocesamiento
                # ⚠️ ESTA PARTE DEBE COINCIDIR EXACTAMENTE CON TU PREPROCESAMIENTO DE ENTRENAMIENTO ⚠️
                input_data = pd.get_dummies(input_data, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=False)
                
                # 2. ASEGURAR COLUMNAS (Añadir las columnas faltantes del OHE si es necesario, y reordenar)
                if st.session_state.feature_names is not None:
                    # Crear columnas faltantes y establecerlas a 0
                    for col in st.session_state.feature_names:
                        if col not in input_data.columns:
                            input_data[col] = 0
                    
                    # Reordenar al orden esperado por el modelo
                    input_data = input_data[st.session_state.feature_names]
                else:
                    st.warning("No se encontraron nombres de características, se asume que el orden de las columnas de entrada es correcto.")


                # 3. ESCALAR DATOS (si es necesario)
                if st.session_state.scaler is not None:
                    input_data_scaled = st.session_state.scaler.transform(input_data)
                else:
                    input_data_scaled = input_data.values
                
                # 4. HACER PREDICCIÓN
                prediction_class = st.session_state.model.predict(input_data_scaled)[0]
                
                if hasattr(st.session_state.model, "predict_proba"):
                    prediction_proba = st.session_state.model.predict_proba(input_data_scaled)[0, 1]
                else:
                    prediction_proba = None
                
                # 5. MOSTRAR RESULTADOS
                
                st.subheader("✅ Resultado de la Predicción")

                if prediction_proba is not None:
                    st.metric("Probabilidad de Ictus (Clase 1)", f"{prediction_proba:.2f}")

                if prediction_class == 1:
                    result_text = "🔴 **RIESGO ALTO DE ICTUS**"
                    result_color = "red"
                else:
                    result_text = "🟢 **RIESGO BAJO/MODERADO DE ICTUS**"
                    result_color = "green"
                    
                st.markdown(f"**Clasificación del Modelo:** <span style='color:{result_color}; font-size: 24px'>{result_text}</span>", 
                           unsafe_allow_html=True)
                
                
                # 6. RECOMENDACIÓN MÉDICA BASADA EN EL UMBRAL
                st.subheader("🏥 Recomendación Médica")
                
                if prediction_proba is not None and prediction_proba >= RISK_THRESHOLD:
                    st.error(f"""
                    **¡ATENCIÓN!**
                    Basado en una probabilidad predicha de **{prediction_proba:.2f}** (superior al umbral de {RISK_THRESHOLD:.2f}), 
                    el modelo sugiere un riesgo significativo de Ictus.
                    """)
                    st.markdown("""
                    **Recomendaciones:**
                    * **Visitar a un Especialista:** Se aconseja encarecidamente una **consulta inmediata con un neurólogo o cardiólogo** para una evaluación clínica exhaustiva.
                    * **Pruebas Correspondientes:** El especialista podría recomendar pruebas como resonancias magnéticas, tomografías computarizadas o ecocardiogramas para confirmar el riesgo.
                    * **Modificación de Estilo de Vida:** Mantener un control riguroso de la presión arterial, glucosa, peso (IMC) y dejar de fumar.
                    """)
                else:
                    st.success("""
                    El modelo predice un riesgo bajo o moderado de Ictus.
                    """)
                    st.markdown("""
                    **Recomendaciones:**
                    * **Control Médico Rutinario:** Continuar con las revisiones médicas de rutina.
                    * **Prevención:** Mantener un estilo de vida saludable: dieta equilibrada, ejercicio regular y evitar el tabaquismo y el consumo excesivo de alcohol.
                    * **Monitoreo de Síntomas:** Estar atento a síntomas de advertencia (FAST: Face drooping, Arm weakness, Speech difficulty, Time to call emergency).
                    """)
                
                st.info(f"🤖 **Modelo utilizado:** {st.session_state.model_name}")
                st.subheader("📋 Datos Procesados (enviados al modelo):")
                st.dataframe(input_data)
                    
            except Exception as e:
                st.error(f"Error al hacer la predicción: {str(e)}")
                st.info("💡 Asegúrate de que los datos de entrada (incluyendo las columnas de OHE) coincidan exactamente con la estructura de entrenamiento de tu modelo.")
    
    else:
        st.info("👆 Primero evalúa el modelo en la pestaña 'Evaluación' para cargar el modelo y el preprocesador en la sesión.")

# FOOTER DE LA APLICACIÓN
st.markdown("---")
st.markdown(f"**Stroke Risk Predictor** - Herramienta de apoyo | Modelos disponibles: {len(available_models)}/5")
st.markdown("Desarrollado con ❤️ usando Streamlit")