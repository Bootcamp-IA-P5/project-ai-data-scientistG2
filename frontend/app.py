# IMPORTACIÓN DE LIBRERÍAS
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    roc_auc_score,
    accuracy_score,
)  # Se añaden métricas de clasificación
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from pathlib import Path

# IMPORTACIÓN DEL CLIENTE API
from api_client import (
    check_backend_status,
    save_prediction_to_backend,
    get_prediction_history,
)

# --- CONFIGURACIÓN INICIAL ---
# Configuración de la página principal
st.set_page_config(
    page_title="Stroke Risk Prediction System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
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
# Asegúrate de que los archivos .pkl en 'notebooks/models/' existan
AVAILABLE_MODELS = {
    "Stroke Prediction Model": os.path.join(
        SCRIPT_DIR, "notebooks", "models", "ictus_model_20251113_211435.pkl"
    )
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
        if isinstance(loaded_data, dict) and "model" in loaded_data:
            model = loaded_data["model"]
            scaler = loaded_data.get("scaler", None)
            feature_names = loaded_data.get("feature_names", None)
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
        train_path = os.path.join(
            SCRIPT_DIR, "data", "processed", "stroke_data_processed.csv"
        )
        test_path = os.path.join(
            SCRIPT_DIR, "data", "processed", "stroke_data_processed_test.csv"
        )
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        # Asume que la columna objetivo es 'stroke'
        if "stroke" not in df_train.columns or "stroke" not in df_test.columns:
            st.error(
                "Error: La columna objetivo 'stroke' (0 o 1) no se encontró en los archivos de datos."
            )
            return None, None, False

        return df_train, df_test, True
    except FileNotFoundError as e:
        st.error(f"Error: No se encontraron los archivos de datos: {str(e)}")
        st.error("Asegúrate de que existan los archivos:")
        st.code(
            f"""
        {train_path}
        {test_path}
        """
        )
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
    "Selecciona el Algoritmo de Clasificación:", available_names, index=0
)

st.sidebar.subheader("📊 Estado de Modelos")
for name, path in AVAILABLE_MODELS.items():
    if name in available_models:
        st.sidebar.success(f"✅ {name}")
    else:
        st.sidebar.error(f"❌ {name}")

# Verificar estado del backend
st.sidebar.markdown("---")
st.sidebar.subheader("🔌 Estado del Backend")
backend_available = check_backend_status()

# CARGA DE DATOS (Opcional - solo para EDA y Evaluación)
df_train, df_test, data_loaded = load_data()

# CONFIGURACIÓN DE DATASET Y MODELO
if data_loaded:
    dataset_option = st.sidebar.selectbox(
        "Selecciona el Dataset para Evaluación:",
        ["Dataset de Entrenamiento", "Dataset de Test", "Ambos Datasets"],
        index=1,
    )
else:
    st.sidebar.warning("⚠️ Datasets de evaluación no disponibles")
    st.sidebar.info("Las pestañas EDA y Evaluación estarán deshabilitadas")
    dataset_option = None

st.sidebar.info(f"🤖 Modelo cargado: {model_type}")
model_path = available_models[model_type]
loaded_model, scaler, feature_names, model_loaded_successfully = load_pretrained_model(
    model_path
)

st.session_state.model_name = model_type

if not model_loaded_successfully:
    st.error(f"No se pudo cargar el modelo: {model_type}")
    st.stop()

# Configurar DataFrame solo si los datos están disponibles
if data_loaded:
    if dataset_option == "Dataset de Entrenamiento":
        df = df_train.copy()
        st.info(f"📊 Usando Dataset de Entrenamiento ({len(df)} registros)")
    elif dataset_option == "Dataset de Test":
        df = df_test.copy()
        st.info(f"📊 Usando Dataset de Test ({len(df)} registros)")
    else:
        df = pd.concat([df_train, df_test], ignore_index=True)
        st.info(
            f"📊 Usando Ambos Datasets ({len(df)} registros: {len(df_train)} entrenamiento + {len(df_test)} test)"
        )
else:
    df = None

# --- INTERFAZ PRINCIPAL - SISTEMA DE PESTAÑAS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Exploración de Datos (EDA)",
        "🔮 Evaluación",
        "📈 Métricas del Modelo",
        "🎯 Predicción Individual",
        "📜 Historial de Predicciones",
    ]
)

# ----------------------------------------------------
# TAB 1: EXPLORACIÓN DE DATOS (EDA)
# ----------------------------------------------------
with tab1:
    st.header("🧠 Exploración de Datos de Ictus (EDA)")

    if not data_loaded or df is None:
        st.warning("⚠️ Esta pestaña requiere los datasets de evaluación")
        st.info("📁 Archivos necesarios:")
        st.code(
            """
data/processed/stroke_data_processed.csv
data/processed/stroke_data_processed_test.csv
"""
        )
        st.markdown(
            "💡 **Esta pestaña es opcional.** Puedes usar la app para predicciones individuales sin estos archivos."
        )
    else:
        st.warning(
            "⚠️ **Nota:** Esta pestaña muestra un ejemplo de EDA. **Asegúrate de que las columnas `gender`, `age` y `stroke` existan en tu dataset.**"
        )
        st.markdown(
            "Si tu dataset tiene otras columnas, adapta los gráficos o utiliza `st.dataframe(df)` para ver la estructura."
        )
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
            if "stroke" in df.columns:
                stroke_counts = (
                    df["stroke"]
                    .value_counts(normalize=True)
                    .mul(100)
                    .rename({0: "No Ictus (0)", 1: "Ictus (1)"})
                )
                fig = px.bar(
                    stroke_counts,
                    x=stroke_counts.index,
                    y=stroke_counts.values,
                    labels={"x": "Clase", "y": "Porcentaje"},
                    title="Distribución del Ictus (0 vs 1)",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Columna 'stroke' no encontrada para la distribución.")

            st.subheader("Riesgo por Edad")
            if "age" in df.columns and "stroke" in df.columns:
                fig2 = px.histogram(
                    df,
                    x="age",
                    color="stroke",
                    barmode="overlay",
                    title="Distribución de Edad por Riesgo de Ictus",
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info(
                    "No se puede mostrar el gráfico de riesgo por edad (columnas no encontradas)."
                )

        st.subheader("Matriz de Correlación")
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_columns].corr()
        fig3 = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Matriz de Correlación de Características Numéricas",
        )
        st.plotly_chart(fig3, use_container_width=True)

# ----------------------------------------------------
# TAB 2: EVALUACIÓN DE MODELOS
# ----------------------------------------------------
with tab2:
    st.header("🔮 Evaluación del Modelo Pre-entrenado")

    if not data_loaded or df is None:
        st.warning("⚠️ Esta pestaña requiere los datasets de evaluación")
        st.info("📁 Archivos necesarios:")
        st.code(
            """
data/processed/stroke_data_processed.csv
data/processed/stroke_data_processed_test.csv
"""
        )
        st.markdown(
            "💡 **Esta pestaña es opcional.** Puedes usar la app para predicciones individuales sin estos archivos."
        )
    elif st.button("🔍 Evaluar Modelo en el Dataset Seleccionado", type="primary"):
        with st.spinner(f"Evaluando modelo {model_type}..."):
            try:
                # Separar características (X) y variable objetivo (y)
                TARGET_COL = "stroke"
                X = df.drop(TARGET_COL, axis=1, errors="ignore")
                y = df[TARGET_COL]

                # Las 20 columnas que el modelo espera (identificadas del entrenamiento)
                MODEL_EXPECTED_FEATURES = [
                    "age",
                    "hypertension",
                    "heart_disease",
                    "avg_glucose_level",
                    "bmi",
                    "risk_factors",
                    "age_risk_interaction",
                    "gender_encoded",
                    "ever_married_encoded",
                    "Residence_type_encoded",
                    "work_type_Private",
                    "work_type_Self-employed",
                    "smoking_status_never smoked",
                    "smoking_status_smokes",
                    "age_group_36-50",
                    "age_group_51-65",
                    "age_group_65+",
                    "bmi_category_Overweight",
                    "bmi_category_Obese",
                    "glucose_category_Prediabetes",
                ]

                st.info(f"🔍 El modelo espera {len(MODEL_EXPECTED_FEATURES)} features")
                st.info(f"📊 Dataset actual tiene {X.shape[1]} columnas")

                # Verificar qué columnas faltan o sobran
                missing_cols = set(MODEL_EXPECTED_FEATURES) - set(X.columns)
                extra_cols = set(X.columns) - set(MODEL_EXPECTED_FEATURES)

                if missing_cols:
                    st.warning(
                        f"⚠️ Columnas faltantes (se crearán con valor 0): {missing_cols}"
                    )
                    # Crear columnas faltantes con valor 0
                    for col in missing_cols:
                        X[col] = 0

                if extra_cols:
                    st.info(
                        f"ℹ️ Columnas extra en el dataset (serán ignoradas): {extra_cols}"
                    )

                # Seleccionar solo las 20 columnas en el orden correcto
                X_processed = X[MODEL_EXPECTED_FEATURES].copy()

                # Aplicar escalado si existe
                if scaler:
                    X_processed_array = scaler.transform(X_processed)
                else:
                    X_processed_array = X_processed.values

                # Predecir: usamos .predict() para clasificación (0 o 1) y .predict_proba() para probabilidad
                y_pred_class = loaded_model.predict(X_processed_array)

                # Intentar obtener probabilidades para AUC
                if hasattr(loaded_model, "predict_proba"):
                    y_pred_proba = loaded_model.predict_proba(X_processed_array)[
                        :, 1
                    ]  # Probabilidad de la clase positiva (ictus=1)
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
                st.write(
                    "- El modelo no es compatible con la estructura actual de datos."
                )
                st.write(
                    "- Faltan o sobran columnas o el preprocesamiento es diferente."
                )

        # VISUALIZACIONES POST-EVALUACIÓN
        if "y_pred_class" in st.session_state:
            y = st.session_state.y
            y_pred_class = st.session_state.y_pred_class

            # Matriz de Confusión simple
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(y, y_pred_class)

            st.subheader("Matriz de Confusión")
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=["No Ictus (0)", "Ictus (1)"],
                yticklabels=["No Ictus (0)", "Ictus (1)"],
                ax=ax,
            )
            ax.set_title("Matriz de Confusión")
            ax.set_xlabel("Predicción")
            ax.set_ylabel("Valor Real")
            st.pyplot(fig)

    elif data_loaded:
        st.info(
            "👆 Haz clic en 'Evaluar Modelo' para ver el rendimiento del modelo pre-entrenado en el dataset seleccionado."
        )

# ----------------------------------------------------
# TAB 3: MÉTRICAS DEL MODELO
# ----------------------------------------------------
with tab3:
    st.header("📈 Métricas del Modelo de Clasificación")

    if "y_pred_class" in st.session_state:
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
        st.markdown(
            f"**Accuracy:** {acc*100:.2f}% de los casos fueron clasificados correctamente."
        )
        st.markdown(
            f"**Recall (Sensibilidad):** {recall*100:.2f}% de los casos de **Ictus (1)** fueron detectados correctamente por el modelo. **(Clave para riesgo médico)**"
        )
        st.markdown(
            f"**Precision:** {precision*100:.2f}% de las predicciones de **Ictus (1)** fueron correctas."
        )

        if y_pred_proba is not None:
            st.subheader("📉 Distribución de Probabilidades Predichas (Clase 1)")
            fig = px.histogram(
                x=y_pred_proba,
                color=y.astype(str),
                nbins=50,
                title="Probabilidad de Ictus (Clase 1)",
            )
            fig.update_layout(
                xaxis_title="Probabilidad Predicha", legend_title="Clase Real"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "El modelo no proporciona probabilidades (`predict_proba`) para el cálculo de AUC y el gráfico de distribución de probabilidad."
            )

    else:
        st.info("👆 Primero evalúa el modelo en la pestaña 'Evaluación'")

# ----------------------------------------------------
# TAB 4: PREDICCIÓN INDIVIDUAL CON RECOMENDACIÓN
# ----------------------------------------------------
with tab4:
    st.header("🎯 Predicción Individual de Riesgo de Ictus")

    # Cargar el modelo en session_state si no está cargado
    if "model" not in st.session_state and loaded_model is not None:
        st.session_state.model = loaded_model
        st.session_state.scaler = scaler
        st.session_state.feature_names = feature_names
        st.session_state.model_name = model_type

    if "model" in st.session_state:
        st.subheader("📝 Ingresa los Factores de Riesgo para la Predicción:")

        # ⚠️ Nota: Las características deben coincidir con las usadas para entrenar tu modelo.
        # Aquí se usa una suposición basada en un dataset común de ictus.

        col1, col2, col3 = st.columns(3)

        with col1:
            # Ejemplo de característica continua/numérica
            age = st.number_input(
                "Edad del Paciente", min_value=1, max_value=120, value=50, step=1
            )
            # Ejemplo de característica categórica
            hypertension = st.selectbox(
                "Hipertensión",
                [0, 1],
                format_func=lambda x: "Sí" if x == 1 else "No",
                index=0,
            )
            heart_disease = st.selectbox(
                "Enfermedad Cardíaca",
                [0, 1],
                format_func=lambda x: "Sí" if x == 1 else "No",
                index=0,
            )

        with col2:
            avg_glucose_level = st.number_input(
                "Nivel Promedio de Glucosa", min_value=50.0, max_value=300.0, value=90.0
            )
            bmi = st.number_input(
                "Índice de Masa Corporal (IMC)",
                min_value=10.0,
                max_value=60.0,
                value=25.0,
            )
            smoking_status = st.selectbox(
                "Estado de Tabaquismo", ["nunca fumó", "anteriormente fumó", "fuma"]
            )

        with col3:
            gender = st.selectbox("Género", ["Male", "Female", "Other"])
            ever_married = st.selectbox("Casado/a Anteriormente", ["Yes", "No"])
            work_type = st.selectbox(
                "Tipo de Trabajo",
                ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
            )
            residence_type = st.selectbox("Tipo de Residencia", ["Urban", "Rural"])

        st.markdown("---")

        # Umbral para la recomendación (puedes ajustarlo)
        RISK_THRESHOLD = st.slider(
            "Umbral de Riesgo para Recomendación (Probabilidad)", 0.0, 1.0, 0.5, 0.05
        )

        if st.button("🔮 **Predecir Riesgo de Ictus**", type="primary"):
            try:
                # 1. CREAR EL DATAFRAME DE ENTRADA
                input_data = pd.DataFrame(
                    {
                        "gender": [gender],
                        "age": [age],
                        "hypertension": [hypertension],
                        "heart_disease": [heart_disease],
                        "ever_married": [ever_married],
                        "work_type": [work_type],
                        "Residence_type": [residence_type],
                        "avg_glucose_level": [avg_glucose_level],
                        "bmi": [bmi],
                        "smoking_status": [smoking_status],
                    }
                )

                # 2. FEATURE ENGINEERING (igual que en el preprocesamiento)
                # Crear rangos de edad
                input_data["age_group"] = pd.cut(
                    input_data["age"],
                    bins=[0, 18, 35, 50, 65, 100],
                    labels=["0-18", "19-35", "36-50", "51-65", "65+"],
                )

                # Crear feature de factores de riesgo
                input_data["risk_factors"] = (
                    input_data["hypertension"] + input_data["heart_disease"]
                )

                # Crear categorías de BMI
                input_data["bmi_category"] = pd.cut(
                    input_data["bmi"],
                    bins=[0, 18.5, 25, 30, 100],
                    labels=["Underweight", "Normal", "Overweight", "Obese"],
                )

                # Crear categorías de glucosa
                input_data["glucose_category"] = pd.cut(
                    input_data["avg_glucose_level"],
                    bins=[0, 100, 125, 300],
                    labels=["Normal", "Prediabetes", "Diabetes"],
                )

                # Crear interacción edad-riesgo
                input_data["age_risk_interaction"] = (
                    input_data["age"] * input_data["risk_factors"]
                )

                # 3. CODIFICACIÓN (Label Encoding para binarias)
                # Codificar gender
                gender_map = {"Female": 0, "Male": 1, "Other": 2}
                input_data["gender_encoded"] = input_data["gender"].map(gender_map)

                # Codificar ever_married
                married_map = {"No": 0, "Yes": 1}
                input_data["ever_married_encoded"] = input_data["ever_married"].map(
                    married_map
                )

                # Codificar Residence_type
                residence_map = {"Rural": 0, "Urban": 1}
                input_data["Residence_type_encoded"] = input_data["Residence_type"].map(
                    residence_map
                )

                # 4. ONE-HOT ENCODING para variables categóricas con múltiples valores
                categorical_cols = [
                    "work_type",
                    "smoking_status",
                    "age_group",
                    "bmi_category",
                    "glucose_category",
                ]
                input_data = pd.get_dummies(
                    input_data, columns=categorical_cols, drop_first=True, dtype=int
                )

                # 5. ELIMINAR COLUMNAS ORIGINALES CATEGÓRICAS
                cols_to_drop = ["gender", "ever_married", "Residence_type"]
                input_data = input_data.drop(columns=cols_to_drop, errors="ignore")

                # 2. SELECCIONAR LAS 20 COLUMNAS CORRECTAS
                # El modelo fue entrenado con estas 20 features específicas
                MODEL_EXPECTED_FEATURES = [
                    "age",
                    "hypertension",
                    "heart_disease",
                    "avg_glucose_level",
                    "bmi",
                    "risk_factors",
                    "age_risk_interaction",
                    "gender_encoded",
                    "ever_married_encoded",
                    "Residence_type_encoded",
                    "work_type_Private",
                    "work_type_Self-employed",
                    "smoking_status_never smoked",
                    "smoking_status_smokes",
                    "age_group_36-50",
                    "age_group_51-65",
                    "age_group_65+",
                    "bmi_category_Overweight",
                    "bmi_category_Obese",
                    "glucose_category_Prediabetes",
                ]

                # Crear columnas faltantes y establecerlas a 0
                for col in MODEL_EXPECTED_FEATURES:
                    if col not in input_data.columns:
                        input_data[col] = 0

                # Reordenar al orden esperado por el modelo (solo las 20 columnas)
                input_data = input_data[MODEL_EXPECTED_FEATURES]

                # 3. ESCALAR DATOS (si es necesario)
                if st.session_state.scaler is not None:
                    input_data_scaled = st.session_state.scaler.transform(input_data)
                else:
                    input_data_scaled = input_data.values

                # 4. HACER PREDICCIÓN
                prediction_class = st.session_state.model.predict(input_data_scaled)[0]

                if hasattr(st.session_state.model, "predict_proba"):
                    prediction_proba = st.session_state.model.predict_proba(
                        input_data_scaled
                    )[0, 1]
                else:
                    prediction_proba = None

                # 5. MOSTRAR RESULTADOS

                st.subheader("✅ Resultado de la Predicción")

                if prediction_proba is not None:
                    st.metric(
                        "Probabilidad de Ictus (Clase 1)", f"{prediction_proba:.2f}"
                    )

                if prediction_class == 1:
                    result_text = "🔴 **RIESGO ALTO DE ICTUS**"
                    result_color = "red"
                else:
                    result_text = "🟢 **RIESGO BAJO/MODERADO DE ICTUS**"
                    result_color = "green"

                st.markdown(
                    f"**Clasificación del Modelo:** <span style='color:{result_color}; font-size: 24px'>{result_text}</span>",
                    unsafe_allow_html=True,
                )

                # 6. RECOMENDACIÓN MÉDICA BASADA EN EL UMBRAL
                st.subheader("🏥 Recomendación Médica")

                if prediction_proba is not None and prediction_proba >= RISK_THRESHOLD:
                    st.error(
                        f"""
                    **¡ATENCIÓN!**
                    Basado en una probabilidad predicha de **{prediction_proba:.2f}** (superior al umbral de {RISK_THRESHOLD:.2f}), 
                    el modelo sugiere un riesgo significativo de Ictus.
                    """
                    )
                    st.markdown(
                        """
                    **Recomendaciones:**
                    * **Visitar a un Especialista:** Se aconseja encarecidamente una **consulta inmediata con un neurólogo o cardiólogo** para una evaluación clínica exhaustiva.
                    * **Pruebas Correspondientes:** El especialista podría recomendar pruebas como resonancias magnéticas, tomografías computarizadas o ecocardiogramas para confirmar el riesgo.
                    * **Modificación de Estilo de Vida:** Mantener un control riguroso de la presión arterial, glucosa, peso (IMC) y dejar de fumar.
                    """
                    )
                else:
                    st.success(
                        """
                    El modelo predice un riesgo bajo o moderado de Ictus.
                    """
                    )
                    st.markdown(
                        """
                    **Recomendaciones:**
                    * **Control Médico Rutinario:** Continuar con las revisiones médicas de rutina.
                    * **Prevención:** Mantener un estilo de vida saludable: dieta equilibrada, ejercicio regular y evitar el tabaquismo y el consumo excesivo de alcohol.
                    * **Monitoreo de Síntomas:** Estar atento a síntomas de advertencia (FAST: Face drooping, Arm weakness, Speech difficulty, Time to call emergency).
                    """
                    )

                st.info(f"🤖 **Modelo utilizado:** {st.session_state.model_name}")
                st.subheader("📋 Datos Procesados (enviados al modelo):")
                st.dataframe(input_data)

                # Guardar predicción en el backend si está disponible
                if backend_available:
                    with st.spinner("Guardando predicción en el historial..."):
                        input_dict = {
                            "age": age,
                            "gender": gender,
                            "hypertension": hypertension,
                            "heart_disease": heart_disease,
                            "ever_married": ever_married,
                            "work_type": work_type,
                            "Residence_type": residence_type,
                            "avg_glucose_level": avg_glucose_level,
                            "bmi": bmi,
                            "smoking_status": smoking_status,
                        }

                        if save_prediction_to_backend(
                            input_dict, prediction_class, prediction_proba
                        ):
                            st.success(
                                "✅ Predicción guardada en el historial del backend"
                            )
                        else:
                            st.warning(
                                "⚠️ No se pudo guardar la predicción en el backend"
                            )

            except Exception as e:
                st.error(f"Error al hacer la predicción: {str(e)}")
                st.info(
                    "💡 Asegúrate de que los datos de entrada (incluyendo las columnas de OHE) coincidan exactamente con la estructura de entrenamiento de tu modelo."
                )

    else:
        st.info(
            "👆 Primero evalúa el modelo en la pestaña 'Evaluación' para cargar el modelo y el preprocesador en la sesión."
        )

# ----------------------------------------------------
# TAB 5: HISTORIAL DE PREDICCIONES
# ----------------------------------------------------
with tab5:
    st.header("📜 Historial de Predicciones")

    if not backend_available:
        st.warning(
            "⚠️ El backend no está disponible. Inicia el servidor para ver el historial."
        )
        st.info("💡 Para iniciar el backend, ejecuta:")
        st.code("cd backend/database && uvicorn main:app --reload", language="bash")
    else:
        st.info("📊 Mostrando las últimas predicciones guardadas en el backend")

        col1, col2 = st.columns([3, 1])
        with col1:
            limit = st.slider("Número de predicciones a mostrar:", 5, 100, 20, 5)
        with col2:
            if st.button("🔄 Actualizar", type="secondary"):
                st.rerun()

        with st.spinner("Cargando historial de predicciones..."):
            predictions = get_prediction_history(limit=limit)

        if predictions:
            st.success(f"✅ Se encontraron {len(predictions)} predicciones")

            # Convertir a DataFrame para mejor visualización
            df_predictions = pd.DataFrame(predictions)

            # Formatear la columna timestamp si existe
            if "timestamp" in df_predictions.columns:
                df_predictions["timestamp"] = pd.to_datetime(
                    df_predictions["timestamp"]
                )
                df_predictions = df_predictions.sort_values(
                    "timestamp", ascending=False
                )

            # Mostrar tabla con formato
            st.subheader("📋 Tabla de Predicciones")
            st.dataframe(
                df_predictions,
                use_container_width=True,
                column_config={
                    "id": st.column_config.NumberColumn("ID", format="%d"),
                    "timestamp": st.column_config.DatetimeColumn(
                        "Fecha y Hora", format="DD/MM/YYYY HH:mm:ss"
                    ),
                    "prediction_result": st.column_config.TextColumn("Resultado"),
                    "confidence": st.column_config.NumberColumn(
                        "Confianza", format="%.4f"
                    ),
                    "input_data": st.column_config.TextColumn("Datos de Entrada"),
                },
            )

            # Estadísticas del historial
            st.subheader("📊 Estadísticas del Historial")
            col1, col2, col3 = st.columns(3)

            with col1:
                total_predictions = len(df_predictions)
                st.metric("Total de Predicciones", total_predictions)

            with col2:
                if "prediction_result" in df_predictions.columns:
                    high_risk = (
                        df_predictions["prediction_result"]
                        .str.contains("ALTO", case=False, na=False)
                        .sum()
                    )
                    st.metric("Predicciones de Alto Riesgo", high_risk)

            with col3:
                if "confidence" in df_predictions.columns:
                    avg_confidence = df_predictions["confidence"].mean()
                    st.metric(
                        "Confianza Promedio",
                        (
                            f"{avg_confidence:.4f}"
                            if not pd.isna(avg_confidence)
                            else "N/A"
                        ),
                    )

            # Gráfico de tendencias si hay suficientes datos
            if (
                len(df_predictions) >= 5
                and "timestamp" in df_predictions.columns
                and "confidence" in df_predictions.columns
            ):
                st.subheader("📈 Tendencia de Confianza en las Predicciones")
                fig = px.line(
                    df_predictions,
                    x="timestamp",
                    y="confidence",
                    title="Evolución de la Confianza del Modelo",
                    markers=True,
                )
                fig.update_layout(xaxis_title="Fecha", yaxis_title="Confianza")
                st.plotly_chart(fig, use_container_width=True)

            # Distribución de resultados
            if "prediction_result" in df_predictions.columns:
                st.subheader("📊 Distribución de Resultados")
                result_counts = df_predictions["prediction_result"].value_counts()
                fig2 = px.pie(
                    values=result_counts.values,
                    names=result_counts.index,
                    title="Distribución de Predicciones por Categoría",
                )
                st.plotly_chart(fig2, use_container_width=True)

        else:
            st.info("📭 No hay predicciones guardadas en el historial todavía.")
            st.markdown(
                "💡 Realiza algunas predicciones en la pestaña **'🎯 Predicción Individual'** para comenzar a crear tu historial."
            )

# FOOTER DE LA APLICACIÓN
st.markdown("---")
st.markdown(
    f"**Stroke Risk Predictor** - Herramienta de apoyo | Modelos disponibles: {len(available_models)}/1"
)
st.markdown("Desarrollado con ❤️ usando Streamlit")
