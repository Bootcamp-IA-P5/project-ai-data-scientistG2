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
import glob

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
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback si __file__ no está definido
    SCRIPT_DIR = os.getcwd()

# Ajustar si estamos en una subcarpeta (ej: frontend/)
if os.path.basename(SCRIPT_DIR) == "frontend":
    SCRIPT_DIR = os.path.dirname(SCRIPT_DIR)

print(f"📂 SCRIPT_DIR: {SCRIPT_DIR}")  # Debug


# FUNCIÓN PARA BUSCAR MODELOS
def find_all_models():
    """Busca automáticamente todos los modelos disponibles."""
    models_dir = os.path.join(SCRIPT_DIR, "models")

    print(f"🔍 Buscando modelos en: {models_dir}")

    if not os.path.exists(models_dir):
        print(f"❌ La carpeta models no existe: {models_dir}")
        return {}, {}

    files_in_dir = os.listdir(models_dir)
    print(f"📁 Archivos encontrados: {files_in_dir}")

    tabular_models = {}
    image_models = {}

    # Buscar modelos tabulares XGBoost (.pkl)
    pkl_files = [
        f for f in files_in_dir if f.endswith(".pkl") and "image" not in f.lower()
    ]
    for pkl_file in pkl_files:
        full_path = os.path.join(models_dir, pkl_file)

        # Distinguir entre XGBoost y MLP
        if "info" in pkl_file.lower():
            model_name = f"MLP ClassWeight ({pkl_file})"
        else:
            model_name = f"XGBoost ({pkl_file})"

        tabular_models[model_name] = full_path
        print(f"✅ Encontrado modelo tabular: {model_name}")

    # Buscar modelos de imágenes (.h5 o mejor_modelo_dense.keras)
    image_files = [
        f
        for f in files_in_dir
        if ("image" in f.lower() or f == "mejor_modelo_dense.keras")
        and f.endswith((".h5", ".keras"))
    ]
    for image_file in image_files:
        full_path = os.path.join(models_dir, image_file)
        model_name = f"Modelo de Imágenes ({image_file})"
        image_models[model_name] = full_path
        print(f"✅ Encontrado modelo de imágenes: {model_name}")

    return tabular_models, image_models


# CONFIGURACIÓN DE MODELOS DISPONIBLES
AVAILABLE_TABULAR_MODELS, AVAILABLE_IMAGE_MODELS = find_all_models()

print(f"\n📊 Modelos tabulares encontrados: {len(AVAILABLE_TABULAR_MODELS)}")
for name, path in AVAILABLE_TABULAR_MODELS.items():
    print(f"   • {name}: {path}")

print(f"\n🖼️ Modelos de imágenes encontrados: {len(AVAILABLE_IMAGE_MODELS)}")
for name, path in AVAILABLE_IMAGE_MODELS.items():
    print(f"   • {name}: {path}")

# INTERFAZ DE USUARIO - ENCABEZADO PRINCIPAL
st.title("🧠 Stroke Risk Prediction System")
st.markdown("---")

# --- FUNCIONES DE CARGA Y VERIFICACIÓN ---


@st.cache_data
def check_available_models():
    """Verifica qué modelos tabulares existen en el disco."""
    available = {}
    for name, path in AVAILABLE_TABULAR_MODELS.items():
        if os.path.exists(path):
            available[name] = path
        else:
            print(f"❌ Modelo no encontrado: {name} en {path}")
    print(f"✅ Modelos tabulares verificados: {len(available)}")
    return available


@st.cache_resource
def load_pretrained_model(model_path):
    """Carga un modelo pre-entrenado desde disco."""
    try:
        # Detectar tipo de modelo por extensión
        if model_path.endswith(".pkl"):
            # Modelo XGBoost/sklearn
            loaded_data = joblib.load(model_path)
            if isinstance(loaded_data, dict) and "model" in loaded_data:
                model = loaded_data["model"]
                scaler = loaded_data.get("scaler", None)
                feature_names = loaded_data.get("feature_names", None)
            else:
                model = loaded_data
                scaler = None
                feature_names = None
            return model, scaler, feature_names, True, "xgboost"

        elif model_path.endswith((".h5", ".keras")):
            # Modelo Keras/TensorFlow
            import tensorflow as tf

            model = tf.keras.models.load_model(model_path)
            return model, None, None, True, "keras"

        else:
            st.error(f"Formato de modelo no soportado: {model_path}")
            return None, None, None, False, None

    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        import traceback

        st.code(traceback.format_exc())
        return None, None, None, False, None


@st.cache_data
def load_data():
    """Carga los datasets de ictus."""
    try:
        train_path = os.path.join(
            SCRIPT_DIR, "data", "processed", "stroke_data_processed.csv"
        )
        test_path = os.path.join(
            SCRIPT_DIR, "data", "processed", "stroke_data_processed_test.csv"
        )

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        if "stroke" not in df_train.columns or "stroke" not in df_test.columns:
            st.error(
                "Error: La columna objetivo 'stroke' no se encontró en los archivos de datos."
            )
            return None, None, False

        return df_train, df_test, True
    except FileNotFoundError as e:
        return None, None, False


# --- SIDEBAR - PANEL DE CONFIGURACIÓN ---
st.sidebar.header("🔧 Configuración del Modelo")

# Mostrar información de debug (opcional, puedes comentarlo después)
with st.sidebar.expander("🔍 Debug Info"):
    st.write(f"**SCRIPT_DIR:** {SCRIPT_DIR}")
    st.write(f"**Models dir:** {os.path.join(SCRIPT_DIR, 'models')}")
    st.write(f"**Exists:** {os.path.exists(os.path.join(SCRIPT_DIR, 'models'))}")

available_models = check_available_models()

# Manejo de error si no hay modelos
if not available_models:
    st.sidebar.error("❌ No se encontraron modelos entrenados")
    st.sidebar.info(f"📂 Carpeta esperada: {os.path.join(SCRIPT_DIR, 'models')}")

    # Mostrar qué archivos hay
    models_dir = os.path.join(SCRIPT_DIR, "models")
    if os.path.exists(models_dir):
        files = os.listdir(models_dir)
        st.sidebar.write("**Archivos en models/:**")
        for f in files:
            st.sidebar.write(f"   • {f}")
    else:
        st.sidebar.write("⚠️ La carpeta 'models/' no existe")

    st.stop()

available_names = list(available_models.keys())
model_type = st.sidebar.selectbox("Selecciona el Modelo:", available_names, index=0)

st.sidebar.subheader("📊 Modelos Tabulares Disponibles")
for name, path in AVAILABLE_TABULAR_MODELS.items():
    if name in available_models:
        st.sidebar.success(f"✅ {name}")
    else:
        st.sidebar.error(f"❌ {name}")

st.sidebar.subheader("🖼️ Modelos de Imágenes Disponibles")
for name, path in AVAILABLE_IMAGE_MODELS.items():
    if os.path.exists(path):
        st.sidebar.success(f"✅ {name}")
    else:
        st.sidebar.error(f"❌ {name}")

model_path = available_models[model_type]
loaded_model, scaler, feature_names, model_loaded_successfully, model_framework = (
    load_pretrained_model(model_path)
)

if not model_loaded_successfully:
    st.sidebar.error(f"❌ Error al cargar: {os.path.basename(model_path)}")
    st.error(f"No se pudo cargar el modelo: {model_type}")
    st.stop()

st.sidebar.success(f"✅ Modelo cargado ({model_framework})")

# Guardar en session state
st.session_state.model_name = model_type
st.session_state.model_framework = model_framework

# Verificar estado del backend
st.sidebar.markdown("---")
st.sidebar.subheader("🔌 Estado del Backend")
backend_available = check_backend_status()

# CARGA DE DATOS (Opcional)
df_train, df_test, data_loaded = load_data()

if data_loaded:
    dataset_option = st.sidebar.selectbox(
        "Dataset para Evaluación:",
        ["Dataset de Entrenamiento", "Dataset de Test", "Ambos Datasets"],
        index=1,
    )
else:
    st.sidebar.warning("⚠️ Datasets no disponibles")
    st.sidebar.info("EDA y Evaluación deshabilitadas")
    dataset_option = None

# Configurar DataFrame
if data_loaded:
    if dataset_option == "Dataset de Entrenamiento":
        df = df_train.copy()
    elif dataset_option == "Dataset de Test":
        df = df_test.copy()
    else:
        df = pd.concat([df_train, df_test], ignore_index=True)
else:
    df = None

# --- INTERFAZ PRINCIPAL - SISTEMA DE PESTAÑAS ---

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📊 Exploración de Datos (EDA)",
        "🔮 Evaluación",
        "📈 Métricas del Modelo",
        "🎯 Predicción Individual",
        "🖼️ Predicción con Imágenes",
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

                # ===== DETERMINAR QUÉ FEATURES ESPERA EL MODELO =====
                # El modelo fue guardado con 25 features (todas las del CSV)
                # NO se aplicó drop_first=True correctamente
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
                    "work_type_children",
                    "smoking_status_formerly smoked",
                    "smoking_status_never smoked",
                    "smoking_status_smokes",
                    "age_group_19-35",
                    "age_group_36-50",
                    "age_group_51-65",
                    "age_group_65+",
                    "bmi_category_Normal",
                    "bmi_category_Overweight",
                    "bmi_category_Obese",
                    "glucose_category_Prediabetes",
                    "glucose_category_Diabetes",
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

                # Seleccionar solo las 25 columnas en el orden correcto
                X_processed = X[MODEL_EXPECTED_FEATURES].copy()

                st.success(f"✅ Dataset procesado: {X_processed.shape}")

                # NO aplicar scaler (datos ya preprocesados según el notebook)
                X_processed_array = X_processed.values

                st.info("ℹ️ No se aplica scaling (datos ya preprocesados)")

                # Predecir
                y_pred_class = loaded_model.predict(X_processed_array)

                # Intentar obtener probabilidades para AUC
                if hasattr(loaded_model, "predict_proba"):
                    y_pred_proba = loaded_model.predict_proba(X_processed_array)[:, 1]
                else:
                    y_pred_proba = None

                # ALMACENAMIENTO EN SESSION STATE
                st.session_state.model = loaded_model
                st.session_state.scaler = None
                st.session_state.feature_names = MODEL_EXPECTED_FEATURES
                st.session_state.X = X_processed_array
                st.session_state.y = y
                st.session_state.y_pred_class = y_pred_class
                st.session_state.y_pred_proba = y_pred_proba
                st.session_state.model_name = model_type

                st.success("✅ Modelo evaluado exitosamente!")

                # Mostrar preview de predicciones
                with st.expander("🔍 Preview de Predicciones"):
                    preview_df = pd.DataFrame(
                        {
                            "Real": y.head(10).values,
                            "Predicho": y_pred_class[:10],
                            "Probabilidad": (
                                y_pred_proba[:10]
                                if y_pred_proba is not None
                                else [None] * 10
                            ),
                        }
                    )
                    st.dataframe(preview_df)

            except Exception as e:
                st.error(f"❌ Error al hacer predicciones: {str(e)}")
                st.info("💡 Posibles causas:")
                st.write(
                    "- El modelo no es compatible con la estructura actual de datos"
                )
                st.write("- Verifica que las columnas coincidan con el entrenamiento")

                with st.expander("🔍 Debug Info"):
                    st.write(
                        "**Features esperadas:**",
                        (
                            MODEL_EXPECTED_FEATURES
                            if "MODEL_EXPECTED_FEATURES" in locals()
                            else "N/A"
                        ),
                    )
                    st.write(
                        "**Columnas del dataset:**",
                        X.columns.tolist() if "X" in locals() else "N/A",
                    )
                    st.write(
                        "**Shape procesado:**",
                        X_processed.shape if "X_processed" in locals() else "N/A",
                    )

                import traceback

                st.code(traceback.format_exc())

        # VISUALIZACIONES POST-EVALUACIÓN
        if "y_pred_class" in st.session_state:
            y = st.session_state.y
            y_pred_class = st.session_state.y_pred_class

            # Matriz de Confusión
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

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Edad del Paciente", min_value=1, max_value=120, value=50, step=1
            )
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

        # Umbral para la recomendación
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

                # 2. FEATURE ENGINEERING
                input_data["age_group"] = pd.cut(
                    input_data["age"],
                    bins=[0, 18, 35, 50, 65, 100],
                    labels=["0-18", "19-35", "36-50", "51-65", "65+"],
                )

                input_data["risk_factors"] = (
                    input_data["hypertension"] + input_data["heart_disease"]
                )

                input_data["bmi_category"] = pd.cut(
                    input_data["bmi"],
                    bins=[0, 18.5, 25, 30, 100],
                    labels=["Underweight", "Normal", "Overweight", "Obese"],
                )

                input_data["glucose_category"] = pd.cut(
                    input_data["avg_glucose_level"],
                    bins=[0, 100, 125, 300],
                    labels=["Normal", "Prediabetes", "Diabetes"],
                )

                input_data["age_risk_interaction"] = (
                    input_data["age"] * input_data["risk_factors"]
                )

                # 3. CODIFICACIÓN (Label Encoding)
                gender_map = {"Female": 0, "Male": 1, "Other": 2}
                input_data["gender_encoded"] = input_data["gender"].map(gender_map)

                married_map = {"No": 0, "Yes": 1}
                input_data["ever_married_encoded"] = input_data["ever_married"].map(
                    married_map
                )

                residence_map = {"Rural": 0, "Urban": 1}
                input_data["Residence_type_encoded"] = input_data["Residence_type"].map(
                    residence_map
                )

                # 4. ONE-HOT ENCODING (SIN drop_first para que coincida con el modelo)
                categorical_cols = [
                    "work_type",
                    "smoking_status",
                    "age_group",
                    "bmi_category",
                    "glucose_category",
                ]
                input_data = pd.get_dummies(
                    input_data, columns=categorical_cols, drop_first=False, dtype=int
                )

                # 5. ELIMINAR COLUMNAS ORIGINALES
                cols_to_drop = ["gender", "ever_married", "Residence_type"]
                input_data = input_data.drop(columns=cols_to_drop, errors="ignore")

                # 6. SELECCIONAR LAS 25 FEATURES DEL MODELO (todas las del CSV)
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
                    "work_type_children",
                    "smoking_status_formerly smoked",
                    "smoking_status_never smoked",
                    "smoking_status_smokes",
                    "age_group_19-35",
                    "age_group_36-50",
                    "age_group_51-65",
                    "age_group_65+",
                    "bmi_category_Normal",
                    "bmi_category_Overweight",
                    "bmi_category_Obese",
                    "glucose_category_Prediabetes",
                    "glucose_category_Diabetes",
                ]

                # Crear columnas faltantes con 0
                for col in MODEL_EXPECTED_FEATURES:
                    if col not in input_data.columns:
                        input_data[col] = 0

                # Reordenar a las 25 columnas en el orden correcto
                input_data = input_data[MODEL_EXPECTED_FEATURES]

                # 7. NO APLICAR SCALER (datos ya preprocesados)
                input_data_scaled = input_data.values

                # 8. HACER PREDICCIÓN
                prediction_class = st.session_state.model.predict(input_data_scaled)[0]

                if hasattr(st.session_state.model, "predict_proba"):
                    prediction_proba = st.session_state.model.predict_proba(
                        input_data_scaled
                    )[0, 1]
                else:
                    prediction_proba = None

                # 9. MOSTRAR RESULTADOS
                st.subheader("✅ Resultado de la Predicción")

                # Determinar nivel de riesgo basado en probabilidad
                if prediction_proba is not None:
                    # Niveles de riesgo
                    if prediction_proba < 0.2:
                        risk_level = "BAJO"
                        result_text = "🟢 **RIESGO BAJO DE ICTUS**"
                        result_color = "green"
                        risk_emoji = "🟢"
                    elif prediction_proba < 0.5:
                        risk_level = "MODERADO"
                        result_text = "🟡 **RIESGO MODERADO DE ICTUS**"
                        result_color = "orange"
                        risk_emoji = "🟡"
                    else:
                        risk_level = "ALTO"
                        result_text = "🔴 **RIESGO ALTO DE ICTUS**"
                        result_color = "red"
                        risk_emoji = "🔴"

                    # Mostrar probabilidad
                    st.metric("Probabilidad de Ictus", f"{prediction_proba:.1%}")

                    # Barra visual de probabilidad
                    progress_color = (
                        "🔴"
                        if prediction_proba >= 0.5
                        else "🟡" if prediction_proba >= 0.2 else "🟢"
                    )
                    st.progress(float(min(prediction_proba, 1.0)))

                else:
                    # Fallback si no hay probabilidad
                    if prediction_class == 1:
                        risk_level = "ALTO"
                        result_text = "🔴 **RIESGO ALTO DE ICTUS**"
                        result_color = "red"
                        risk_emoji = "🔴"
                    else:
                        risk_level = "BAJO"
                        result_text = "🟢 **RIESGO BAJO DE ICTUS**"
                        result_color = "green"
                        risk_emoji = "🟢"

                st.markdown(
                    f"**Clasificación del Modelo:** <span style='color:{result_color}; font-size: 24px'>{result_text}</span>",
                    unsafe_allow_html=True,
                )

                # 10. RECOMENDACIÓN MÉDICA
                st.subheader("🏥 Recomendación Médica")

                if risk_level == "ALTO":
                    st.error(
                        f"""
                    **⚠️ ¡ATENCIÓN - RIESGO ALTO!**
                    
                    Probabilidad de ictus: **{prediction_proba:.1%}**
                    
                    El modelo indica un riesgo significativo de ictus.
                    """
                    )
                    st.markdown(
                        """
                    **Recomendaciones URGENTES:**
                    * 🏥 **Visitar a un Especialista:** Consulta inmediata con neurólogo o cardiólogo
                    * 🔬 **Pruebas Diagnósticas:** TC, RM cerebral o ecocardiogramas según indicación
                    * 💊 **Control Médico Estricto:** Monitoreo de presión arterial y glucosa
                    * 🚭 **Modificación Inmediata:** Dejar de fumar, control de peso, dieta cardiosaludable
                    * ⚡ **Signos de Alarma:** Ante cualquier síntoma FAST, llamar emergencias inmediatamente
                    """
                    )

                elif risk_level == "MODERADO":
                    st.warning(
                        f"""
                    **⚠️ RIESGO MODERADO**
                    
                    Probabilidad de ictus: **{prediction_proba:.1%}**
                    
                    El modelo detecta factores de riesgo que requieren atención.
                    """
                    )
                    st.markdown(
                        """
                    **Recomendaciones:**
                    * 👨‍⚕️ **Evaluación Médica:** Consulta con médico de cabecera en las próximas semanas
                    * 📋 **Control de Factores:** Monitorear presión arterial, glucosa, colesterol
                    * 🏃 **Prevención Activa:** Ejercicio regular (30min/día), dieta mediterránea
                    * 🚭 **Reducir Riesgos:** Si fumas, considera dejarlo; reduce alcohol
                    * 👀 **Vigilancia:** Atento a síntomas como mareos, pérdida de fuerza o habla confusa
                    """
                    )

                else:  # BAJO
                    st.success(
                        f"""
                    **✅ RIESGO BAJO**
                    
                    Probabilidad de ictus: **{prediction_proba:.1%}**
                    
                    El modelo indica bajo riesgo de ictus.
                    """
                    )
                    st.markdown(
                        """
                    **Recomendaciones de Prevención:**
                    * 🏥 **Chequeos Rutinarios:** Continuar con revisiones médicas anuales
                    * 💪 **Estilo de Vida Saludable:** Mantener dieta equilibrada y ejercicio regular
                    * 🚭 **Prevención:** Evitar tabaquismo y consumo excesivo de alcohol
                    * 📊 **Monitoreo:** Control periódico de presión arterial y glucosa
                    * 🧠 **Educación:** Conocer síntomas FAST (Face, Arm, Speech, Time)
                    """
                    )

                st.info(f"🤖 **Modelo utilizado:** {st.session_state.model_name}")

                with st.expander("📋 Ver Datos Procesados"):
                    st.write(f"**Shape:** {input_data.shape}")
                    st.dataframe(input_data)

                # Guardar en backend si está disponible
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
                    "💡 Verifica que los datos de entrada coincidan con la estructura del modelo"
                )

                import traceback

                st.code(traceback.format_exc())

    else:
        st.info(
            "👆 Espera a que el modelo se cargue o evalúa el modelo primero en la pestaña 'Evaluación'"
        )
# ----------------------------------------------------
# TAB 5: PREDICCIÓN CON IMÁGENES MÉDICAS
# ----------------------------------------------------
with tab5:
    st.header("🖼️ Predicción de Ictus con Imágenes Médicas")

    # Cargar modelo de imágenes
    @st.cache_resource
    def load_image_model():
        """Carga el modelo de clasificación de imágenes"""
        try:
            import tensorflow as tf

            # Buscar en los modelos de imágenes disponibles
            if not AVAILABLE_IMAGE_MODELS:
                return None, None, False

            # Tomar el primer modelo de imágenes disponible
            model_name = list(AVAILABLE_IMAGE_MODELS.keys())[0]
            model_path = AVAILABLE_IMAGE_MODELS[model_name]
            metadata_path = model_path.replace(".h5", "_metadata.json").replace(
                ".keras", "_metadata.json"
            )

            if not os.path.exists(model_path):
                return None, None, False

            model = tf.keras.models.load_model(model_path)

            # Cargar metadata
            metadata = None
            if os.path.exists(metadata_path):
                import json

                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

            return model, metadata, True
        except Exception as e:
            st.error(f"Error al cargar modelo de imágenes: {str(e)}")
            import traceback

            st.code(traceback.format_exc())
            return None, None, False

    image_model, image_metadata, image_model_loaded = load_image_model()

    if not image_model_loaded:
        st.warning("⚠️ Modelo de imágenes no disponible")
        st.info("📁 Archivo necesario: `models/stroke_image_model.h5`")
        st.markdown(
            "💡 Entrena el modelo usando el notebook de imágenes y guárdalo en la carpeta `models/`"
        )
    else:
        st.success("✅ Modelo de imágenes cargado correctamente")

        # Mostrar información del modelo
        if image_metadata:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Recall en Test", f"{image_metadata.get('test_recall', 0):.2%}"
                )
            with col2:
                st.metric(
                    "Precision en Test",
                    f"{image_metadata.get('test_precision', 0):.2%}",
                )
            with col3:
                input_shape = image_metadata.get("input_shape", [224, 224, 1])
                st.metric("Input Shape", f"{input_shape[0]}x{input_shape[1]}")

        st.markdown("---")
        st.subheader("📤 Cargar Imagen Médica")

        # Upload de imagen
        uploaded_file = st.file_uploader(
            "Selecciona una imagen de resonancia magnética cerebral (MRI)",
            type=["png", "jpg", "jpeg", "dcm"],
            help="Formatos soportados: PNG, JPG, JPEG, DICOM",
        )

        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("🖼️ Imagen Original")

                # Leer y mostrar imagen
                from PIL import Image
                import io

                # Leer imagen
                image_bytes = uploaded_file.read()
                image = Image.open(io.BytesIO(image_bytes))

                # Mostrar imagen original
                st.image(image, caption="Imagen cargada", use_container_width=True)

                # Info de la imagen
                st.info(f"📐 Dimensiones: {image.size[0]} x {image.size[1]}")

            with col2:
                st.subheader("🔬 Procesamiento")

                with st.spinner("Procesando imagen..."):
                    try:
                        # Obtener shape esperado
                        if image_metadata:
                            target_size = tuple(image_metadata["input_shape"][:2])
                        else:
                            target_size = (224, 224)

                        # Convertir a escala de grises si es necesario
                        if image.mode != "L":
                            image_gray = image.convert("L")
                        else:
                            image_gray = image

                        # Redimensionar
                        image_resized = image_gray.resize(
                            target_size, Image.Resampling.LANCZOS
                        )

                        # Mostrar imagen procesada
                        st.image(
                            image_resized,
                            caption=f"Imagen procesada ({target_size[0]}x{target_size[1]})",
                            use_container_width=True,
                        )

                        # Convertir a array numpy
                        img_array = np.array(image_resized)

                        # Normalizar [0, 1]
                        img_array = img_array / 255.0

                        # Añadir dimensiones: (1, H, W, 1)
                        img_array = img_array[np.newaxis, ..., np.newaxis]

                        st.success(f"✅ Imagen procesada: {img_array.shape}")

                    except Exception as e:
                        st.error(f"Error al procesar imagen: {str(e)}")
                        img_array = None

            # Botón de predicción
            st.markdown("---")

            if img_array is not None and st.button(
                "🔮 **Analizar Imagen**", type="primary", use_container_width=True
            ):
                with st.spinner("Analizando imagen con IA..."):
                    try:
                        # Hacer predicción
                        prediction_proba = image_model.predict(img_array, verbose=0)[0][
                            0
                        ]

                        # Threshold
                        threshold = (
                            image_metadata.get("threshold", 0.5)
                            if image_metadata
                            else 0.5
                        )
                        prediction_class = 1 if prediction_proba >= threshold else 0

                        # Mostrar resultados
                        st.markdown("---")
                        st.subheader("✅ Resultado del Análisis")

                        # Visualización del resultado
                        col1, col2, col3 = st.columns([1, 2, 1])

                        with col2:
                            # Medidor de probabilidad
                            fig = go.Figure(
                                go.Indicator(
                                    mode="gauge+number+delta",
                                    value=prediction_proba * 100,
                                    domain={"x": [0, 1], "y": [0, 1]},
                                    title={"text": "Probabilidad de Ictus (%)"},
                                    delta={"reference": threshold * 100},
                                    gauge={
                                        "axis": {"range": [None, 100]},
                                        "bar": {
                                            "color": (
                                                "darkred"
                                                if prediction_class == 1
                                                else "green"
                                            )
                                        },
                                        "steps": [
                                            {
                                                "range": [0, threshold * 100],
                                                "color": "lightgreen",
                                            },
                                            {
                                                "range": [threshold * 100, 100],
                                                "color": "lightcoral",
                                            },
                                        ],
                                        "threshold": {
                                            "line": {"color": "red", "width": 4},
                                            "thickness": 0.75,
                                            "value": threshold * 100,
                                        },
                                    },
                                )
                            )
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)

                        # Resultado textual
                        if prediction_class == 1:
                            st.error(
                                f"""
                            ### 🔴 DETECCIÓN POSITIVA DE ICTUS
                            
                            **Probabilidad:** {prediction_proba:.2%}
                            
                            El modelo detecta patrones compatibles con ictus en la imagen.
                            """
                            )

                            st.warning(
                                """
                            **⚠️ RECOMENDACIONES URGENTES:**
                            - ✅ Derivar inmediatamente a neurólogo
                            - ✅ Realizar estudios complementarios (TC, RM adicionales)
                            - ✅ Considerar tratamiento de emergencia
                            - ✅ Monitoreo continuo del paciente
                            """
                            )
                        else:
                            st.success(
                                f"""
                            ### 🟢 NO SE DETECTA ICTUS
                            
                            **Probabilidad:** {prediction_proba:.2%}
                            
                            El modelo no detecta patrones significativos de ictus en la imagen.
                            """
                            )

                            st.info(
                                """
                            **📋 RECOMENDACIONES:**
                            - ✅ Continuar con evaluación clínica estándar
                            - ✅ Considerar otros diagnósticos diferenciales
                            - ✅ Monitoreo de síntomas del paciente
                            - ⚠️ Este resultado no descarta completamente el diagnóstico
                            """
                            )

                        # Disclaimer médico
                        st.markdown("---")
                        st.warning(
                            """
                        **⚠️ ADVERTENCIA MÉDICA IMPORTANTE**
                        
                        Esta herramienta es de **apoyo al diagnóstico** y NO reemplaza el criterio médico profesional.
                        - El resultado debe ser validado por un especialista
                        - Se requiere evaluación clínica completa del paciente
                        - Considerar síntomas, historia clínica y otros estudios
                        - En caso de duda, siempre priorizar la evaluación humana
                        """
                        )

                        # Información técnica (expandible)
                        with st.expander("🔧 Información Técnica del Modelo"):
                            if image_metadata:
                                st.json(image_metadata)
                            st.write(f"**Shape de entrada:** {img_array.shape}")
                            st.write(f"**Threshold usado:** {threshold}")
                            st.write(f"**Probabilidad raw:** {prediction_proba:.6f}")

                    except Exception as e:
                        st.error(f"Error al hacer predicción: {str(e)}")
                        st.exception(e)
        else:
            st.info("👆 Carga una imagen médica para comenzar el análisis")

            # Mostrar ejemplos de uso
            with st.expander("💡 Guía de Uso"):
                st.markdown(
                    """
                ### Cómo usar esta herramienta:
                
                1. **Preparar la imagen:**
                   - Formato: PNG, JPG o JPEG
                   - Contenido: Resonancia magnética cerebral (MRI)
                   - Calidad: Preferiblemente alta resolución
                
                2. **Cargar la imagen:**
                   - Click en "Browse files" o arrastra la imagen
                   - Espera a que se procese
                
                3. **Analizar:**
                   - Click en "Analizar Imagen"
                   - Revisa el resultado y las recomendaciones
                
                4. **Interpretar:**
                   - Verde (🟢): No se detecta ictus
                   - Rojo (🔴): Posible ictus detectado
                   - Siempre consultar con un especialista
                
                ### Limitaciones:
                - El modelo fue entrenado con un dataset específico
                - La precisión puede variar según la calidad de la imagen
                - No reemplaza el diagnóstico médico profesional
                """
                )
# ----------------------------------------------------
# TAB 6: HISTORIAL DE PREDICCIONES
# ----------------------------------------------------
with tab6:
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
    f"**Stroke Risk Predictor** - Modelos Tabulares: {len(available_models)} | Modelos Imágenes: {len(AVAILABLE_IMAGE_MODELS)}"
)
st.markdown("Desarrollado con ❤️ usando Streamlit")
