import streamlit as st
import pandas as pd
import sys
import argparse
import pickle
from pathlib import Path
# Importaciones de tu back-end (asumiendo que están en src/)
# Estas funciones DEBEN existir para que el modelo funcione
#from src.models.predict_model import make_prediction 
#from src.data.feature_engineering import preprocess_data # Solo si no está dentro de make_prediction

# --- Configuración de rutas (Ajustar según necesidad) ---
# El script ahora está en la raíz, por lo que las rutas deben ser relativas desde aquí.
ARTIFACTS_PATH = Path("artifacts")
MODEL_FILENAME = "best_model.pkl"

# -------------------------------------------------------------
# 1. LÓGICA DE LA APLICACIÓN DE LÍNEA DE COMANDOS (CLI)
# -------------------------------------------------------------

def cli_interface():
    """
    Función que maneja la interfaz de línea de comandos (Antiguo app_cli.py).
    Solicita datos y muestra la predicción en la terminal.
    """
    print("\n--- EJECUTANDO INTERFAZ DE LÍNEA DE COMANDOS ---")
    
    # 1. Cargar el modelo
    model = load_model()
    
    # 2. Recoger datos del paciente
    patient_df = get_cli_patient_data()
    
    # Si la recolección falló (e.g., ValueError), ya se habrá salido
    if patient_df is None:
        return 

    # 3. Realizar la predicción
    try:
        prediction, probability = make_prediction(model, patient_df)
    except Exception as e:
        print(f"❌ ERROR al realizar la predicción: {e}")
        print("Asegúrate de que los datos de entrada coincidan con el entrenamiento.")
        return

    # 4. Mostrar el resultado
    print("\n--- RESULTADO DE LA PREDICCIÓN ---")
    if prediction[0] == 1:
        print(f"🔴 ALERTA: El paciente presenta ALTO RIESGO de sufrir un ictus.")
        print(f"Probabilidad de ictus: {probability[0]:.2%}")
    else:
        print(f"🟢 BAJO RIESGO: El paciente presenta bajo riesgo de sufrir un ictus.")
        print(f"Probabilidad de ictus: {probability[0]:.2%}")
        
    print("\nRecomendación: Consulte a un doctor para un diagnóstico definitivo.")

def get_cli_patient_data():
    """Solicita los datos del paciente por línea de comandos."""
    try:
        print("\nIngrese los datos requeridos (ejemplos):")
        gender = input("Género (Male/Female/Other): ").strip()
        age = float(input("Edad: "))
        hypertension = int(input("Hipertensión (0: No, 1: Sí): "))
        heart_disease = int(input("Enfermedad cardíaca (0: No, 1: Sí): "))
        avg_glucose_level = float(input("Nivel promedio de glucosa: "))
        bmi = float(input("Índice de Masa Corporal (IMC): "))
        
        # Crear un DataFrame con los datos (Solo un ejemplo, añade las demás features)
        data = {
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'avg_glucose_level': [avg_glucose_level],
            'bmi': [bmi]
        }
        return pd.DataFrame(data)
        
    except ValueError:
        print("\nERROR: Asegúrate de ingresar el tipo de dato correcto (número para edad, etc.).")
        return None
    except EOFError:
        print("\nEntrada cancelada.")
        return None


# -------------------------------------------------------------
# 2. LÓGICA DE LA APLICACIÓN WEB (Streamlit)
# -------------------------------------------------------------

def web_interface():
    """
    Función que maneja la interfaz web con Streamlit (Antiguo app_streamlit.py).
    """
    st.set_page_config(
        page_title="Riesgo de Ictus | Hospital F5",
        layout="wide"
    )

    st.title("🧠 Evaluación de Riesgo de Ictus (Stroke)")
    st.markdown("Herramienta de Criba Previa con Inteligencia Artificial.")
    st.markdown("---")
    
    model = load_model() # Cargar el modelo una vez al inicio
    
    st.sidebar.header("Datos del Paciente")
    
    # --- Widgets de Entrada de Datos ---
    gender = st.sidebar.selectbox("Género", ('Female', 'Male', 'Other'))
    hypertension = st.sidebar.radio("Hipertensión", (0, 1), format_func=lambda x: 'Sí' if x == 1 else 'No')
    heart_disease = st.sidebar.radio("Enfermedad Cardíaca", (0, 1), format_func=lambda x: 'Sí' if x == 1 else 'No')
    age = st.sidebar.slider("Edad", min_value=0.0, max_value=100.0, value=45.0, step=0.1)
    avg_glucose_level = st.sidebar.number_input("Nivel Promedio de Glucosa (mg/dL)", 
                                                 min_value=50.0, max_value=350.0, value=95.0, step=0.1)
    bmi = st.sidebar.number_input("Índice de Masa Corporal (IMC)", 
                                  min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    # ⚠️ AÑADIR AQUÍ EL RESTO DE LAS 11 CARACTERÍSTICAS

    input_data = {
        'gender': [gender], 'age': [age], 'hypertension': [hypertension], 
        'heart_disease': [heart_disease], 'avg_glucose_level': [avg_glucose_level], 'bmi': [bmi]
    }
    input_df = pd.DataFrame(input_data)
    
    st.markdown("---")
    if st.button("Evaluar Riesgo de Ictus", type="primary"):
        with st.spinner('Evaluando datos...'):
            try:
                # Realizar la predicción real
                prediction, probability = make_prediction(model, input_df)
                
                # Mostrar resultados
                st.subheader("Resultado de la Evaluación")
                if prediction[0] == 1:
                    st.error("🔴 ALERTA: Alto Riesgo de Ictus.")
                else:
                    st.success("🟢 BAJO RIESGO de Ictus.")
                    
                st.markdown(f"**Probabilidad Estimada de Ictus:** `{probability[0]:.2%}`")
            
            except Exception as e:
                st.error(f"Error al procesar la predicción. Asegúrate de que el modelo esté cargado y las características coincidan. Error: {e}")

        st.info("⚠️ **Descargo de Responsabilidad:** Esta es una criba previa de IA.")

# -------------------------------------------------------------
# 3. LÓGICA COMÚN Y PUNTO DE ENTRADA
# -------------------------------------------------------------

@st.cache_resource # Carga el modelo una sola vez para Streamlit
def load_model():
    """Carga el modelo serializado desde la carpeta artifacts."""
    model_path = ARTIFACTS_PATH / MODEL_FILENAME
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        error_msg = f"❌ ERROR: Modelo no encontrado en {model_path}. Por favor, entrene el modelo primero."
        if 'streamlit' in sys.modules:
            st.error(error_msg)
        else:
            print(error_msg)
        # Esto es un error fatal para la aplicación
        sys.exit(1)


if __name__ == "__main__":
    # Si el script se ejecuta directamente con 'python app.py'
    if 'streamlit' not in sys.modules:
        parser = argparse.ArgumentParser(description="Ejecuta la interfaz web o CLI del modelo de ictus.")
        parser.add_argument('--cli', action='store_true', 
                            help='Ejecuta la aplicación en modo Línea de Comandos (CLI).')
        args = parser.parse_args()
        
        if args.cli:
            cli_interface()
        else:
            # Opción por defecto: Instrucción para ejecutar la web
            print("--- INICIADOR DE LA APLICACIÓN ---")
            print("Para ejecutar la aplicación web (Streamlit):")
            print("streamlit run app.py")
            print("\nPara ejecutar la aplicación por línea de comandos (CLI):")
            print("python app.py --cli")
            
    # Si el script es invocado por Streamlit ('streamlit run app.py')
    else:
        web_interface()