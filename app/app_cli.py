# Versión de línea de comandos

import pandas as pd
import sys
import pickle
from pathlib import Path
from src.models.predict_model import make_prediction

# --- Configuración de rutas (Ajustar según necesidad) ---
# Se asume que el script se ejecuta desde la raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_PATH = ROOT_DIR / "artifacts"
MODEL_FILENAME = "best_model.pkl"

def get_patient_data():
    """
    Solicita los datos del paciente por línea de comandos.
    
    NOTA: Las variables a pedir deben coincidir con las características
    usadas para entrenar el modelo (incluyendo codificación/escalado).
    Este es solo un ejemplo de cómo pedirlas.
    """
    print("\n--- INGRESO DE DATOS DEL PACIENTE ---")
    
    # ⚠️ EJEMPLO: Se deben pedir todas las características requeridas por el modelo
    # Se utiliza input() para simular la línea de comandos
    try:
        gender = input("Género (Male/Female/Other): ").strip()
        age = float(input("Edad: "))
        hypertension = int(input("Hipertensión (0: No, 1: Sí): "))
        heart_disease = int(input("Enfermedad cardíaca (0: No, 1: Sí): "))
        avg_glucose_level = float(input("Nivel promedio de glucosa: "))
        bmi = float(input("Índice de Masa Corporal (IMC): "))
        
        # Crear un DataFrame con los datos (el modelo espera este formato)
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
        print("\nERROR: Asegúrate de ingresar el tipo de dato correcto (número para edad, glucosa, etc.).")
        sys.exit(1)


def load_model(path: Path):
    """Carga el modelo serializado desde el disco."""
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Modelo cargado exitosamente desde: {path.name}")
        return model
    except FileNotFoundError:
        print(f"❌ ERROR: No se encontró el modelo en la ruta esperada: {path}")
        print("Asegúrate de haber entrenado y guardado el modelo correctamente.")
        sys.exit(1)


def main():
    """Función principal de la aplicación de línea de comandos."""
    
    # 1. Cargar el modelo
    model_path = ARTIFACTS_PATH / MODEL_FILENAME
    model = load_model(model_path)
    
    # 2. Recoger datos del paciente
    patient_df = get_patient_data()
    
    # 3. Realizar la predicción
    # Se llama a la función de predicción que debe manejar el preprocesamiento
    # y aplicar el modelo
    prediction, probability = make_prediction(model, patient_df)
    
    # 4. Mostrar el resultado
    print("\n--- RESULTADO DE LA PREDICCIÓN ---")
    if prediction[0] == 1:
        print(f"🔴 ALERTA: El paciente presenta ALTO RIESGO de sufrir un ictus.")
        print(f"Probabilidad de ictus: {probability[0]:.2%}")
    else:
        print(f"🟢 BAJO RIESGO: El paciente presenta bajo riesgo de sufrir un ictus.")
        print(f"Probabilidad de ictus: {probability[0]:.2%}")
        
    print("\nRecomendación: Esta es una criba previa. Consulte a un doctor para un diagnóstico definitivo.")


if __name__ == "__main__":
    main() 