"""
Aplicación CLI para predicción de riesgo de ictus.
Hospital F5 - Herramienta de línea de comandos.
"""

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


def print_header():
    """Imprime el encabezado de la aplicación."""
    print("\n" + "=" * 60)
    print("🧠 PREDICCIÓN DE RIESGO DE ICTUS - Hospital F5")
    print("   Aplicación de Línea de Comandos")
    print("=" * 60)


def get_patient_data():
    """
    Solicita los datos del paciente por línea de comandos.

    Returns:
        DataFrame con los datos del paciente
    """
    print("\n📋 INGRESO DE DATOS DEL PACIENTE")
    print("-" * 60)

    try:
        # Datos demográficos
        print("\n--- Datos Demográficos ---")
        gender = input("Género (Male/Female/Other): ").strip()
        age = float(input("Edad: "))
        ever_married = input("¿Alguna vez casado? (Yes/No): ").strip()

        # Datos de salud
        print("\n--- Datos de Salud ---")
        hypertension = int(input("Hipertensión (0: No, 1: Sí): "))
        heart_disease = int(input("Enfermedad cardíaca (0: No, 1: Sí): "))
        avg_glucose_level = float(input("Nivel promedio de glucosa (mg/dL): "))
        bmi = float(input("Índice de Masa Corporal (IMC): "))

        # Estilo de vida
        print("\n--- Estilo de Vida ---")
        print(
            "Opciones de trabajo: Private, Self-employed, Govt_job, children, Never_worked"
        )
        work_type = input("Tipo de trabajo: ").strip()

        print("Opciones de residencia: Urban, Rural")
        residence_type = input("Tipo de residencia: ").strip()

        print("Opciones de fumador: never smoked, formerly smoked, smokes, Unknown")
        smoking_status = input("Estado de fumador: ").strip()

        # Crear DataFrame con los datos
        data = {
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

        return pd.DataFrame([data])

    except ValueError as e:
        print(f"\n❌ ERROR: Tipo de dato incorrecto. {e}")
        print("   Asegúrate de ingresar números para edad, glucosa, IMC, etc.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Operación cancelada por el usuario.")
        sys.exit(0)


def display_result(prediction, probability, risk_level, patient_data):
    """
    Muestra el resultado de la predicción de forma formateada.

    Args:
        prediction: Predicción (0 o 1)
        probability: Probabilidad de ictus
        risk_level: Nivel de riesgo (BAJO, MODERADO, ALTO)
        patient_data: Datos del paciente
    """
    print("\n" + "=" * 60)
    print("🎯 RESULTADO DE LA EVALUACIÓN")
    print("=" * 60)

    # Determinar color y mensaje
    if prediction[0] == 1:
        icon = "🔴"
        message = "ALERTA: Alto Riesgo de Ictus"
        color = "CRÍTICO"
    else:
        icon = "🟢"
        message = "Bajo Riesgo de Ictus"
        color = "NORMAL"

    print(f"\n{icon} {message}")
    print(f"\n📊 Detalles:")
    print(f"   Predicción: {'ICTUS' if prediction[0] == 1 else 'SIN ICTUS'}")
    print(f"   Probabilidad: {probability[0]:.2%}")
    print(f"   Nivel de Riesgo: {risk_level}")

    # Resumen del paciente
    print(f"\n👤 Resumen del Paciente:")
    print(f"   Edad: {patient_data['age'].iloc[0]:.0f} años")
    print(f"   Género: {patient_data['gender'].iloc[0]}")
    print(f"   IMC: {patient_data['bmi'].iloc[0]:.1f}")
    print(f"   Glucosa: {patient_data['avg_glucose_level'].iloc[0]:.1f} mg/dL")
    print(
        f"   Hipertensión: {'Sí' if patient_data['hypertension'].iloc[0] == 1 else 'No'}"
    )
    print(
        f"   Enfermedad Cardíaca: {'Sí' if patient_data['heart_disease'].iloc[0] == 1 else 'No'}"
    )

    # Interpretación
    print(f"\n💡 Interpretación:")
    if risk_level == "ALTO":
        print("   • El paciente presenta múltiples factores de riesgo.")
        print("   • Se recomienda EVALUACIÓN MÉDICA INMEDIATA.")
        print(
            "   • Considerar estudios complementarios (neuroimagen, doppler carotídeo)."
        )
    elif risk_level == "MODERADO":
        print("   • El paciente presenta algunos factores de riesgo.")
        print("   • Se recomienda seguimiento médico regular.")
        print("   • Implementar medidas preventivas (dieta, ejercicio).")
    else:
        print("   • El paciente presenta bajo riesgo de ictus.")
        print("   • Mantener hábitos de vida saludables.")
        print("   • Realizar chequeos médicos periódicos.")

    print(f"\n⚠️ DISCLAIMER:")
    print("   Esta es una herramienta de CRIBA PREVIA basada en IA.")
    print("   NO reemplaza el diagnóstico médico profesional.")
    print("   Consulte SIEMPRE a un profesional de la salud.")
    print("=" * 60)


def main():
    """Función principal de la aplicación CLI."""

    try:
        # 1. Mostrar encabezado
        print_header()

        # 2. Cargar modelo
        print("\n🔄 Cargando modelo de predicción...")
        model_path = (
            Path(__file__).parent.parent / "artifacts" / "best_model_balanced.pkl"
        )

        try:
            artifacts = load_model_artifacts(str(model_path))
            print(f"✅ Modelo cargado correctamente")
            print(f"   Estrategia: {artifacts.get('strategy', 'N/A')}")
            print(f"   Threshold óptimo: {artifacts['optimal_threshold']:.2f}")
        except FileNotFoundError:
            print("\n❌ ERROR: No se encontró el modelo entrenado.")
            print("   Por favor, ejecuta primero los notebooks:")
            print("   1. notebooks/Advanced_Balancing.ipynb")
            print("   2. notebooks/Threshold_Optimization_Balanced.ipynb")
            sys.exit(1)

        # 3. Solicitar datos del paciente
        patient_data = get_patient_data()

        # 4. Validar datos
        print("\n🔍 Validando datos...")
        is_valid, error_msg = validate_input_data(patient_data)
        if not is_valid:
            print(f"❌ ERROR en validación: {error_msg}")
            sys.exit(1)
        print("✅ Datos validados correctamente")

        # 5. Realizar predicción
        print("\n🤖 Realizando predicción...")
        prediction, probability, risk_level = make_prediction(artifacts, patient_data)

        # 6. Mostrar características importantes
        print("\n📊 Analizando factores de riesgo...")
        contributions = get_feature_contributions(artifacts, patient_data, top_n=5)
        print("\nTop 5 factores más influyentes:")
        for idx, row in contributions.iterrows():
            print(f"   • {row['feature']}: {row['value']:.4f}")

        # 7. Mostrar resultado
        display_result(prediction, probability, risk_level, patient_data)

        # 8. Opción para nueva predicción
        print("\n¿Desea realizar otra predicción? (s/n): ", end="")
        response = input().strip().lower()
        if response in ["s", "si", "sí", "yes", "y"]:
            print("\n" + "-" * 60)
            main()  # Recursión para nueva predicción
        else:
            print("\n👋 Gracias por usar el sistema de predicción de ictus.")
            print("   Hospital F5 - Cuidando tu salud con Inteligencia Artificial")

    except KeyboardInterrupt:
        print("\n\n⚠️ Operación cancelada por el usuario.")
        print("👋 Hasta pronto!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")
        print("   Por favor, contacte al administrador del sistema.")
        sys.exit(1)


if __name__ == "__main__":
    main()
