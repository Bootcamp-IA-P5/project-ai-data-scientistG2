"""
Cliente HTTP para comunicarse con el Backend API (FastAPI)
Este módulo proporciona funciones para interactuar con los endpoints del backend.
"""

import requests
from typing import Dict, List, Optional
import streamlit as st

# Configuración de la URL base del backend
# Puedes cambiar esto según tu configuración de despliegue
BASE_URL = "http://localhost:8000"


class APIClient:
    """Cliente para interactuar con la API de predicciones de ictus."""

    def __init__(self, base_url: str = BASE_URL):
        """
        Inicializa el cliente API.

        Args:
            base_url: URL base del backend FastAPI
        """
        self.base_url = base_url.rstrip("/")

    def health_check(self) -> Dict:
        """
        Verifica si el backend está disponible.

        Returns:
            Respuesta del endpoint raíz o None si hay error
        """
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.warning(f"⚠️ Backend no disponible: {str(e)}")
            return None

    def save_prediction(
        self,
        input_data: str,
        prediction_result: str,
        confidence: Optional[float] = None,
    ) -> Optional[Dict]:
        """
        Guarda una predicción en el historial del backend.

        Args:
            input_data: Datos de entrada (JSON string o texto descriptivo)
            prediction_result: Resultado de la predicción (ej: "0" o "1")
            confidence: Nivel de confianza/probabilidad (0.0 - 1.0)

        Returns:
            Diccionario con la predicción guardada o None si hay error
        """
        try:
            payload = {
                "input_data": str(input_data),
                "prediction_result": str(prediction_result),
                "confidence": float(confidence) if confidence is not None else None,
            }

            response = requests.post(
                f"{self.base_url}/predictions/", json=payload, timeout=10
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error al guardar predicción: {str(e)}")
            return None

    def get_predictions(self, skip: int = 0, limit: int = 100) -> Optional[List[Dict]]:
        """
        Obtiene el historial de predicciones desde el backend.

        Args:
            skip: Número de registros a omitir (para paginación)
            limit: Número máximo de registros a devolver

        Returns:
            Lista de predicciones o None si hay error
        """
        try:
            response = requests.get(
                f"{self.base_url}/predictions/",
                params={"skip": skip, "limit": limit},
                timeout=10,
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error al obtener predicciones: {str(e)}")
            return None

    def get_prediction_by_id(self, prediction_id: int) -> Optional[Dict]:
        """
        Obtiene una predicción específica por su ID.

        Args:
            prediction_id: ID de la predicción

        Returns:
            Diccionario con la predicción o None si hay error
        """
        try:
            response = requests.get(
                f"{self.base_url}/predictions/{prediction_id}", timeout=10
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            if "404" in str(e):
                st.warning(f"⚠️ Predicción con ID {prediction_id} no encontrada")
            else:
                st.error(f"❌ Error al obtener predicción: {str(e)}")
            return None

    def is_backend_available(self) -> bool:
        """
        Verifica rápidamente si el backend está disponible.

        Returns:
            True si el backend responde, False en caso contrario
        """
        try:
            response = requests.get(f"{self.base_url}/", timeout=3)
            return response.status_code == 200
        except:
            return False


# Instancia global del cliente (puedes configurar la URL desde variables de entorno)
api_client = APIClient(base_url=BASE_URL)


# --- Funciones de conveniencia para usar directamente en la app ---


def save_prediction_to_backend(
    input_data: Dict, prediction_class: int, prediction_proba: Optional[float] = None
) -> bool:
    """
    Función de conveniencia para guardar una predicción.

    Args:
        input_data: Diccionario con los datos de entrada del paciente
        prediction_class: Clase predicha (0 o 1)
        prediction_proba: Probabilidad de la clase positiva

    Returns:
        True si se guardó exitosamente, False en caso contrario
    """
    import json

    # Convertir el diccionario de entrada a JSON string
    input_json = json.dumps(input_data, ensure_ascii=False)

    # Determinar el texto del resultado
    result_text = (
        "RIESGO ALTO DE ICTUS" if prediction_class == 1 else "RIESGO BAJO/MODERADO"
    )

    # Guardar en el backend
    saved_prediction = api_client.save_prediction(
        input_data=input_json,
        prediction_result=result_text,
        confidence=prediction_proba,
    )

    return saved_prediction is not None


def get_prediction_history(limit: int = 50) -> Optional[List[Dict]]:
    """
    Función de conveniencia para obtener el historial de predicciones.

    Args:
        limit: Número máximo de predicciones a recuperar

    Returns:
        Lista de predicciones o None si hay error
    """
    return api_client.get_predictions(skip=0, limit=limit)


def check_backend_status() -> bool:
    """
    Verifica el estado del backend y muestra un mensaje en Streamlit.

    Returns:
        True si el backend está disponible, False en caso contrario
    """
    if api_client.is_backend_available():
        st.sidebar.success("✅ Backend conectado")
        return True
    else:
        st.sidebar.warning("⚠️ Backend no disponible (modo offline)")
        st.sidebar.info(
            "💡 Para habilitar el historial, ejecuta: `uvicorn backend.database.main:app --reload`"
        )
        return False
