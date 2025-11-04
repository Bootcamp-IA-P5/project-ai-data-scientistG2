# ===============================================================
# 🧠 Proyecto: Predicción de Ictus - Data Scientist / AI Developer
# Entorno: CPU-only (TensorFlow + scikit-learn + Streamlit)
# ===============================================================

FROM python:3.10-slim

# Evitar prompts interactivos y establecer variables de entorno
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_DEFAULT_TIMEOUT=1000
ENV PIP_RETRIES=20

# Crear y establecer el directorio de trabajo
WORKDIR /app

# Copiar requirements.txt e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir --retries ${PIP_RETRIES} -r requirements.txt

# Copiar todo el código del proyecto
COPY . .

# Exponer puerto para Streamlit
EXPOSE 8501

# Variables de entorno para Streamlit
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Comando para ejecutar la aplicación Streamlit
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
