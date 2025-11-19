# Backend Dockerfile para FastAPI
FROM python:3.12-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias (solo las necesarias para backend)
RUN pip install --no-cache-dir fastapi uvicorn sqlalchemy

# Copiar el código del backend
COPY backend/ ./backend/

# Cambiar al directorio del backend
WORKDIR /app/backend/database

# Exponer puerto
EXPOSE 8000

# Comando para iniciar el servidor (desde backend/database/)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
