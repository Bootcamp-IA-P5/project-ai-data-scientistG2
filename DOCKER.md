# 🐳 Docker Deployment Guide

Guía para ejecutar la aplicación Stroke Risk Prediction System usando Docker.

## 📋 Prerequisitos

- Docker instalado (versión 20.10 o superior)
- Docker Compose instalado (versión 1.29 o superior)

### Verificar instalación:

```bash
docker --version
docker-compose --version
```

---

## 🚀 Inicio Rápido

### 1. Construir y levantar los servicios

Desde la raíz del proyecto:

```bash
docker-compose up --build
```

Esto:

- ✅ Construye las imágenes del backend y frontend
- ✅ Inicia ambos servicios
- ✅ Crea la red interna entre contenedores
- ✅ Expone los puertos necesarios

### 2. Acceder a la aplicación

Una vez que veas estos mensajes:

```
stroke-backend   | INFO:     Application startup complete.
stroke-frontend  | You can now view your Streamlit app in your browser.
```

Abre tu navegador:

- **Frontend (Streamlit)**: http://localhost:8501
- **Backend (FastAPI)**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## 🛠️ Comandos Útiles

### Iniciar servicios (modo detached)

```bash
docker-compose up -d
```

### Ver logs en tiempo real

```bash
# Todos los servicios
docker-compose logs -f

# Solo backend
docker-compose logs -f backend

# Solo frontend
docker-compose logs -f frontend
```

### Detener servicios

```bash
docker-compose down
```

### Detener y eliminar volúmenes (limpia la base de datos)

```bash
docker-compose down -v
```

### Reconstruir después de cambios

```bash
docker-compose up --build
```

### Ver contenedores en ejecución

```bash
docker-compose ps
```

---

## 📂 Arquitectura Docker

```
┌─────────────────────────────────────────┐
│         Docker Network (stroke-network) │
│                                         │
│  ┌──────────────┐    ┌──────────────┐  │
│  │   Backend    │    │   Frontend   │  │
│  │   FastAPI    │◄───│  Streamlit   │  │
│  │   :8000      │    │   :8501      │  │
│  └──────────────┘    └──────────────┘  │
│         │                               │
│    ┌────▼─────┐                         │
│    │ SQLite   │                         │
│    │  (vol)   │                         │
│    └──────────┘                         │
└─────────────────────────���───────────────┘
         │            │
    localhost:8000  localhost:8501
```

---

## 📦 Servicios

### Backend (FastAPI)

- **Puerto**: 8000
- **Imagen base**: python:3.12-slim
- **Healthcheck**: Verifica disponibilidad cada 30s
- **Persistencia**: Base de datos SQLite en volumen

### Frontend (Streamlit)

- **Puerto**: 8501
- **Imagen base**: python:3.12-slim
- **Depende de**: Backend
- **Variables de entorno**: BACKEND_URL apunta al backend interno

---

## 🗃️ Persistencia de Datos

La base de datos se persiste en:

```
./backend/database/predictions.db
```

Este archivo se mapea al contenedor, por lo que los datos sobreviven a reinicios.

### Limpiar base de datos

```bash
# Opción 1: Eliminar el archivo
rm backend/database/predictions.db
docker-compose restart backend

# Opción 2: Eliminar volúmenes
docker-compose down -v
docker-compose up -d
```

---

## 🔧 Configuración Avanzada

### Cambiar puertos

Edita `docker-compose.yml`:

```yaml
services:
  backend:
    ports:
      - "8080:8000" # Puerto host:contenedor

  frontend:
    ports:
      - "8502:8501"
```

### Variables de entorno personalizadas

Agrega en `docker-compose.yml`:

```yaml
services:
  frontend:
    environment:
      - BACKEND_URL=http://backend:8000
      - STREAMLIT_THEME_PRIMARY_COLOR="#FF4B4B"
```

### Usar archivo .env

Crea `.env` en la raíz:

```env
BACKEND_PORT=8000
FRONTEND_PORT=8501
```

Y modifica `docker-compose.yml`:

```yaml
services:
  backend:
    ports:
      - "${BACKEND_PORT}:8000"
```

---

## 🐛 Troubleshooting

### Error: "port is already allocated"

**Causa**: Puerto 8000 o 8501 ya en uso

**Solución**:

```bash
# Ver qué proceso usa el puerto
lsof -i :8000
lsof -i :8501

# Matar el proceso o cambiar puerto en docker-compose.yml
```

### Error: "Cannot connect to backend"

**Solución**:

```bash
# Ver logs del backend
docker-compose logs backend

# Verificar que el backend está corriendo
curl http://localhost:8000
```

### Error: "Model not found"

**Causa**: Falta el modelo en la carpeta `models/`

**Solución**:

```bash
# Verificar que existe el modelo
ls models/xgboost_modelo_final.pkl

# Si no existe, ejecutar el notebook XGBoost.ipynb antes de dockerizar
```

### Limpiar todo y empezar de nuevo

```bash
# Detener y eliminar todo
docker-compose down -v --rmi all

# Reconstruir desde cero
docker-compose up --build
```

---

## 📊 Monitoreo

### Ver uso de recursos

```bash
docker stats
```

### Inspeccionar contenedor

```bash
docker inspect stroke-backend
docker inspect stroke-frontend
```

### Entrar al contenedor (debug)

```bash
# Backend
docker exec -it stroke-backend /bin/bash

# Frontend
docker exec -it stroke-frontend /bin/bash
```

---

## 🚀 Despliegue en Producción

### Consideraciones:

1. **Usar variables de entorno** para configuración sensible
2. **Reverse proxy** (nginx) para HTTPS
3. **Volúmenes externos** para backups de la DB
4. **Health checks** configurados correctamente
5. **Logs centralizados** (ELK stack, CloudWatch, etc.)

### Ejemplo con nginx:

```nginx
server {
    listen 80;
    server_name tu-dominio.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location /api {
        proxy_pass http://localhost:8000;
    }
}
```

---

## 📝 Notas

- Los notebooks (`notebooks/`) NO se copian a las imágenes (ver `.dockerignore`)
- Los datos procesados (`data/processed/`) son opcionales
- El modelo XGBoost (`models/`) DEBE existir antes de construir
- Para desarrollo, es más rápido usar los comandos normales (sin Docker)
- Docker es ideal para despliegue y compartir la aplicación

---

## 🆘 Soporte

Si encuentras problemas:

1. Revisa los logs: `docker-compose logs -f`
2. Verifica que todos los archivos necesarios existen
3. Asegúrate de tener las versiones correctas de Docker
4. Revisa el README principal para más detalles

---

**¡Listo!** Tu aplicación de predicción de ictus ahora corre en contenedores Docker. 🎉
