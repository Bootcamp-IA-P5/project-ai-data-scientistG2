# ===============================================================
# 🐳 Comandos Docker para el Proyecto de Predicción de Ictus
# ===============================================================

# 1. Construir la imagen Docker
docker build -t stroke-prediction:latest .

# 2. Ejecutar el contenedor
docker run -d --name stroke-app -p 8501:8501 stroke-prediction:latest

# 3. Ver logs del contenedor
docker logs stroke-app

# 4. Acceder al shell del contenedor
docker exec -it stroke-app /bin/bash

# 5. Detener el contenedor
docker stop stroke-app

# 6. Eliminar el contenedor
docker rm stroke-app

# 7. Eliminar la imagen
docker rmi stroke-prediction:latest

# 8. Ver contenedores en ejecución
docker ps

# 9. Ver todas las imágenes
docker images

# ===============================================================
# 🚀 Comandos usando los scripts bash creados
# ===============================================================

# Ejecutar aplicación directamente (sin Docker)
./run_app.sh

# Usar script Docker completo
./docker.sh build    # Construir imagen
./docker.sh run      # Ejecutar contenedor
./docker.sh stop     # Detener contenedor
./docker.sh logs     # Ver logs
./docker.sh clean    # Limpiar recursos
./docker.sh shell    # Acceder al shell

# Ejecutar Jupyter Notebook
./run_jupyter.sh

# ===============================================================
# 📱 URLs de acceso
# ===============================================================

# Streamlit App: http://localhost:8501
# Jupyter Notebook: http://localhost:8888

# ===============================================================
# 💡 Notas importantes
# ===============================================================

# 1. Asegúrate de tener Docker instalado
# 2. Los archivos del modelo deben estar en data/
# 3. El puerto 8501 debe estar libre
# 4. Para desarrollo, usa ./run_app.sh (más rápido)
# 5. Para producción, usa Docker