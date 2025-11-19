# 🧠 Stroke Risk Prediction System

Sistema de predicción de riesgo de ictus usando Machine Learning con arquitectura cliente-servidor.

## 📋 Descripción

Aplicación completa de predicción de riesgo de ictus que integra:

- **Backend API** (FastAPI) para almacenar historial de predicciones
- **Frontend** (Streamlit) con 6 pestañas interactivas
- **Modelo ML** (XGBoost) optimizado con Optuna para alta Recall (82%)
- **Base de datos** (SQLite) para persistencia de datos
- **Sistema de riesgo** en 3 niveles (BAJO/MODERADO/ALTO)

---

## 🏗️ Estructura del Proyecto

```
project-ai-data-scientistG2/
├── backend/
│   └── database/
│       ├── __init__.py
│       ├── database.py          # Configuración SQLAlchemy + modelos
│       ├── main.py              # Endpoints FastAPI
│       └── predictions.db       # Base de datos SQLite (generada automáticamente)
│
├── frontend/
│   ├── api_client.py            # Cliente HTTP para backend
│   └── app.py                   # Aplicación Streamlit (6 tabs)
│
├── notebooks/
│   ├── PreprocessingCleaned.ipynb  # Preprocesamiento final (25 features)
│   ├── XGBoost.ipynb               # Entrenamiento XGBoost con Optuna
│   ├── EDA.ipynb                   # Análisis exploratorio
│   ├── CNNImage.ipynb              # Modelo de imágenes (OPCIONAL)
│   └── MLP_classweight.ipynb       # Modelo MLP (EN DESARROLLO)
│
├── models/
│   ├── xgboost_modelo_final.pkl    # Modelo XGBoost optimizado
│   └── preprocessed_data.pkl       # Scaler y metadatos
│
├── data/
│   ├── stroke_dataset.csv          # Dataset original (3984 registros)
│   └── processed/
│       ├── stroke_data_processed.csv       # Dataset entrenamiento (25 features + target)
│       └── stroke_data_processed_test.csv  # Dataset test (997 registros)
│
├── test/
│   └── test_train_model_tuning.py
│
├── MODEL_FEATURES.md            # Documentación detallada de 25 features
├── requirements.txt             # Dependencias del proyecto
└── README.md                    # Este archivo
```

---

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/Bootcamp-IA-P5/project-ai-data-scientistG2.git
cd project-ai-data-scientistG2
```

### 2. Crear entorno virtual

```bash
python -m venv .venv
```

### 3. Activar entorno virtual

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux/Mac:**

```bash
source .venv/bin/activate
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 5. Generar datos preprocesados y entrenar modelo

**IMPORTANTE**: Debes ejecutar los notebooks en orden:

#### **Paso 1: Preprocesamiento de datos**

```bash
jupyter notebook
```

Abre y ejecuta `notebooks/PreprocessingCleaned.ipynb` (Run All)

Genera:

- ✅ `data/processed/stroke_data_processed.csv` (3984 registros, 25 features + target)
- ✅ `data/processed/stroke_data_processed_test.csv` (997 registros)
- ✅ `models/preprocessed_data.pkl` (scaler + metadatos)

#### **Paso 2: Entrenar modelo XGBoost**

Abre y ejecuta `notebooks/XGBoost.ipynb` (Run All)

Genera:

- ✅ `models/xgboost_modelo_final.pkl` (modelo + pipeline + metadatos)
- ⏱️ Duración: ~34 segundos (80 trials Optuna)
- 📊 Métricas: Recall 82%, Precision 16%, F1 26.8%, AUC 84.4%

**Nota**: Los modelos ya están entrenados en el repositorio, este paso es opcional salvo que quieras reentrenar.

---

## 📦 Archivos Necesarios

### ✅ Archivos Críticos (Ya incluidos en el repositorio)

| Archivo               | Ruta                              | Descripción                       | Estado      |
| --------------------- | --------------------------------- | --------------------------------- | ----------- |
| **Dataset original**  | `data/stroke_dataset.csv`         | Datos originales (3984 registros) | ✅ Incluido |
| **Modelo XGBoost**    | `models/xgboost_modelo_final.pkl` | Modelo optimizado con Optuna      | ✅ Incluido |
| **Scaler + metadata** | `models/preprocessed_data.pkl`    | StandardScaler + feature names    | ✅ Incluido |
| **Backend API**       | `backend/database/main.py`        | Endpoints FastAPI                 | ✅ Incluido |
| **Frontend app**      | `frontend/app.py`                 | Aplicación Streamlit (6 tabs)     | ✅ Incluido |

### ⚠️ Archivos Opcionales (Solo para EDA y evaluación)

| Archivo               | Ruta                                            | Descripción               | Generado por                 |
| --------------------- | ----------------------------------------------- | ------------------------- | ---------------------------- |
| **Dataset procesado** | `data/processed/stroke_data_processed.csv`      | Train set (25 features)   | `PreprocessingCleaned.ipynb` |
| **Dataset test**      | `data/processed/stroke_data_processed_test.csv` | Test set (997 registros)  | `PreprocessingCleaned.ipynb` |
| **Base de datos**     | `backend/database/predictions.db`               | Historial de predicciones | FastAPI (automático)         |

**Nota**: Sin los datasets procesados, las pestañas 1 (EDA) y 2 (Evaluación) mostrarán un aviso. La pestaña 4 (Predicción Individual) funciona siempre.

---

## 🎯 Uso del Sistema

### ⚡ Inicio Rápido (3 pasos)

El modelo ya está entrenado en el repositorio. Solo necesitas:

```bash
# 1. Activar entorno virtual
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 2. (Opcional) Iniciar backend en otra terminal
cd backend/database
uvicorn main:app --reload

# 3. Lanzar Streamlit
streamlit run frontend/app.py
```

✅ **Listo!** Abre http://localhost:8501 en tu navegador

---

### Opción 1: Solo Predicciones Individuales (Sin Backend)

Si solo necesitas hacer predicciones individuales:

```bash
streamlit run frontend/app.py
```

**Pestañas disponibles:**

- ✅ **Tab 4**: 🎯 Predicción Individual (siempre funcional)
- ⚠️ **Tab 1**: 📊 EDA (requiere datasets procesados)
- ⚠️ **Tab 2**: 🔮 Evaluación (requiere datasets procesados)
- ❌ **Tab 6**: 📜 Historial (requiere backend)

---

### Opción 2: Sistema Completo (Backend + Frontend)

Para usar todas las 6 pestañas (incluido historial de predicciones):

#### **Terminal 1: Iniciar Backend API**

```bash
cd backend/database
uvicorn main:app --reload
```

Salida esperada:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [xxxxx] using StatReload
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Verificar backend: http://localhost:8000

#### **Terminal 2: Iniciar Frontend Streamlit**

```bash
streamlit run frontend/app.py
```

Salida esperada:

```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

Abrir navegador: http://localhost:8501

---

## 🔧 Comandos Útiles

### Generar Datasets Procesados (Solo si quieres usar EDA/Evaluación)

```bash
jupyter notebook
# Abrir y ejecutar PreprocessingCleaned.ipynb (Run All)
```

Genera:

- `data/processed/stroke_data_processed.csv` (3984 registros, 25 features)
- `data/processed/stroke_data_processed_test.csv` (997 registros)

### Reentrenar Modelo (Opcional)

```bash
jupyter notebook
# Abrir y ejecutar XGBoost.ipynb (Run All)
```

Genera nuevo modelo optimizado con Optuna (80 trials, ~34 segundos)

### Verificar API Backend

```bash
curl http://localhost:8000
```

Respuesta esperada: `{"message": "Stroke Prediction API"}`

### Ver Predicciones Guardadas

```bash
curl http://localhost:8000/predictions/
```

### Ejecutar Tests

```bash
pytest test/test_train_model_tuning.py -v
```

---

## 📊 Características del Sistema

### 🔹 Backend API (FastAPI)

- **GET /**  
  Health check del servidor

- **POST /predictions/**  
  Guardar nueva predicción

  ```json
  {
    "input_data": {...},
    "prediction_result": "RIESGO ALTO",
    "confidence": 0.85
  }
  ```

- **GET /predictions/**  
  Listar todas las predicciones (con paginación)

- **GET /predictions/{id}**  
  Obtener predicción por ID

### 🔹 Frontend Streamlit (6 Pestañas)

#### ✅ Pestaña 1: 📊 EDA (Exploración de Datos)

- Visualización de distribución de datos
- Matriz de correlación de 25 features
- Análisis de riesgo por edad
- **Requiere**: datasets procesados

#### ✅ Pestaña 2: 🔮 Evaluación del Modelo

- Evaluación en datasets de test/train/ambos
- Matriz de confusión
- Métricas de clasificación
- Preview de predicciones
- **Requiere**: datasets procesados

#### ✅ Pestaña 3: 📈 Métricas del Modelo

- Accuracy, Precision, Recall, F1-Score
- AUC-ROC (84.4%)
- Distribución de probabilidades
- Interpretación de métricas clave
- **Requiere**: evaluación previa en Tab 2

#### ✅ Pestaña 4: 🎯 Predicción Individual (SIEMPRE FUNCIONAL)

- Formulario interactivo con 10 campos
- Predicción en tiempo real con XGBoost
- **Sistema de riesgo en 3 niveles**:
  - 🟢 BAJO (<20%): Recomendaciones preventivas
  - 🟡 MODERADO (20-50%): Evaluación médica
  - 🔴 ALTO (≥50%): Atención urgente
- Barra visual de probabilidad
- Recomendaciones médicas personalizadas
- Guardado automático en backend (si está activo)
- Vista de datos procesados

#### ⚠️ Pestaña 5: �️ Predicción con Imágenes (OPCIONAL)

- Análisis de resonancias magnéticas cerebrales
- Carga de imágenes PNG/JPG/JPEG
- Procesamiento y predicción con CNN
- **Requiere**: modelo `stroke_image_model.h5` (en desarrollo)
- **Estado**: Funcionalidad experimental

#### ✅ Pestaña 6: 📜 Historial de Predicciones

- Tabla de predicciones pasadas
- Estadísticas del historial
- Gráficos de tendencias de confianza
- Distribución de resultados por categoría
- Filtros y paginación
- **Requiere**: backend activo

---

## 🤖 Modelo de Machine Learning

### Características Principales

- **Algoritmo**: XGBoost optimizado con Optuna
- **Objetivo**: Maximizar Recall (detección de casos positivos)
- **Features**: 25 características (ver [MODEL_FEATURES.md](MODEL_FEATURES.md))
- **Balanceo**: SMOTE (20% sampling strategy) en ImbPipeline
- **Optimización**: 80 trials Optuna (~34 segundos)
- **Threshold**: 0.50 (optimizado para recall)

### Métricas del Modelo (Test Set)

| Métrica    | Valor | Descripción                                            |
| ---------- | ----- | ------------------------------------------------------ |
| **Recall** | 82%   | Detecta 82% de casos positivos (✅ Objetivo principal) |
| Precision  | 16%   | 16% de predicciones positivas correctas                |
| F1-Score   | 26.8% | Balance recall-precision                               |
| AUC-ROC    | 84.4% | Excelente capacidad discriminativa                     |
| Accuracy   | 86%   | Precisión global del modelo                            |

**Enfoque médico**: Prioriza Recall sobre Precision para minimizar falsos negativos (casos de ictus no detectados).

### Features del Modelo (25 columnas)

```python
[
    # Numéricas originales (4)
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',

    # Features engineered (2)
    'risk_factors', 'age_risk_interaction',

    # Label encoding (3)
    'gender_encoded', 'ever_married_encoded', 'Residence_type_encoded',

    # One-hot: work_type (3 de 5)
    'work_type_Private', 'work_type_Self-employed', 'work_type_children',

    # One-hot: smoking_status (3 de 4)
    'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes',

    # One-hot: age_group (4 de 5)
    'age_group_19-35', 'age_group_36-50', 'age_group_51-65', 'age_group_65+',

    # One-hot: bmi_category (3 de 4)
    'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese',

    # One-hot: glucose_category (2 de 3)
    'glucose_category_Prediabetes', 'glucose_category_Diabetes'
]
```

**Preprocesamiento**: Solo 4 features numéricas son escaladas con StandardScaler.  
Ver [MODEL_FEATURES.md](MODEL_FEATURES.md) para detalles completos de cada feature.

---

## 🐛 Troubleshooting

### Error: "No se encontraron modelos entrenados"

**Causa**: Archivo `models/xgboost_modelo_final.pkl` no encontrado.

**Solución**:

```bash
# Verificar que existe
ls models/xgboost_modelo_final.pkl

# Si no existe, reentrenar con Jupyter
jupyter notebook
# Abrir notebooks/XGBoost.ipynb y ejecutar todas las celdas
```

El modelo ya está incluido en el repositorio. Este error solo ocurre si lo borraste accidentalmente.

### Error: "Backend no disponible"

**Síntoma**: Tab 6 (Historial) muestra "⚠️ El backend no está disponible"

**Solución**:

```bash
# Verificar que el backend esté corriendo
curl http://localhost:8000

# Si no responde, iniciar el backend en otra terminal
cd backend/database
uvicorn main:app --reload
```

**Nota**: La Tab 4 (Predicción Individual) funciona sin backend, pero no guarda historial.

### Error: "Feature shape mismatch"

**Solución**: Este error está resuelto en la versión actual. El modelo espera **25 features** exactas.

Si persiste:

1. Verificar que `PreprocessingCleaned.ipynb` genera 25 features
2. Verificar que `app.py` use `MODEL_EXPECTED_FEATURES` con 25 columnas
3. Revisar [MODEL_FEATURES.md](MODEL_FEATURES.md) para la lista correcta

### Warning: "⚠️ Esta pestaña requiere los datasets de evaluación"

**Síntoma**: Tabs 1 (EDA) y 2 (Evaluación) muestran advertencia.

**Solución**: Generar datasets procesados (opcional):

```bash
jupyter notebook
# Abrir notebooks/PreprocessingCleaned.ipynb y ejecutar Run All
```

Genera:

- `data/processed/stroke_data_processed.csv`
- `data/processed/stroke_data_processed_test.csv`

**Nota**: La Tab 4 (Predicción Individual) funciona sin estos archivos.

### Error: "Progress Value has invalid type: float32"

**Solución**: Actualizado en la última versión de `app.py`. Si persiste, actualiza desde el repositorio:

```bash
git pull origin dev
```

---

## 🛠️ Tecnologías Utilizadas

| Categoría    | Tecnología       | Versión    | Uso                               |
| ------------ | ---------------- | ---------- | --------------------------------- |
| **Backend**  | FastAPI          | 0.115.5    | API REST para predicciones        |
|              | Uvicorn          | 0.34.0     | Servidor ASGI                     |
|              | SQLAlchemy       | 2.0.44     | ORM para base de datos            |
| **Frontend** | Streamlit        | 1.51.0     | Interfaz web interactiva (6 tabs) |
|              | Plotly           | 6.4.0      | Visualizaciones dinámicas         |
| **ML**       | XGBoost          | 3.1.1      | Modelo principal de predicción    |
|              | scikit-learn     | 1.7.2      | Preprocesamiento y métricas       |
|              | imbalanced-learn | 0.14.0     | SMOTE para balanceo de clases     |
|              | Optuna           | 4.2.0      | Optimización de hiperparámetros   |
| **DL**       | TensorFlow       | 2.20.0     | Modelo CNN imágenes (opcional)    |
|              | Keras            | 3.9.0      | API alto nivel para DL            |
| **Data**     | pandas           | 2.3.3      | Manipulación de datos             |
|              | numpy            | 2.3.4      | Operaciones numéricas             |
| **DB**       | SQLite           | (built-in) | Almacenamiento de historial       |

---

## 👥 Equipo

Bootcamp IA - Promoción 5 (P5)  
Grupo 2 - Data Scientist

---

## 📄 Licencia

Este proyecto es parte del Bootcamp de IA.

---

## 📞 Soporte

Para problemas o preguntas:

1. Revisar [MODEL_FEATURES.md](MODEL_FEATURES.md) para detalles del modelo
2. Verificar que todos los archivos necesarios existan
3. Revisar la sección de Troubleshooting arriba
