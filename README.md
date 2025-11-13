# 🧠 Stroke Risk Prediction System

Sistema de predicción de riesgo de ictus usando Machine Learning con arquitectura cliente-servidor.

## 📋 Descripción

Aplicación completa de predicción de riesgo de ictus que integra:
- **Backend API** (FastAPI) para almacenar historial de predicciones
- **Frontend** (Streamlit) con 5 pestañas interactivas
- **Modelos ML** (LogisticRegression + LightGBM) con threshold optimization
- **Base de datos** (SQLite) para persistencia de datos

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
│   └── app.py                   # Aplicación Streamlit
│
├── notebooks/
│   ├── Preprocessing.ipynb      # Preprocesamiento de datos
│   ├── EDA.ipynb                # Análisis exploratorio
│   ├── train_model_tuning.py    # Entrenamiento de modelos
│   └── models/
│       └── ictus_model_*.pkl    # Modelos entrenados
│
├── data/
│   ├── stroke_dataset.csv       # Dataset original
│   └── processed/
│       ├── stroke_data_processed.csv       # Dataset entrenamiento procesado
│       └── stroke_data_processed_test.csv  # Dataset test procesado
│
├── test/
│   └── test_train_model_tuning.py
│
├── MODEL_FEATURES.md            # Documentación de features del modelo
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

---

## 📦 Archivos Necesarios

### ✅ Archivos Críticos (Deben existir)

| Archivo | Ruta | Descripción | Generado por |
|---------|------|-------------|--------------|
| **Dataset original** | `data/stroke_dataset.csv` | Datos originales de ictus | Manual |
| **Modelo entrenado** | `notebooks/models/ictus_model_20251113_211435.pkl` | Modelo ML entrenado | `train_model_tuning.py` |
| **Backend main** | `backend/database/main.py` | API FastAPI | Ya existe |
| **Frontend app** | `frontend/app.py` | Aplicación Streamlit | Ya existe |

### ⚠️ Archivos Opcionales (Se generan automáticamente)

| Archivo | Ruta | Descripción | Generado por |
|---------|------|-------------|--------------|
| **Datasets procesados** | `data/processed/stroke_data_processed.csv` | Para EDA/Evaluación | `Preprocessing.ipynb` |
| **Dataset test** | `data/processed/stroke_data_processed_test.csv` | Para evaluación | `Preprocessing.ipynb` |
| **Base de datos** | `backend/database/predictions.db` | Historial de predicciones | FastAPI (automático) |

---

## 🎯 Uso del Sistema

### Opción 1: Solo Predicciones Individuales (Sin Backend)

Si solo necesitas hacer predicciones individuales:

```bash
streamlit run frontend/app.py
```

✅ **Funciona**: Pestaña "🎯 Predicción Individual"
❌ **No funciona**: Pestaña "📜 Historial de Predicciones"

---

### Opción 2: Sistema Completo (Backend + Frontend)

Para usar todas las funcionalidades (incluido historial de predicciones):

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

### Generar Datasets Procesados (Opcional)

Si necesitas usar las pestañas EDA y Evaluación:

1. Abrir `notebooks/Preprocessing.ipynb` en Jupyter
2. Ejecutar todas las celdas
3. Se generarán automáticamente:
   - `data/processed/stroke_data_processed.csv`
   - `data/processed/stroke_data_processed_test.csv`

### Verificar API Backend

```bash
curl http://localhost:8000
```

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

### 🔹 Frontend Streamlit

#### Pestaña 1: 📊 EDA (Exploración de Datos)
- Visualización de distribución de datos
- Matriz de correlación
- Análisis de riesgo por edad

#### Pestaña 2: 🔮 Evaluación del Modelo
- Evaluación en datasets de test/train
- Matriz de confusión
- Métricas de clasificación

#### Pestaña 3: 📈 Métricas del Modelo
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC
- Distribución de probabilidades

#### Pestaña 4: 🎯 Predicción Individual
- Formulario interactivo
- Predicción en tiempo real
- Recomendaciones médicas
- Guardado automático en backend

#### Pestaña 5: 📜 Historial de Predicciones
- Visualización de predicciones pasadas
- Estadísticas del historial
- Gráficos de tendencias
- Distribución de resultados

---

## 🤖 Modelo de Machine Learning

### Características Principales

- **Algoritmo**: Voting Ensemble (LogisticRegression + LightGBM)
- **Features**: 20 características (ver [MODEL_FEATURES.md](MODEL_FEATURES.md))
- **Calibración**: Platt Scaling (sigmoid)
- **Threshold Optimization**: Cross-validation con recall mínimo
- **Balanceo**: SMOTE/ADASYN

### Features del Modelo (20 columnas)

```python
[
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
    'risk_factors', 'age_risk_interaction', 'gender_encoded',
    'ever_married_encoded', 'Residence_type_encoded',
    'work_type_Private', 'work_type_Self-employed',
    'smoking_status_never smoked', 'smoking_status_smokes',
    'age_group_36-50', 'age_group_51-65', 'age_group_65+',
    'bmi_category_Overweight', 'bmi_category_Obese',
    'glucose_category_Prediabetes'
]
```

**Nota**: El preprocesamiento genera 25 columnas, pero el modelo usa solo 20.  
Ver [MODEL_FEATURES.md](MODEL_FEATURES.md) para detalles completos.

---

## 🐛 Troubleshooting

### Error: "No se encontraron modelos entrenados"

**Solución**: Verificar que exista el archivo del modelo:
```bash
ls notebooks/models/ictus_model_*.pkl
```

Si no existe, entrenar el modelo:
```bash
cd notebooks
python train_model_tuning.py
```

### Error: "Backend no disponible"

**Solución**: Verificar que el backend esté corriendo:
```bash
curl http://localhost:8000
```

Si no responde, iniciar el backend:
```bash
cd backend/database
uvicorn main:app --reload
```

### Error: "X has 25 features, but expecting 20"

**Solución**: Este error está resuelto en la versión actual. Si persiste, verificar que `frontend/app.py` use `MODEL_EXPECTED_FEATURES` (20 columnas).

### Error: "FileNotFoundError: stroke_data_processed.csv"

**Solución**: Los datasets procesados son opcionales. Opciones:
1. Usar solo la pestaña de Predicción Individual (funciona sin datasets)
2. Generar los datasets ejecutando `notebooks/Preprocessing.ipynb`

---

## 🛠️ Tecnologías Utilizadas

| Categoría | Tecnología | Versión |
|-----------|-----------|---------|
| **Backend** | FastAPI | 0.115.5 |
| | Uvicorn | 0.34.0 |
| | SQLAlchemy | 2.0.44 |
| **Frontend** | Streamlit | 1.51.0 |
| | Plotly | 6.4.0 |
| **ML** | scikit-learn | 1.7.2 |
| | LightGBM | 4.6.0 |
| | imbalanced-learn | 0.14.0 |
| **Data** | pandas | 2.3.3 |
| | numpy | 2.3.4 |
| **DB** | SQLite | (built-in) |

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