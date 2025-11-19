# 🧠 Stroke Risk Prediction System

Sistema de predicción de riesgo de ictus usando Machine Learning con arquitectura cliente-servidor.

## 📋 Descripción

Aplicación completa de predicción de riesgo de ictus que integra:

- **Backend API** (FastAPI) para almacenar historial de predicciones
- **Frontend** (Streamlit) con 6 pestañas interactivas
- **Modelos ML Tabulares** (XGBoost optimizado con Optuna) para datos clínicos
- **Modelo de Imágenes** (CNN Dense) para análisis de MRI/TAC cerebral _(opcional)_
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
│   ├── PreprocessingCleaned.ipynb   # Preprocesamiento de datos tabulares
│   ├── XGBoost.ipynb                # Entrenamiento XGBoost con Optuna
│   ├── CNNImage.ipynb               # Modelo CNN para imágenes (Kaggle)
│   ├── EDA.ipynb                    # Análisis exploratorio
│   └── train_model_tuning.py        # Script de entrenamiento (legacy)
│
├── models/
│   ├── xgboost_modelo_final.pkl     # Modelo XGBoost entrenado
│   └── stroke_image_model.h5        # Modelo CNN (opcional)
│
├── data/
│   ├── stroke_dataset.csv       # Dataset original
│   ├── processed/
│   │   ├── stroke_data_processed.csv       # Dataset train procesado (25 features)
│   │   ├── stroke_data_processed_test.csv  # Dataset test procesado (25 features)
│   │   └── preprocessed_data.pkl           # Datos + scaler para modelos
│   └── images/                  # Imágenes MRI/TAC (opcional, para CNN)
│
├── test/
│   └── test_train_model_tuning.py
│
├── MODEL_FEATURES.md            # Documentación de features del modelo
├── requirements.txt             # Dependencias del proyecto
└── README.md                    # Este archivo
```

---

## ⚡ Inicio Rápido (5 minutos)

¿Primera vez usando el proyecto? Sigue estos pasos:

```bash
# 1. Clonar y configurar entorno
git clone https://github.com/Bootcamp-IA-P5/project-ai-data-scientistG2.git
cd project-ai-data-scientistG2
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Generar datos y entrenar modelo (en VS Code o Jupyter)
# - Ejecuta notebooks/PreprocessingCleaned.ipynb (Run All)
# - Ejecuta notebooks/XGBoost.ipynb (Run All)

# 3. Lanzar aplicación
streamlit run frontend/app.py
```

✅ **Listo!** Abre http://localhost:8501 y usa la pestaña "🎯 Predicción Individual"

---

## 🚀 Instalación Detallada

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

### 5. Generar datos preprocesados

**Primero, debes ejecutar el notebook de preprocesamiento:**

#### Opción A: Usando VS Code (Recomendado)

1. Abre `notebooks/PreprocessingCleaned.ipynb` en VS Code
2. Ejecuta todas las celdas (Run All)
3. Se generarán automáticamente los archivos en `data/processed/`

#### Opción B: Usando Jupyter

```bash
jupyter notebook notebooks/PreprocessingCleaned.ipynb
```

Luego ejecuta todas las celdas (Cell → Run All)

Este proceso genera:

- ✅ `data/processed/stroke_data_processed.csv` (3,984 filas × 26 columnas)
- ✅ `data/processed/stroke_data_processed_test.csv` (997 filas × 26 columnas)
- ✅ `data/processed/preprocessed_data.pkl` (datos + scaler)
- ⏱️ Duración: ~30 segundos

### 6. Entrenar el modelo XGBoost

**Ejecuta el notebook de entrenamiento:**

1. Abre `notebooks/XGBoost.ipynb` en VS Code o Jupyter
2. Ejecuta todas las celdas (Run All)

Este proceso:

- ✅ Carga datos desde `data/processed/preprocessed_data.pkl`
- ✅ Optimiza hiperparámetros con Optuna (80 trials, ~34 segundos)
- ✅ Aplica SMOTE en pipeline (evita data leakage)
- ✅ Entrena modelo XGBoost optimizado
- ✅ Guarda `models/xgboost_modelo_final.pkl`
- ⏱️ Duración estimada: 2-5 minutos

**Métricas esperadas:**

- Recall: 82% (detecta 41 de 50 ictus)
- Precision: 16%
- F1-Score: 26.8%
- AUC-ROC: 84.4%

### 7. (Opcional) Entrenar modelo de imágenes CNN

**⚠️ Este paso es OPCIONAL.** La aplicación funciona sin el modelo de imágenes.

El modelo de imágenes requiere:

- Datos de imágenes MRI/TAC en formato `.npy`
- Se recomienda entrenar en **Kaggle** (GPU gratuita)

**Pasos:**

1. Sube `notebooks/CNNImage.ipynb` a Kaggle
2. Conecta el dataset de imágenes
3. Ejecuta con GPU activada
4. Descarga `stroke_image_model.h5` y colócalo en `models/`

**Sin el modelo de imágenes:**

- ✅ Pestañas 1-5 funcionan normalmente (datos tabulares)
- ⚠️ Pestaña 6 "🖼️ Predicción con Imágenes" mostrará mensaje de advertencia

---

## 📦 Archivos Necesarios

### ✅ Archivos Críticos (Deben existir)

| Archivo              | Ruta                                   | Descripción                  | Generado por                     |
| -------------------- | -------------------------------------- | ---------------------------- | -------------------------------- |
| **Dataset original** | `data/stroke_dataset.csv`              | Datos originales de ictus    | Manual (incluido en repo)        |
| **Datos procesados** | `data/processed/preprocessed_data.pkl` | Datos preprocesados + scaler | **`PreprocessingCleaned.ipynb`** |
| **Modelo XGBoost**   | `models/xgboost_modelo_final.pkl`      | Modelo ML entrenado          | **`XGBoost.ipynb`**              |
| **Backend main**     | `backend/database/main.py`             | API FastAPI                  | Ya existe                        |
| **Frontend app**     | `frontend/app.py`                      | Aplicación Streamlit         | Ya existe                        |

### ⚠️ Archivos Opcionales

| Archivo                | Ruta                                            | Descripción                 | Generado por                 |
| ---------------------- | ----------------------------------------------- | --------------------------- | ---------------------------- |
| **Datasets CSV**       | `data/processed/stroke_data_processed.csv`      | Para EDA/Evaluación         | `PreprocessingCleaned.ipynb` |
| **Dataset test CSV**   | `data/processed/stroke_data_processed_test.csv` | Para evaluación             | `PreprocessingCleaned.ipynb` |
| **Modelo de imágenes** | `models/stroke_image_model.h5`                  | CNN para MRI/TAC (opcional) | `CNNImage.ipynb` (Kaggle)    |
| **Metadata imágenes**  | `models/stroke_image_model_metadata.json`       | Métricas del modelo CNN     | `CNNImage.ipynb` (Kaggle)    |
| **Base de datos**      | `backend/database/predictions.db`               | Historial de predicciones   | FastAPI (automático)         |

---

## 🎯 Uso del Sistema

### ⚠️ PREREQUISITOS: Generar Datos y Entrenar Modelo

**Antes de usar la aplicación por primera vez:**

1. **Preprocesar datos:** Ejecuta `PreprocessingCleaned.ipynb`

   - Genera `data/processed/preprocessed_data.pkl` (obligatorio)
   - Genera CSVs para EDA (opcional)

2. **Entrenar modelo XGBoost:** Ejecuta `XGBoost.ipynb`
   - Genera `models/xgboost_modelo_final.pkl` (obligatorio)
   - Duración: 2-5 minutos

✅ Con estos 2 pasos, la aplicación estará lista para usarse.

---

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

### 🔹 Frontend Streamlit (6 Pestañas)

#### Pestaña 1: 📊 EDA (Exploración de Datos)

- Visualización de distribución de datos
- Matriz de correlación
- Análisis de riesgo por edad
- **Requiere:** CSVs procesados (opcional)

#### Pestaña 2: 🔮 Evaluación del Modelo

- Evaluación en datasets de test/train
- Matriz de confusión
- Métricas de clasificación
- **Requiere:** CSVs procesados (opcional)

#### Pestaña 3: 📈 Métricas del Modelo

- Accuracy, Precision, Recall, F1-Score
- AUC-ROC
- Distribución de probabilidades
- **Requiere:** Evaluación previa en Pestaña 2

#### Pestaña 4: 🎯 Predicción Individual (Datos Tabulares)

- Formulario interactivo con factores de riesgo
- Predicción en tiempo real con XGBoost
- Recomendaciones médicas basadas en probabilidad
- Guardado automático en backend
- **✅ Siempre disponible** (no requiere CSVs)

#### Pestaña 5: �️ Predicción con Imágenes Médicas

- Carga de imágenes MRI/TAC cerebral
- Análisis con CNN Dense
- Probabilidad de ictus con visualización
- **⚠️ Requiere:** `stroke_image_model.h5` (opcional)
- **Sin modelo:** Muestra advertencia pero no bloquea la app

#### Pestaña 6: �📜 Historial de Predicciones

- Visualización de predicciones pasadas
- Estadísticas del historial
- Gráficos de tendencias
- Distribución de resultados
- **Requiere:** Backend API activo

---

## 🤖 Modelos de Machine Learning

### 📊 Modelo Principal: XGBoost Optimizado

#### Características Principales

- **Algoritmo**: XGBoost con optimización Optuna (80 trials)
- **Features**: 25 características clínicas
- **Balanceo**: SMOTE 20% en pipeline (evita data leakage)
- **Validación**: 5-fold StratifiedKFold Cross-Validation
- **Threshold**: 0.50 (optimizado para balance Recall/Precision)

#### Métricas de Rendimiento

| Métrica       | Test  | CV (5-fold) |
| ------------- | ----- | ----------- |
| **Recall**    | 82.0% | 74.3% ±8.1% |
| **Precision** | 16.0% | 14.3% ±1.9% |
| **F1-Score**  | 26.8% | 23.1% ±3.2% |
| **AUC-ROC**   | 84.4% | -           |

**Interpretación clínica:**

- ✅ Detecta **41 de 50** ictus reales (82% Recall)
- ⚠️ 215 falsas alarmas por 41 detecciones correctas (ratio 5.2:1)
- 🎯 Ideal para screening: prioriza **no perder casos de ictus**

#### Features del Modelo (25 columnas en orden)

```python
[
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
    'risk_factors', 'age_risk_interaction',
    'gender_encoded', 'ever_married_encoded', 'Residence_type_encoded',
    'work_type_Private', 'work_type_Self-employed', 'work_type_children',
    'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes',
    'age_group_19-35', 'age_group_36-50', 'age_group_51-65', 'age_group_65+',
    'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese',
    'glucose_category_Prediabetes', 'glucose_category_Diabetes'
]
```

Ver [MODEL_FEATURES.md](MODEL_FEATURES.md) para detalles completos.

---

### 🖼️ Modelo Opcional: CNN para Imágenes Médicas

#### Características

- **Arquitectura**: Dense Neural Network (512→256→128→1)
- **Input**: Imágenes MRI/TAC 128×128 o 224×224 (escala de grises)
- **Framework**: TensorFlow/Keras
- **Regularización**: L2 + BatchNormalization + Dropout
- **Entrenamiento**: En Kaggle con GPU (más rápido)

#### Estado

- ⚠️ **No disponible por defecto** (requiere entrenamiento manual)
- ✅ La aplicación funciona sin este modelo
- 📋 Instrucciones: Ver `notebooks/CNNImage.ipynb`

#### Métricas Esperadas

| Métrica       | Objetivo |
| ------------- | -------- |
| **Recall**    | ≥ 80%    |
| **Precision** | ≥ 40%    |
| **AUC-ROC**   | ≥ 0.85   |

---

## 🐛 Troubleshooting

### Error: "No se encontraron modelos entrenados"

**Causa**: No has ejecutado los notebooks de preprocesamiento y entrenamiento.

**Solución**:

1. **Verificar archivos necesarios:**

   ```bash
   # Verificar datos preprocesados
   ls data/processed/preprocessed_data.pkl

   # Verificar modelo entrenado
   ls models/xgboost_modelo_final.pkl
   ```

2. **Si faltan, generar en este orden:**

   **Paso 1:** Preprocesar datos

   - Abrir `notebooks/PreprocessingCleaned.ipynb`
   - Ejecutar todas las celdas (Run All)
   - Verifica que se creó: `data/processed/preprocessed_data.pkl`

   **Paso 2:** Entrenar modelo

   - Abrir `notebooks/XGBoost.ipynb`
   - Ejecutar todas las celdas (Run All)
   - Verifica que se creó: `models/xgboost_modelo_final.pkl`

3. **Reiniciar Streamlit:**
   ```bash
   # Detener con Ctrl+C y volver a ejecutar
   streamlit run frontend/app.py
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

### Error: "Feature shape mismatch"

**Causa**: Incompatibilidad entre features del CSV y del modelo.

**Solución**: Este error está resuelto. El modelo espera exactamente **25 features** y los archivos procesados generan 25.

Si persiste:

1. Regenera los datos ejecutando `PreprocessingCleaned.ipynb`
2. Verifica columnas del CSV:
   ```bash
   head -1 data/processed/stroke_data_processed.csv | tr ',' '\n' | wc -l
   # Debe mostrar: 26 (25 features + 1 target 'stroke')
   ```

### Error: "FileNotFoundError: stroke_data_processed.csv"

**Solución**: Los CSV son opcionales (solo para EDA/Evaluación).

**Opciones:**

1. **Usar solo predicciones individuales** (Pestaña 4) - funciona sin CSVs
2. **Generar los CSVs**:
   - Ejecuta `notebooks/PreprocessingCleaned.ipynb`
   - Se crearán automáticamente en `data/processed/`

### Error: "Modelo de imágenes no disponible"

**Solución**: El modelo CNN es **opcional**.

**Si no lo necesitas:**

- ⚠️ La pestaña "🖼️ Predicción con Imágenes" mostrará advertencia
- ✅ Todas las demás pestañas funcionan normalmente

**Si lo necesitas:**

1. Sube `notebooks/CNNImage.ipynb` a Kaggle
2. Conecta dataset de imágenes MRI/TAC
3. Entrena con GPU (gratuito en Kaggle)
4. Descarga `stroke_image_model.h5` y colócalo en `models/`

---

## 🛠️ Tecnologías Utilizadas

| Categoría    | Tecnología       | Versión    |
| ------------ | ---------------- | ---------- |
| **Backend**  | FastAPI          | 0.115.5    |
|              | Uvicorn          | 0.34.0     |
|              | SQLAlchemy       | 2.0.44     |
| **Frontend** | Streamlit        | 1.51.0     |
|              | Plotly           | 6.4.0      |
| **ML**       | scikit-learn     | 1.7.2      |
|              | LightGBM         | 4.6.0      |
|              | imbalanced-learn | 0.14.0     |
| **Data**     | pandas           | 2.3.3      |
|              | numpy            | 2.3.4      |
| **DB**       | SQLite           | (built-in) |

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
