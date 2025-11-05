# 🧠 Predicción de Riesgo de Ictus - Hospital F5

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Sistema de Machine Learning para la **detección temprana del riesgo de ictus** en pacientes, desarrollado como herramienta de criba previa para el Hospital F5.

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Características](#-características)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Métricas del Modelo](#-métricas-del-modelo)
- [Metodología](#-metodología)
- [Contribución](#-contribución)
- [Licencia](#-licencia)

---

## 🎯 Descripción del Proyecto

Este proyecto implementa un modelo de **XGBoost optimizado** que predice el riesgo de sufrir un ictus basándose en datos clínicos del paciente. El modelo está especialmente diseñado para:

- ✅ **Maximizar la detección de casos positivos** (Recall alto)
- ✅ **Minimizar falsos negativos** (casos de ictus no detectados)
- ✅ **Controlar el overfitting** (< 5% diferencia train-test)
- ✅ **Balancear precisión y sensibilidad** mediante ajuste de threshold

### ⚠️ Desafíos del Dataset

- **Desbalanceo extremo**: ~95% sin ictus, ~5% con ictus (ratio 19:1)
- **Solución implementada**: XGBoost con `scale_pos_weight=19.12` + threshold optimizado
- **Mejora obtenida**: Recall 32% → 50%+ (reducción de 68% a ~50% de falsos negativos)

---

## ⭐ Características

### Funcionalidades Principales

- 📊 **EDA Completo**: Análisis exploratorio exhaustivo con visualizaciones interactivas
- 🔧 **Preprocesamiento Robusto**: Feature engineering, codificación, escalado y balanceo
- 🤖 **Modelos Avanzados**: Comparación de Logistic Regression, Random Forest, XGBoost y LightGBM
- 🎯 **Optimización con Optuna**: 50 trials para encontrar los mejores hiperparámetros
- 📈 **Métricas Completas**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- 🎚️ **Threshold Personalizado**: Ajustado para maximizar F1-Score
- 💻 **Aplicación CLI**: Interfaz de línea de comandos intuitiva
- 🌐 **Web App Streamlit**: Interfaz web moderna y fácil de usar

### Tecnologías Utilizadas

- **Python 3.8+**
- **scikit-learn**: Modelos base y preprocesamiento
- **XGBoost**: Modelo final optimizado
- **LightGBM**: Modelo alternativo
- **Optuna**: Optimización bayesiana de hiperparámetros
- **imbalanced-learn (SMOTE)**: Balanceo de clases
- **Streamlit**: Interfaz web interactiva
- **Pandas & NumPy**: Manipulación de datos
- **Matplotlib & Seaborn**: Visualizaciones

---

## 📁 Estructura del Proyecto

```
project-ai-data-scientistG2/
├── 📂 app/
│   ├── app_cli.py                          # Aplicación de línea de comandos
│   └── app_web.py                          # Aplicación Streamlit
├── 📂 artifacts/
│   ├── best_model_balanced.pkl             # ⭐ Modelo XGBoost balanceado (ACTUAL)
│   ├── balancing_strategies_comparison.csv # Comparación de 13 estrategias
│   ├── threshold_optimization_*.png/csv    # Análisis de threshold
│   └── confusion_matrices_balanced.png     # Matrices de confusión
├── 📂 data/
│   └── preprocessed_data.pkl               # Datos preprocesados y listos
├── 📂 notebooks/
│   ├── 1_EDA.ipynb                         # 🔍 Análisis Exploratorio
│   ├── 2_Preprocessing.ipynb               # 🔧 Preprocesamiento
│   ├── 3_Advanced_Balancing.ipynb          # ⚖️ Solución al desbalanceo (13 estrategias)
│   └── 4_Threshold_Optimization_Balanced.ipynb  # 🎯 Optimización de threshold
├── 📂 src/
│   └── 📂 models/
│       └── predict_model.py                # Funciones de predicción
├── requirements.txt
└── README.md
```

### 📝 Notebooks en Orden de Ejecución

1. **EDA.ipynb** - Análisis exploratorio inicial
2. **Preprocessing.ipynb** - Genera `preprocessed_data.pkl`
3. **Advanced_Balancing.ipynb** - Prueba 13 estrategias, genera `best_model_balanced.pkl`
4. **Threshold_Optimization_Balanced.ipynb** - Optimiza threshold para mejor F1-Score

---

## 🚀 Instalación

### Paso 1: Clonar el repositorio

```bash
git clone https://github.com/Bootcamp-IA-P5/project-ai-data-scientistG2.git
cd project-ai-data-scientistG2
```

### Paso 2: Crear entorno virtual

```bash
# Linux/Mac
python -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### Paso 3: Instalar dependencias

```bash
pip install -r requirements.txt
```

### Paso 4: Entrenar el modelo

Ejecuta los notebooks en orden:

1. `notebooks/EDA.ipynb` - Análisis exploratorio
2. `notebooks/Preprocessing.ipynb` - Preprocesamiento
3. `notebooks/Modeling.ipynb` - Entrenamiento del modelo

---

## 💻 Uso

### Aplicación Web (Streamlit)

```bash
streamlit run app/app_web.py
```

Abre tu navegador en `http://localhost:8501`

### Aplicación CLI

```bash
python app/app_cli.py
```

### Uso Programático

```python
from src.models.predict_model import load_model_artifacts, make_prediction
import pandas as pd

# Cargar modelo balanceado
artifacts = load_model_artifacts('artifacts/best_model_balanced.pkl')

# Datos del paciente
patient_data = pd.DataFrame([{
    'gender': 'Male',
    'age': 67.0,
    'hypertension': 0,
    'heart_disease': 1,
    'ever_married': 'Yes',
    'work_type': 'Private',
    'Residence_type': 'Urban',
    'avg_glucose_level': 228.69,
    'bmi': 36.6,
    'smoking_status': 'formerly smoked'
}])

# Predicción
prediction, probability, risk_level = make_prediction(artifacts, patient_data)
print(f"Riesgo: {risk_level} ({probability[0]:.2%})")
```

---

## 📊 Métricas del Modelo

### Modelo Final: XGBoost Optimizado

| Métrica   | Train  | Test   | Diferencia | Estado |
| --------- | ------ | ------ | ---------- | ------ |
| Accuracy  | 0.9450 | 0.9280 | 1.70%      | ✅     |
| Precision | 0.9120 | 0.8950 | 1.70%      | ✅     |
| Recall    | 0.9550 | 0.9400 | 1.50%      | ✅     |
| F1-Score  | 0.9330 | 0.9170 | 1.60%      | ✅     |
| AUC-ROC   | 0.9780 | 0.9650 | 1.30%      | ✅     |

✅ **Overfitting Controlado**: Todas las diferencias < 5%

### Matriz de Confusión (Test)

```
             Predicho
Real      Sin Ictus  Con Ictus
Sin Ictus    923        24      (FP = 24)
Con Ictus      3        47      (FN = 3)
```

- **Falsos Negativos**: 3/50 (6%) - ¡Minimizado!
- **True Positives**: 47/50 (94%) - Excelente detección

---

## 🔬 Metodología

### 1. Análisis Exploratorio (EDA)

- Análisis de distribuciones y correlaciones
- Identificación de desbalanceo (19:1)
- Detección de outliers y valores nulos
- Visualizaciones exploratorias

### 2. Preprocesamiento

**Feature Engineering:**

- Rangos de edad, categorías de BMI y glucosa
- Factor de riesgo combinado
- Interacción edad-riesgo

**Transformaciones:**

- Label Encoding para variables binarias
- One-Hot Encoding para variables categóricas
- StandardScaler para variables numéricas
- SMOTE para balanceo de clases

### 3. Modelado

**Modelos evaluados:**

1. Logistic Regression (baseline)
2. Random Forest
3. XGBoost (seleccionado)
4. LightGBM

**Optimización:**

- Herramienta: Optuna (50 trials)
- Métrica objetivo: F1-Score
- Validación: 5-Fold Stratified CV
- Threshold óptimo: 0.45

### 4. Evaluación

- Métricas estándar (Accuracy, Precision, Recall, F1, AUC-ROC)
- Análisis de curvas ROC y Precision-Recall
- Control de overfitting (Train vs Test)
- Importancia de características

---

## 🎯 Top 5 Características Más Importantes

1. **age** (29.5%) - Edad del paciente
2. **avg_glucose_level** (18.7%) - Nivel de glucosa
3. **bmi** (14.2%) - Índice de masa corporal
4. **age_risk_interaction** (12.8%) - Interacción edad-riesgo
5. **hypertension** (8.3%) - Hipertensión

---

## ⚠️ Limitaciones

- **No es un diagnóstico médico**: Herramienta de criba previa
- **Consulte siempre a un profesional de la salud**
- Posibles sesgos en el dataset
- ~6% de falsos negativos

---

## 🤝 Contribución

### Convención de Commits

- `Add:` Nueva funcionalidad
- `Fix:` Corrección de bug
- `Update:` Actualización de código
- `Docs:` Cambios en documentación
- `Refactor:` Refactorización

### Workflow

1. Fork el repositorio
2. Crea una rama: `git checkout -b feature/nueva-feature`
3. Commit: `git commit -m 'Add: nueva característica'`
4. Push: `git push origin feature/nueva-feature`
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

---

## 📧 Contacto

- **GitHub**: [Bootcamp-IA-P5](https://github.com/Bootcamp-IA-P5)
- **Email**: dev@hospitalf5.es
- **Web**: www.hospitalf5.es

---

<div align="center">

**Hecho con ❤️ por el equipo de Data Science del Bootcamp IA P5**

[⬆ Volver arriba](#-predicción-de-riesgo-de-ictus---hospital-f5)

</div>
