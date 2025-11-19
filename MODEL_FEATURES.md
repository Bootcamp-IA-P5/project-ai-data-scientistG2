# 🎯 Features del Modelo de Ictus

## Resumen

El modelo XGBoost espera **exactamente 25 features** en el siguiente orden:

**⚠️ IMPORTANTE:** Las 25 features se generan automáticamente al ejecutar `PreprocessingCleaned.ipynb`

## Lista de Features (25 columnas)

### 1-7: Features Numéricas Base y Engineeradas

1. `age` - Edad del paciente (escalada con StandardScaler)
2. `hypertension` - Hipertensión (0=No, 1=Sí)
3. `heart_disease` - Enfermedad cardíaca (0=No, 1=Sí)
4. `avg_glucose_level` - Nivel promedio de glucosa (escalado)
5. `bmi` - Índice de masa corporal (escalado)
6. `risk_factors` - Suma de hipertension + heart_disease (feature engineerada)
7. `age_risk_interaction` - age × risk_factors (escalado)

### 8-10: Variables Binarias Codificadas (Label Encoding)

8. `gender_encoded` - Género (0=Female, 1=Male)
9. `ever_married_encoded` - Casado anteriormente (0=No, 1=Yes)
10. `Residence_type_encoded` - Tipo de residencia (0=Rural, 1=Urban)

### 11-13: Work Type (One-Hot Encoding, drop_first=True)

11. `work_type_Private` - Trabajo privado (1 si aplica, 0 si no)
12. `work_type_Self-employed` - Autónomo (1 si aplica, 0 si no)
13. `work_type_children` - Niño/estudiante (1 si aplica, 0 si no)

- **Baseline (todas = 0)**: Govt_job o Never_worked

### 14-16: Smoking Status (One-Hot Encoding, drop_first=True)

14. `smoking_status_formerly smoked` - Fumó anteriormente (1 si aplica, 0 si no)
15. `smoking_status_never smoked` - Nunca fumó (1 si aplica, 0 si no)
16. `smoking_status_smokes` - Fuma actualmente (1 si aplica, 0 si no)

- **Baseline (todas = 0)**: Estado de tabaquismo desconocido

### 17-20: Age Group (One-Hot Encoding, drop_first=True)

17. `age_group_19-35` - Edad entre 19-35 años (1 si aplica, 0 si no)
18. `age_group_36-50` - Edad entre 36-50 años (1 si aplica, 0 si no)
19. `age_group_51-65` - Edad entre 51-65 años (1 si aplica, 0 si no)
20. `age_group_65+` - Edad 65+ años (1 si aplica, 0 si no)

- **Baseline (todas = 0)**: age_group_0-18

21. `bmi_category_Normal` - Peso normal (1 si aplica, 0 si no)
22. `bmi_category_Overweight` - Sobrepeso (1 si aplica, 0 si no)
23. `bmi_category_Obese` - Obesidad (1 si aplica, 0 si no)

- **Baseline (todas = 0)**: bmi_category_Underweight

### 24-25: Glucose Category (One-Hot Encoding, drop_first=True)

24. `glucose_category_Prediabetes` - Prediabetes (100-125 mg/dL) (1 si aplica, 0 si no)
25. `glucose_category_Diabetes` - Diabetes (>125 mg/dL) (1 si aplica, 0 si no)

- **Baseline (todas = 0)**: glucose_category_Normal (<100 mg/dL)

---

## ⚠️ Notas Importantes

### Preprocesamiento Actualizado

El notebook `PreprocessingCleaned.ipynb` genera **exactamente 25 columnas** que coinciden con las features del modelo.

### ✅ Sin Discrepancias

- El preprocesamiento y el modelo están **perfectamente alineados**
- **NO** se eliminan columnas
- Las 25 features se usan directamente sin transformaciones adicionales

### Escalado de Variables Numéricas

Solo las siguientes 4 features se escalan con `StandardScaler`:

- `age`
- `avg_glucose_level`
- `bmi`
- `age_risk_interaction`

Las variables binarias (one-hot encoding) **NO** se escalan (permanecen en 0 o 1).

### Aplicación en el Código

Para hacer predicciones correctamente:

```python
MODEL_EXPECTED_FEATURES = [
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
    'risk_factors', 'age_risk_interaction', 'gender_encoded',
    'ever_married_encoded', 'Residence_type_encoded',
    'work_type_Private', 'work_type_Self-employed',
    'smoking_status_never smoked', 'smoking_status_smokes',
    'age_group_36-50', 'age_group_51-65', 'age_group_65+',
    'bmi_category_Overweight', 'bmi_category_Obese',
    'glucose_category_Prediabetes'
]

# Después del preprocesamiento, seleccionar solo estas columnas:
X_for_model = X[MODEL_EXPECTED_FEATURES]
```

---

## 📅 Información del Modelo

- **Archivo del modelo**: `notebooks/models/ictus_model_20251113_211435.pkl`
- **Tipo de modelo**: `VotingClassifier` (LogisticRegression + LightGBM)
- **Features esperadas**: 20
- **Última actualización**: 13 de noviembre de 2025
