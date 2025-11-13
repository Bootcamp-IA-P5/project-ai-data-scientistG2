# 🎯 Features del Modelo de Ictus

## Resumen

El modelo de clasificación espera **exactamente 20 features** en el siguiente orden:

## Lista de Features (20 columnas)

### 1-7: Features Numéricas Base y Engineeradas

1. `age` - Edad del paciente
2. `hypertension` - Hipertensión (0=No, 1=Sí)
3. `heart_disease` - Enfermedad cardíaca (0=No, 1=Sí)
4. `avg_glucose_level` - Nivel promedio de glucosa
5. `bmi` - Índice de masa corporal
6. `risk_factors` - Suma de hipertension + heart_disease
7. `age_risk_interaction` - age \* risk_factors

### 8-10: Variables Binarias Codificadas (Label Encoding)

8. `gender_encoded` - Género (0=Female, 1=Male, 2=Other)
9. `ever_married_encoded` - Casado anteriormente (0=No, 1=Yes)
10. `Residence_type_encoded` - Tipo de residencia (0=Rural, 1=Urban)

### 11-12: Work Type (One-Hot Encoding, drop_first=True)

11. `work_type_Private` - Trabajo privado
12. `work_type_Self-employed` - Autónomo

- **Categorías eliminadas**: `work_type_children` (primera alfabéticamente)
- **Categorías no presentes**: `work_type_Govt_job`, `work_type_Never_worked`

### 13-14: Smoking Status (One-Hot Encoding, drop_first=True)

13. `smoking_status_never smoked` - Nunca fumó
14. `smoking_status_smokes` - Fuma actualmente

- **Categoría eliminada**: `smoking_status_formerly smoked` (primera alfabéticamente)

### 15-17: Age Group (One-Hot Encoding, drop_first=True)

15. `age_group_36-50` - Edad entre 36-50 años
16. `age_group_51-65` - Edad entre 51-65 años
17. `age_group_65+` - Edad 65+ años

- **Categoría eliminada**: `age_group_19-35` (primera alfabéticamente)
- **Categoría no presente**: `age_group_0-18`

### 18-19: BMI Category (One-Hot Encoding, drop_first=True)

18. `bmi_category_Overweight` - Sobrepeso
19. `bmi_category_Obese` - Obesidad

- **Categoría eliminada**: `bmi_category_Normal` (primera alfabéticamente)
- **Categoría no presente**: `bmi_category_Underweight`

### 20: Glucose Category (One-Hot Encoding, drop_first=True)

20. `glucose_category_Prediabetes` - Prediabetes

- **Categoría eliminada**: `glucose_category_Diabetes` (segunda alfabéticamente)
- **Categoría no presente**: `glucose_category_Normal`

---

## ⚠️ Notas Importantes

### Preprocesamiento Original

El notebook `Preprocessing.ipynb` genera **25 columnas** después del feature engineering y encoding.

### Columnas NO usadas por el modelo (5 columnas eliminadas)

Las siguientes columnas se generan en el preprocesamiento pero **NO** se usan para entrenar el modelo:

1. `work_type_children`
2. `smoking_status_formerly smoked`
3. `age_group_19-35`
4. `bmi_category_Normal`
5. `glucose_category_Diabetes`

### Motivo de la Discrepancia

- El preprocesamiento usa `drop_first=True` en `pd.get_dummies()`, pero no de forma consistente
- El modelo se entrenó con un subset específico de 20 features
- Es necesario **seleccionar explícitamente** estas 20 columnas en el orden correcto antes de hacer predicciones

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
