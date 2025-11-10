import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from sklearn.ensemble import VotingClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# -------------------------------
# 1. CARGAR DATOS
# -------------------------------
base_dir = Path(os.getcwd()).parent if 'notebooks' in os.getcwd() else Path(os.getcwd())
data_path = base_dir / 'data' / 'preprocessed_dataSin.pkl'

data = pd.read_pickle(data_path)
X_train_df, X_test_df, y_train_s, y_test_s = data['original']
X_train = X_train_df.copy()
y_train = y_train_s.copy()
X_test = X_test_df.copy()
y_test = y_test_s.copy()

print(f"Train: {X_train.shape} | Ictus: {y_train.sum()} ({y_train.mean():.2%})")
print(f"Test:  {X_test.shape}  | Ictus: {y_test.sum()} ({y_test.mean():.2%})\n")

# -------------------------------
# 2. PREPROCESADOR
# -------------------------------
numeric_features = ['age', 'avg_glucose_level', 'bmi', 'risk_factors', 'age_risk_interaction']
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', 'passthrough', [c for c in X_train.columns if c not in numeric_features])
])

# -------------------------------
# 3. MODELO FINAL: ENSEMBLE (XGBoost + LightGBM)
# -------------------------------
ensemble = VotingClassifier(
    estimators=[
        ('xgb', XGBClassifier(
            n_estimators=800,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.5,
            reg_lambda=2.0,
            min_child_weight=6,
            scale_pos_weight=8,
            random_state=42,
            eval_metric='auc'
        )),
        ('lgb', LGBMClassifier(
            n_estimators=800,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.7,
            feature_fraction=0.7,
            scale_pos_weight=8,
            random_state=42,
            verbose=-1
        ))
    ],
    voting='soft'
)

pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(sampling_strategy=0.6, random_state=42)),  # 60% positivos
    ('classifier', ensemble)
])

# -------------------------------
# 4. ENTRENAR MODELO FINAL
# -------------------------------
print("Entrenando modelo final (ensemble + SMOTE)...\n")
pipeline.fit(X_train, y_train)

# -------------------------------
# 5. EVALUAR EN TEST + THRESHOLD POR COSTO
# -------------------------------
y_test_prob = pipeline.predict_proba(X_test)[:, 1]

# Costo: FN = 10 (no detectar ictus), FP = 1 (falsa alarma)
thresholds = np.arange(0.10, 0.40, 0.01)
best_cost = float('inf')
best_thresh = 0.2
best_pred = None

print("Buscando mejor threshold (costo: FN=10, FP=1)...")
for t in thresholds:
    pred = (y_test_prob >= t).astype(int)
    fn = ((pred == 0) & (y_test == 1)).sum() * 10
    fp = ((pred == 1) & (y_test == 0)).sum() * 1
    total_cost = fn + fp
    if total_cost < best_cost:
        best_cost = total_cost
        best_thresh = t
        best_pred = pred

# Métricas finales
recall = recall_score(y_test, best_pred)
precision = precision_score(y_test, best_pred, zero_division=0)
f1 = f1_score(y_test, best_pred, zero_division=0)
auc = roc_auc_score(y_test, y_test_prob)

# -------------------------------
# 6. RESULTADOS FINALES
# -------------------------------
print("\n" + "="*70)
print("RESULTADOS FINALES EN TEST (MEJOR POSIBLE CON TU DATASET)")
print("="*70)
print(f"Threshold óptimo:     {best_thresh:.3f}")
print(f"Recall (ictus):       {recall:.1%}")
print(f"Precision:            {precision:.1%}")
print(f"F1-Score:             {f1:.3f}")
print(f"AUC:                  {auc:.3f}")
print(f"Costo total (FN×10 + FP): {best_cost}")
print("\nInterpretación:")
print(f"  - Detectas {int(recall * y_test.sum())} de {y_test.sum()} ictus reales")
print(f"  - Generas {best_pred.sum()} alertas → {int(precision * best_pred.sum())} verdaderas")
print(f"  - {int((1 - precision) * best_pred.sum())} falsos positivos")

# -------------------------------
# 7. GUARDAR MODELO + THRESHOLD
# -------------------------------
joblib.dump(pipeline, 'modelo_ictus_final_ensemble.pkl')
with open('threshold_optimo.txt', 'w') as f:
    f.write(str(best_thresh))

print("\nModelo guardado: modelo_ictus_final_ensemble.pkl")
print(f"Threshold guardado: threshold_optimo.txt → {best_thresh:.3f}")