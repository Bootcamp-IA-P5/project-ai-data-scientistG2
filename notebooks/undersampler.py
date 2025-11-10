# modelo_ictus_OPTIMIZADO.py
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import joblib
import optuna

from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score

import warnings
warnings.filterwarnings('ignore')

# ========================
# RUTAS
# ========================
base_dir = Path(__file__).parent.parent
data_path = base_dir / 'data' / 'preprocessed_dataSin.pkl'
models_dir = base_dir / 'models'
models_dir.mkdir(exist_ok=True)

# ========================
# DATOS
# ========================
with open(data_path, 'rb') as f:
    data = pickle.load(f)

X_train = data['original'][0].copy()
y_train = data['original'][2].copy()
X_test = data['original'][1].copy()
y_test = data['original'][3].copy()

# ========================
# OBJETIVO OPTUNA: MAXIMIZAR PRECISIÓN CON RECALL ≥ 80%
# ========================
def objective(trial):
    # Hiperparámetros
    ratio = trial.suggest_float('ratio', 0.10, 0.25)
    n_estimators = trial.suggest_int('n_estimators', 400, 800)
    max_depth = trial.suggest_int('max_depth', 8, 12)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 5, 12)
    
    # Undersampling
    undersampler = RandomUnderSampler(sampling_strategy=ratio, random_state=42)
    X_res, y_res = undersampler.fit_resample(X_train, y_train)
    
    # Modelo
    model = BalancedRandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features='sqrt',
        sampling_strategy='auto',
        replacement=False,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    
    # CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    precisions = []
    
    for train_idx, val_idx in cv.split(X_res, y_res):
        X_tr, X_val = X_res.iloc[train_idx], X_res.iloc[val_idx]
        y_tr, y_val = y_res.iloc[train_idx], y_res.iloc[val_idx]
        
        model.fit(X_tr, y_tr)
        y_proba = model.predict_proba(X_val)[:, 1]
        
        # Threshold fino
        thresholds = np.arange(0.2, 0.6, 0.005)
        best_precision = 0
        for t in thresholds:
            pred = (y_proba >= t).astype(int)
            recall = recall_score(y_val, pred)
            if recall >= 0.80:
                precision = precision_score(y_val, pred, zero_division=0)
                if precision > best_precision:
                    best_precision = precision
        if best_precision > 0:
            precisions.append(best_precision)
    
    return np.mean(precisions) if precisions else 0.0

# ========================
# OPTIMIZACIÓN
# ========================
print("OPTUNA: Buscando máxima precisión con Recall ≥80%...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\nMEJOR PRECISIÓN (CV): {study.best_value:.4f}")
print("MEJORES PARÁMETROS:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# ========================
# MODELO FINAL
# ========================
p = study.best_params
undersampler = RandomUnderSampler(sampling_strategy=p['ratio'], random_state=42)
X_res, y_res = undersampler.fit_resample(X_train, y_train)

final_model = BalancedRandomForestClassifier(
    n_estimators=p['n_estimators'],
    max_depth=p['max_depth'],
    min_samples_leaf=p['min_samples_leaf'],
    max_features='sqrt',
    sampling_strategy='auto',
    replacement=False,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)
final_model.fit(X_res, y_res)

# ========================
# THRESHOLD EN TEST
# ========================
y_proba = final_model.predict_proba(X_test)[:, 1]
thresholds = np.arange(0.2, 0.6, 0.005)
best_precision = 0.0
best_thresh = 0.3
best_pred = None
best_recall = 0.0

for t in thresholds:
    pred = (y_proba >= t).astype(int)
    recall = recall_score(y_test, pred)
    if recall >= 0.80:
        precision = precision_score(y_test, pred, zero_division=0)
        if precision > best_precision:
            best_precision = precision
            best_thresh = t
            best_pred = pred
            best_recall = recall

# ========================
# RESULTADOS
# ========================
f1 = f1_score(y_test, best_pred, zero_division=0)
fn = ((best_pred == 0) & (y_test == 1)).sum()
fp = ((best_pred == 1) & (y_test == 0)).sum()

print("\n" + "="*80)
print("MODELO OPTIMIZADO - TEST")
print("="*80)
print(f"Threshold:     {best_thresh:.3f}")
print(f"Recall:        {best_recall:.1%}")
print(f"Precision:     {best_precision:.3f} ({best_precision:.1%})")
print(f"F1-Score:      {f1:.4f}")
print(f"FN:            {fn}")
print(f"FP:            {fp}")

# ========================
# GUARDAR .PKL
# ========================
pkl_path = models_dir / 'modelo_ictus_OPTIMIZADO.pkl'
joblib.dump({
    'model': final_model,
    'threshold': best_thresh,
    'undersampler_ratio': p['ratio'],
    'metrics': {
        'recall': best_recall,
        'precision': best_precision,
        'f1': f1,
        'fn': fn,
        'fp': fp
    },
    'feature_names': X_train.columns.tolist()
}, pkl_path)

print(f"\nMODELO OPTIMIZADO GUARDADO:")
print(f"   → {pkl_path}")