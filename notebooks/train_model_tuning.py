from pathlib import Path
from datetime import datetime
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score

warnings.filterwarnings('ignore')


class ThresholdOptimizerEnsemble:
    def __init__(self, random_state: int = 42, models_dir: Path | str = './models', reports_dir: Path | str = './reports'):
        self.random_state = random_state
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_threshold = None
        self.best_model_name = None

    # ---------------------- factories ----------------------
    def _build_lr(self):
        return LogisticRegression(C=0.1, penalty='l2', solver='saga', max_iter=2000,
                                  class_weight='balanced', random_state=self.random_state, n_jobs=1)

    def _build_lgbm(self):
        return LGBMClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, num_leaves=31,
                              min_child_samples=20, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.3,
                              reg_lambda=1.0, is_unbalance=True, random_state=self.random_state, n_jobs=1, verbose=-1)

    # ---------------------- training base models ----------------------
    def train_base_models(self, X_train, y_train, calibrate=True):
        lr = self._build_lr()
        lgbm = self._build_lgbm()
        lr.fit(X_train, y_train)
        lgbm.fit(X_train, y_train)
        self.models = {'LogisticRegression': lr, 'LightGBM': lgbm}

        if calibrate:
            lr_cal = CalibratedClassifierCV(self._build_lr(), method='sigmoid', cv=3)
            lgbm_cal = CalibratedClassifierCV(self._build_lgbm(), method='sigmoid', cv=3)
            lr_cal.fit(X_train, y_train)
            lgbm_cal.fit(X_train, y_train)
            self.models['LR_Calibrated'] = lr_cal
            self.models['LGBM_Calibrated'] = lgbm_cal

        # Ensemble
        estimators = []
        if 'LR_Calibrated' in self.models and 'LGBM_Calibrated' in self.models:
            estimators = [('lr', clone(self.models['LR_Calibrated'])), ('lgbm', clone(self.models['LGBM_Calibrated']))]
            weights = [0.6, 0.4]
        else:
            estimators = [('lr', clone(self.models['LogisticRegression'])), ('lgbm', clone(self.models['LightGBM']))]
            weights = [0.5, 0.5]

        ensemble = VotingClassifier(estimators=estimators, voting='soft', weights=weights)
        ensemble.fit(X_train, y_train)
        self.models['Ensemble_Voting'] = ensemble
        return self.models

    # ---------------------- threshold search with CV ----------------------
    def find_best_threshold_cv(self, model, X, y, thresholds=None, cv_splits=5, scoring='f1', min_recall=None):
        if thresholds is None:
            thresholds = np.arange(0.05, 0.9, 0.01)

        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        scores = {t: {'f1': [], 'precision': [], 'recall': []} for t in thresholds}

        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = (X.iloc[train_idx], X.iloc[val_idx]) if isinstance(X, pd.DataFrame) else (X[train_idx], X[val_idx])
            y_tr, y_val = (y.iloc[train_idx], y.iloc[val_idx]) if isinstance(y, pd.Series) else (y[train_idx], y[val_idx])

            m = clone(model)
            m.fit(X_tr, y_tr)
            proba = m.predict_proba(X_val)[:, 1]

            for t in thresholds:
                pred = (proba >= t).astype(int)
                scores[t]['f1'].append(f1_score(y_val, pred) if pred.sum() > 0 else 0.0)
                scores[t]['precision'].append(precision_score(y_val, pred, zero_division=0))
                scores[t]['recall'].append(recall_score(y_val, pred, zero_division=0))

        # Aggregate
        rows = []
        for t in thresholds:
            mean_f1 = np.mean(scores[t]['f1'])
            mean_prec = np.mean(scores[t]['precision'])
            mean_rec = np.mean(scores[t]['recall'])
            std_f1 = np.std(scores[t]['f1'])
            rows.append({'threshold': t, 'mean_f1': mean_f1, 'std_f1': std_f1, 'mean_precision': mean_prec, 'mean_recall': mean_rec})

        df = pd.DataFrame(rows)

        if (min_recall is not None) and (df[df['mean_recall'] >= min_recall].shape[0] > 0):
            df_filt = df[df['mean_recall'] >= min_recall]
            best_row = df_filt.loc[df_filt['mean_f1'].idxmax()]
            strategy = f"Max mean F1 with mean_recall >= {min_recall}"
        else:
            best_row = df.loc[df['mean_f1'].idxmax()]
            strategy = 'Max mean F1 (no recall constraint)'

        return float(best_row['threshold']), df, strategy

    # ---------------------- evaluate with threshold ----------------------
    def evaluate_with_threshold(self, model, X_test, y_test, threshold=0.5, X_train=None, y_train=None):
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        metrics = {
            'threshold': threshold,
            'precision': precision_score(y_test, pred, zero_division=0),
            'recall': recall_score(y_test, pred, zero_division=0),
            'f1': f1_score(y_test, pred, zero_division=0),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
            'auc': roc_auc_score(y_test, proba)
        }

        if X_train is not None and y_train is not None:
            train_proba = model.predict_proba(X_train)[:, 1]
            train_pred = (train_proba >= threshold).astype(int)
            train_recall = recall_score(y_train, train_pred, zero_division=0)
            metrics['recall_train'] = train_recall
            metrics['overfit_recall_diff'] = train_recall - metrics['recall']

        return metrics

    # ---------------------- visualization ----------------------
    def plot_threshold_cv(self, df_thresh, model_name):
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='threshold', y='mean_f1', data=df_thresh, label='Mean F1')
        sns.lineplot(x='threshold', y='mean_precision', data=df_thresh, label='Mean Precision')
        sns.lineplot(x='threshold', y='mean_recall', data=df_thresh, label='Mean Recall')
        plt.title(f'Threshold tuning CV for {model_name}')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.legend()

        png_path = self.reports_dir / f"threshold_cv_{model_name}.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"Gráfica guardada en: {png_path}")

        plt.show()

    # ---------------------- full pipeline ----------------------
    def run_pipeline(self, X_train, y_train, X_test, y_test, calibrate=True, cv_splits=5, thresholds=None, min_recall=0.8):
        base_models = self.train_base_models(X_train, y_train, calibrate=calibrate)

        for name, model in base_models.items():
            if not hasattr(model, 'predict_proba'):
                continue

            best_t, df_thresh, strategy = self.find_best_threshold_cv(model, X_train, y_train, thresholds=thresholds, cv_splits=cv_splits, min_recall=min_recall)
            model_final = clone(model)
            model_final.fit(X_train, y_train)
            metrics = self.evaluate_with_threshold(model_final, X_test, y_test, threshold=best_t, X_train=X_train, y_train=y_train)
            metrics['cv_strategy'] = strategy
            metrics['threshold_cv_results'] = df_thresh
            self.results[name] = {'model': model_final, 'best_threshold': best_t, 'metrics': metrics}

            self.plot_threshold_cv(df_thresh, name)

        # Super ensemble logic (top 3)
        ranking = [(n, info['metrics']['recall']*0.6 + info['metrics']['precision']*0.4) for n, info in self.results.items()]
        top3_names = [n for n, _ in sorted(ranking, key=lambda x: x[1], reverse=True)[:3]]

        estimators, weights = [], []
        for i, name in enumerate(top3_names):
            estimators.append((f'm{i}', clone(self.results[name]['model'])))
            weights.append([0.4, 0.3, 0.3][i] if i<3 else 0.3)

        if estimators:
            super_ens = VotingClassifier(estimators=estimators, voting='soft', weights=weights)
            super_ens.fit(X_train, y_train)
            best_t, df_thresh, strategy = self.find_best_threshold_cv(super_ens, X_train, y_train, thresholds=thresholds, cv_splits=cv_splits, min_recall=min_recall)
            super_final = clone(super_ens)
            super_final.fit(X_train, y_train)
            metrics = self.evaluate_with_threshold(super_final, X_test, y_test, threshold=best_t, X_train=X_train, y_train=y_train)
            metrics['cv_strategy'] = strategy
            metrics['threshold_cv_results'] = df_thresh
            self.results['Super_Ensemble'] = {'model': super_final, 'best_threshold': best_t, 'metrics': metrics}
            self.plot_threshold_cv(df_thresh, 'Super_Ensemble')

        # Guardamos mejor modelo según recall
        best_name = max(self.results.items(), key=lambda kv: kv[1]['metrics']['recall'])[0]
        self.best_model_name = best_name
        self.best_model = self.results[best_name]['model']
        self.best_threshold = self.results[best_name]['best_threshold']

        return {**self.results, "best_model_name": self.best_model_name}  # para tests

    # ---------------------- saving ----------------------
    def save_best(self, filename_prefix='ictus_model'):
        if self.best_model is None:
            raise RuntimeError('No hay modelo entrenado para guardar')
        model_path = self.models_dir / f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        payload = {'model': self.best_model, 'best_threshold': self.best_threshold, 'best_model_name': self.best_model_name,
                   'results': self.results, 'created_at': datetime.now().isoformat(), 'version': '1.0.0'}
        with open(model_path, 'wb') as f:
            pickle.dump(payload, f)
        return model_path

     # ---------------------- método para tests ----------------------
    def get_model(self, model_name, X, y):
        """Compatibilidad con tests: devuelve un modelo base entrenado"""
        if model_name.lower() == "lr":
            model = self._build_lr()
        elif model_name.lower() == "lgbm":
            model = self._build_lgbm()
        else:
            raise ValueError(f"Modelo desconocido: {model_name}")
        model.fit(X, y)
        return model

# ---------------------- Demo / CLI ----------------------
def demo_run():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=3000, n_features=20, n_informative=6, n_redundant=2,
                               n_clusters_per_class=2, weights=[0.95, 0.05], flip_y=0.01, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    opt = ThresholdOptimizerEnsemble()
    results = opt.run_pipeline(X_train, y_train, X_test, y_test, calibrate=True, cv_splits=4, min_recall=0.8)

    print('\n--- Model Summary ---')
    for name, info in results.items():
        if name == "best_model_name":
            continue
        m = info['metrics']
        print(f"{name}: threshold={info['best_threshold']:.3f} recall={m['recall']:.3f} precision={m['precision']:.3f} f1={m['f1']:.3f} overfit_recall_diff={m.get('overfit_recall_diff', 0):.3f}")

    saved = opt.save_best()
    print(f'Guardado mejor modelo en: {saved}')


if __name__ == '__main__':
    demo_run()