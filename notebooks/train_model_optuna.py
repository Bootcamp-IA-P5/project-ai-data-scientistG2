"""
CÓDIGO COMPLETO: Threshold Tuning + Ensemble Calibrado
Optimización final para predicción de ictus
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from pathlib import Path
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    recall_score, precision_score, f1_score, accuracy_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')


class ThresholdOptimizerEnsemble:
    """Optimizador de threshold + Ensemble para predicción de ictus"""
    
    def __init__(self, data_path, random_state=42):
        self.data_path = Path(data_path)
        self.random_state = random_state
        self.models = {}
        self.best_threshold = 0.5
        self.best_model = None
        self.results = {}
        
        # Directorios
        self.base_dir = Path(os.getcwd()).parent if 'notebooks' in os.getcwd() else Path(os.getcwd())
        self.models_dir = self.base_dir / "models"
        self.reports_dir = self.base_dir / "reports"
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.reports_dir.mkdir(exist_ok=True, parents=True)
    
    def load_data(self):
        """Cargar datos preprocesados"""
        print("=" * 80)
        print("CARGANDO DATOS")
        print("=" * 80)
        
        with open(self.data_path, 'rb') as f:
            datasets = pickle.load(f)
        
        self.X_train = datasets['original'][0]
        self.X_test = datasets['original'][1]
        self.y_train = datasets['original'][2]
        self.y_test = datasets['original'][3]
        self.scaler = datasets['scaler']
        self.feature_names = datasets['feature_names']
        
        print(f"\n✓ Datos cargados")
        print(f"   Train: {self.X_train.shape}")
        print(f"   Test: {self.X_test.shape}")
        print(f"   Prevalencia ictus: {(self.y_train==1).sum()/len(self.y_train)*100:.2f}%")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_base_models(self):
        """Entrenar modelos base optimizados"""
        print("\n" + "=" * 80)
        print("ENTRENANDO MODELOS BASE")
        print("=" * 80)
        
        # 1. Logistic Regression (mejor del Optuna)
        print("\n🔹 Logistic Regression...")
        self.models['LogisticRegression'] = LogisticRegression(
            C=0.1,
            penalty='l2',
            solver='saga',
            max_iter=2000,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=1
        )
        self.models['LogisticRegression'].fit(self.X_train, self.y_train)
        
        # 2. LightGBM (segundo mejor)
        print("🔹 LightGBM...")
        self.models['LightGBM'] = LGBMClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.3,
            reg_lambda=1.0,
            is_unbalance=True,
            random_state=self.random_state,
            n_jobs=1,
            verbose=-1
        )
        self.models['LightGBM'].fit(self.X_train, self.y_train)
        
        # 3. Logistic Regression Calibrado
        print("🔹 Logistic Regression Calibrado...")
        lr_base = LogisticRegression(
            C=0.1,
            penalty='l2',
            solver='saga',
            max_iter=2000,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=1
        )
        self.models['LR_Calibrated'] = CalibratedClassifierCV(
            lr_base,
            method='sigmoid',
            cv=3
        )
        self.models['LR_Calibrated'].fit(self.X_train, self.y_train)
        
        # 4. LightGBM Calibrado
        print("🔹 LightGBM Calibrado...")
        lgbm_base = LGBMClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.3,
            reg_lambda=1.0,
            is_unbalance=True,
            random_state=self.random_state,
            n_jobs=1,
            verbose=-1
        )
        self.models['LGBM_Calibrated'] = CalibratedClassifierCV(
            lgbm_base,
            method='sigmoid',
            cv=3
        )
        self.models['LGBM_Calibrated'].fit(self.X_train, self.y_train)
        
        # 5. Ensemble Voting (LR + LGBM calibrados)
        print("🔹 Ensemble Voting...")
        self.models['Ensemble_Voting'] = VotingClassifier(
            estimators=[
                ('lr', self.models['LR_Calibrated']),
                ('lgbm', self.models['LGBM_Calibrated'])
            ],
            voting='soft',
            weights=[0.6, 0.4]  # Más peso a LR (mejor precision)
        )
        self.models['Ensemble_Voting'].fit(self.X_train, self.y_train)
        
        print("\n✓ 5 modelos entrenados")
        
        return self.models
    
    def optimize_threshold(self, model, model_name, min_recall=0.80):
        """
        Optimizar threshold para un modelo
        
        Args:
            model: Modelo entrenado
            model_name: Nombre del modelo
            min_recall: Recall mínimo aceptable (default 80%)
        """
        print(f"\n{'─' * 80}")
        print(f"OPTIMIZANDO THRESHOLD: {model_name}")
        print(f"{'─' * 80}")
        
        # Obtener probabilidades
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Probar diferentes thresholds
        thresholds = np.arange(0.05, 0.9, 0.01)
        results = []
        
        for t in thresholds:
            y_pred_t = (y_pred_proba >= t).astype(int)
            
            if y_pred_t.sum() == 0:
                continue
            
            cm = confusion_matrix(self.y_test, y_pred_t)
            tn, fp, fn, tp = cm.ravel()
            
            if (tp + fp) > 0 and (tp + fn) > 0:
                recall = tp / (tp + fn)
                precision = tp / (tp + fp)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                results.append({
                    'threshold': t,
                    'recall': recall,
                    'precision': precision,
                    'f1': f1,
                    'fn': fn,
                    'fp': fp,
                    'tp': tp,
                    'tn': tn,
                    'n_predicted_positive': tp + fp
                })
        
        df_results = pd.DataFrame(results)
        
        # Estrategia 1: Mejor F1 manteniendo recall >= min_recall
        df_filtered = df_results[df_results['recall'] >= min_recall]
        
        if len(df_filtered) > 0:
            best_idx = df_filtered['f1'].idxmax()
            best_t = df_filtered.loc[best_idx, 'threshold']
            strategy = f'Max F1 (Recall >= {min_recall*100:.0f}%)'
        else:
            # Si no hay ninguno con recall >= min_recall, tomar mejor F1
            best_idx = df_results['f1'].idxmax()
            best_t = df_results.loc[best_idx, 'threshold']
            strategy = 'Max F1 (sin restricción)'
        
        best_result = df_results.loc[best_idx]
        
        print(f"\n🎯 Estrategia: {strategy}")
        print(f"   Threshold óptimo:    {best_t:.3f}")
        print(f"   Recall:              {best_result['recall']:.4f} ({best_result['recall']*100:.1f}%)")
        print(f"   Precision:           {best_result['precision']:.4f} ({best_result['precision']*100:.1f}%)")
        print(f"   F1-Score:            {best_result['f1']:.4f}")
        print(f"   Falsos Negativos:    {int(best_result['fn'])} ⚠️")
        print(f"   Falsos Positivos:    {int(best_result['fp'])}")
        print(f"   Casos predichos (+): {int(best_result['n_predicted_positive'])}")
        
        # Métricas médicas
        vpp = best_result['precision']
        vpn = best_result['tn'] / (best_result['tn'] + best_result['fn'])
        sensibilidad = best_result['recall']
        especificidad = best_result['tn'] / (best_result['tn'] + best_result['fp'])
        
        print(f"\n📊 Métricas Médicas:")
        print(f"   Sensibilidad:        {sensibilidad:.4f}")
        print(f"   Especificidad:       {especificidad:.4f}")
        print(f"   VPP:                 {vpp:.4f}")
        print(f"   VPN:                 {vpn:.4f}")
        
        return best_t, df_results, best_result
    
    def evaluate_all_models(self):
        """Evaluar todos los modelos con threshold optimizado"""
        print("\n" + "=" * 80)
        print("EVALUACIÓN COMPLETA DE MODELOS")
        print("=" * 80)
        
        for model_name, model in self.models.items():
            best_t, df_thresh, best_result = self.optimize_threshold(model, model_name, min_recall=0.80)
            
            self.results[model_name] = {
                'model': model,
                'best_threshold': best_t,
                'threshold_results': df_thresh,
                'best_result': best_result,
                'recall': best_result['recall'],
                'precision': best_result['precision'],
                'f1': best_result['f1'],
                'fn': int(best_result['fn']),
                'fp': int(best_result['fp']),
                'tp': int(best_result['tp']),
                'tn': int(best_result['tn'])
            }
        
        return self.results
    
    def get_best_model(self):
        """Seleccionar el mejor modelo"""
        print("\n" + "=" * 80)
        print("SELECCIÓN DEL MEJOR MODELO")
        print("=" * 80)
        
        comparison = []
        for name, data in self.results.items():
            comparison.append({
                'Modelo': name,
                'Threshold': data['best_threshold'],
                'Recall': data['recall'],
                'Precision': data['precision'],
                'F1': data['f1'],
                'FN': data['fn'],
                'FP': data['fp'],
                'Score': data['recall'] * 0.6 + data['precision'] * 0.4
            })
        
        df_comp = pd.DataFrame(comparison).sort_values('Score', ascending=False)
        
        print("\n📊 COMPARACIÓN DE TODOS LOS MODELOS:")
        print(df_comp.to_string(index=False))
        
        best_model_name = df_comp.iloc[0]['Modelo']
        self.best_model_name = best_model_name
        self.best_model = self.results[best_model_name]['model']
        self.best_threshold = self.results[best_model_name]['best_threshold']
        
        print(f"\n🏆 MEJOR MODELO: {best_model_name}")
        print(f"   Threshold:   {self.best_threshold:.3f}")
        print(f"   Recall:      {df_comp.iloc[0]['Recall']:.4f}")
        print(f"   Precision:   {df_comp.iloc[0]['Precision']:.4f}")
        print(f"   F1-Score:    {df_comp.iloc[0]['F1']:.4f}")
        print(f"   FN:          {int(df_comp.iloc[0]['FN'])} casos perdidos")
        
        return self.best_model, self.best_model_name, df_comp
    
    def create_super_ensemble(self):
        """Crear super-ensemble con los 3 mejores modelos"""
        print("\n" + "=" * 80)
        print("CREANDO SUPER-ENSEMBLE")
        print("=" * 80)
        
        # Obtener top 3 modelos
        comparison = []
        for name, data in self.results.items():
            comparison.append({
                'name': name,
                'score': data['recall'] * 0.6 + data['precision'] * 0.4,
                'model': data['model']
            })
        
        top3 = sorted(comparison, key=lambda x: x['score'], reverse=True)[:3]
        
        print(f"\nTop 3 modelos seleccionados:")
        for i, model_info in enumerate(top3, 1):
            print(f"  {i}. {model_info['name']}")
        
        # Crear super-ensemble
        super_ensemble = VotingClassifier(
            estimators=[
                (f'model{i}', model_info['model'])
                for i, model_info in enumerate(top3)
            ],
            voting='soft',
            weights=[0.4, 0.3, 0.3]  # Más peso al mejor
        )
        
        super_ensemble.fit(self.X_train, self.y_train)
        
        # Optimizar threshold
        best_t, df_thresh, best_result = self.optimize_threshold(
            super_ensemble, 
            'Super_Ensemble', 
            min_recall=0.80
        )
        
        # Guardar resultados
        self.results['Super_Ensemble'] = {
            'model': super_ensemble,
            'best_threshold': best_t,
            'threshold_results': df_thresh,
            'best_result': best_result,
            'recall': best_result['recall'],
            'precision': best_result['precision'],
            'f1': best_result['f1'],
            'fn': int(best_result['fn']),
            'fp': int(best_result['fp']),
            'tp': int(best_result['tp']),
            'tn': int(best_result['tn'])
        }
        
        print(f"\n✅ Super-Ensemble creado y optimizado")
        
        return super_ensemble
    def visualize_results(self):
        """Crear visualizaciones completas"""
        print("\n" + "=" * 80)
        print("GENERANDO VISUALIZACIONES")
        print("=" * 80)
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Threshold optimization del mejor modelo
        ax1 = fig.add_subplot(gs[0, 0])
        df_thresh = self.results[self.best_model_name]['threshold_results']
        ax1.plot(df_thresh['threshold'], df_thresh['recall'], 'o-', label='Recall', linewidth=2, markersize=4)
        ax1.plot(df_thresh['threshold'], df_thresh['precision'], 's-', label='Precision', linewidth=2, markersize=4)
        ax1.plot(df_thresh['threshold'], df_thresh['f1'], '^-', label='F1-Score', linewidth=2, markersize=4)
        ax1.axvline(x=self.best_threshold, color='red', linestyle='--', 
                   label=f'Óptimo: {self.best_threshold:.3f}', linewidth=2)
        ax1.axhline(y=0.80, color='orange', linestyle=':', alpha=0.5)
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title(f'Optimización Threshold - {self.best_model_name}', fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)
        
        # 2. Trade-off FN vs FP
        ax2 = fig.add_subplot(gs[0, 1])
        ax2_twin = ax2.twinx()
        ax2.plot(df_thresh['threshold'], df_thresh['fn'], 'ro-', 
                label='Falsos Negativos', linewidth=2, markersize=5)
        ax2_twin.plot(df_thresh['threshold'], df_thresh['fp'], 'bs-', 
                     label='Falsos Positivos', linewidth=2, markersize=5)
        ax2.axvline(x=self.best_threshold, color='green', linestyle='--', alpha=0.7, linewidth=2)
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Falsos Negativos', color='red')
        ax2_twin.set_ylabel('Falsos Positivos', color='blue')
        ax2.set_title('Trade-off: FN vs FP', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2_twin.tick_params(axis='y', labelcolor='blue')
        ax2.grid(alpha=0.3)
        
        # 3. Comparación de modelos
        ax3 = fig.add_subplot(gs[0, 2])
        comparison_data = []
        for name in self.results.keys():
            comparison_data.append({
                'Modelo': name[:15],
                'Recall': self.results[name]['recall'],
                'Precision': self.results[name]['precision'],
                'F1': self.results[name]['f1']
            })
        df_plot = pd.DataFrame(comparison_data).set_index('Modelo')
        df_plot.plot(kind='bar', ax=ax3, rot=45)
        ax3.set_ylabel('Score')
        ax3.set_title('Comparación de Modelos', fontweight='bold')
        ax3.legend(fontsize=8, loc='lower right')
        ax3.grid(alpha=0.3, axis='y')
        
        # 4. Matriz de confusión del mejor modelo
        ax4 = fig.add_subplot(gs[1, 0])
        best_res = self.results[self.best_model_name]['best_result']
        cm = np.array([[int(best_res['tn']), int(best_res['fp'])], 
               [int(best_res['fn']), int(best_res['tp'])]], dtype=int)  # ← Forzar int
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax4,
                   xticklabels=['No Ictus', 'Ictus'],
                   yticklabels=['No Ictus', 'Ictus'],
                   cbar_kws={'label': 'Frecuencia'})
        ax4.set_title(f'Matriz de Confusión - {self.best_model_name}', fontweight='bold')
        ax4.set_ylabel('Real')
        ax4.set_xlabel('Predicho')
        
        # 5. Precision vs Recall scatter
        ax5 = fig.add_subplot(gs[1, 1])
        for name in self.results.keys():
            ax5.scatter(self.results[name]['recall'], self.results[name]['precision'], 
                       s=200, alpha=0.7, label=name[:15])
        ax5.axhline(y=0.20, color='orange', linestyle='--', alpha=0.5, label='Target Precision')
        ax5.axvline(x=0.80, color='red', linestyle='--', alpha=0.5, label='Target Recall')
        ax5.set_xlabel('Recall')
        ax5.set_ylabel('Precision')
        ax5.set_title('Trade-off Precision vs Recall', fontweight='bold')
        ax5.legend(fontsize=7, loc='lower left')
        ax5.grid(alpha=0.3)
        
        # 6. Curvas ROC
        ax6 = fig.add_subplot(gs[1, 2])
        for name, data in self.results.items():
            model = data['model']
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            ax6.plot(fpr, tpr, label=f'{name[:12]} (AUC={auc:.3f})', linewidth=2)
        ax6.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        ax6.set_xlabel('False Positive Rate')
        ax6.set_ylabel('True Positive Rate')
        ax6.set_title('Curvas ROC', fontweight='bold')
        ax6.legend(fontsize=7, loc='lower right')
        ax6.grid(alpha=0.3)
        
        # 7. Curvas Precision-Recall
        ax7 = fig.add_subplot(gs[2, 0])
        for name, data in self.results.items():
            model = data['model']
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            precision_curve, recall_curve, _ = precision_recall_curve(self.y_test, y_pred_proba)
            auc_pr = average_precision_score(self.y_test, y_pred_proba)
            ax7.plot(recall_curve, precision_curve, label=f'{name[:12]} (AP={auc_pr:.3f})', linewidth=2)
        baseline = (self.y_test == 1).sum() / len(self.y_test)
        ax7.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.3f})', linewidth=1)
        ax7.set_xlabel('Recall')
        ax7.set_ylabel('Precision')
        ax7.set_title('Curvas Precision-Recall', fontweight='bold')
        ax7.legend(fontsize=7, loc='lower left')
        ax7.grid(alpha=0.3)
        
        # 8. Distribución de probabilidades
        ax8 = fig.add_subplot(gs[2, 1])
        y_pred_proba_best = self.best_model.predict_proba(self.X_test)[:, 1]
        ax8.hist(y_pred_proba_best[self.y_test == 0], bins=30, alpha=0.6, 
                label='No Ictus', color='green', edgecolor='black')
        ax8.hist(y_pred_proba_best[self.y_test == 1], bins=30, alpha=0.6, 
                label='Ictus', color='red', edgecolor='black')
        ax8.axvline(x=self.best_threshold, color='blue', linestyle='--', 
                   label=f'Threshold: {self.best_threshold:.3f}', linewidth=2)
        ax8.set_xlabel('Probabilidad Predicha')
        ax8.set_ylabel('Frecuencia')
        ax8.set_title('Distribución de Probabilidades', fontweight='bold')
        ax8.legend()
        ax8.grid(alpha=0.3, axis='y')
        
        # 9. Tabla resumen
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        summary_text = f"""
RESUMEN DEL MEJOR MODELO
{'='*40}

Modelo: {self.best_model_name}
Threshold: {self.best_threshold:.3f}

MÉTRICAS:
  Recall:      {self.results[self.best_model_name]['recall']:.4f}
  Precision:   {self.results[self.best_model_name]['precision']:.4f}
  F1-Score:    {self.results[self.best_model_name]['f1']:.4f}

CONFUSIÓN:
  TP: {self.results[self.best_model_name]['tp']}
  FP: {self.results[self.best_model_name]['fp']}
  TN: {self.results[self.best_model_name]['tn']}
  FN: {self.results[self.best_model_name]['fn']} ⚠️

INTERPRETACIÓN:
  • De {self.results[self.best_model_name]['tp'] + self.results[self.best_model_name]['fn']} casos con ictus
    detectamos {self.results[self.best_model_name]['tp']}
  • Perdemos {self.results[self.best_model_name]['fn']} casos
  • Generamos {self.results[self.best_model_name]['fp']} falsos positivos
        """
        
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle('ANÁLISIS COMPLETO - OPTIMIZACIÓN DE THRESHOLD + ENSEMBLE', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Guardar
        output_path = self.reports_dir / 'threshold_ensemble_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Visualización guardada en: {output_path}")
        plt.show()
    
    def save_best_model(self):
        """Guardar el mejor modelo"""
        print("\n" + "=" * 80)
        print("GUARDANDO MODELO FINAL")
        print("=" * 80)
        
        model_path = self.models_dir / 'ictus_model_threshold_optimized.pkl'
        
        best_res = self.results[self.best_model_name]
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'threshold': self.best_threshold,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metrics': {
                'recall': best_res['recall'],
                'precision': best_res['precision'],
                'f1': best_res['f1'],
                'tp': best_res['tp'],
                'fp': best_res['fp'],
                'tn': best_res['tn'],
                'fn': best_res['fn']
            },
            'threshold_results': best_res['threshold_results']
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✅ Modelo guardado en: {model_path}")
        
        # Guardar reporte
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f'threshold_optimization_report_{timestamp}.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("REPORTE - OPTIMIZACIÓN DE THRESHOLD + ENSEMBLE\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Mejor Modelo: {self.best_model_name}\n")
            f.write(f"Threshold Optimizado: {self.best_threshold:.3f}\n\n")
            
            f.write("MÉTRICAS:\n")
            f.write(f"  Recall:      {best_res['recall']:.4f}\n")
            f.write(f"  Precision:   {best_res['precision']:.4f}\n")
            f.write(f"  F1-Score:    {best_res['f1']:.4f}\n\n")
            
            f.write("MATRIZ DE CONFUSIÓN:\n")
            f.write(f"  TN: {best_res['tn']}  FP: {best_res['fp']}\n")
            f.write(f"  FN: {best_res['fn']}  TP: {best_res['tp']}\n\n")
            
            f.write("INTERPRETACIÓN:\n")
            f.write(f"  • Casos con ictus detectados: {best_res['tp']} de {best_res['tp']+best_res['fn']}\n")
            f.write(f"  • Casos perdidos (FN): {best_res['fn']}\n")
            f.write(f"  • Falsos positivos (FP): {best_res['fp']}\n")
        
        print(f"✅ Reporte guardado en: {report_path}")
        
        print("\n✅ Guardado completado")


def main():
    """Función principal"""
    print("\n" + "=" * 80)
    print("OPTIMIZACIÓN FINAL - THRESHOLD + ENSEMBLE")
    print("=" * 80)
    
    # Configuración
    base_dir = Path(os.getcwd()).parent if 'notebooks' in os.getcwd() else Path(os.getcwd())
    data_path = base_dir / 'data' / 'preprocessed_dataSin.pkl'
    
    # Crear optimizador
    optimizer = ThresholdOptimizerEnsemble(data_path=data_path, random_state=42)
    
    # Pipeline completo
    optimizer.load_data()
    optimizer.train_base_models()
    optimizer.evaluate_all_models()
    optimizer.get_best_model()
    optimizer.visualize_results()
    optimizer.save_best_model()
    
    print("\n" + "=" * 80)
    print("✅ OPTIMIZACIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 80)
    
    return optimizer


if __name__ == "__main__":
    optimizer = main()