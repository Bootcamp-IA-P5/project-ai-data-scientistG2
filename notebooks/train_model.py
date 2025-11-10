"""
Script de entrenamiento de modelos para predicción de ictus
Adaptado para dataset de ictus con técnicas especializadas para desbalanceo extremo
Incluye optimización de hiperparámetros y técnicas avanzadas
"""
import pandas as pd
import numpy as np
import pickle
import json
import os
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score
)

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Manejo de desbalanceo
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

import warnings
warnings.filterwarnings('ignore')


class IctusPredictor:
    """Clase para entrenar y evaluar modelos de predicción de ictus"""
    
    def __init__(self, data_path, random_state=42):
        """
        Inicializar el predictor
        
        Args:
            data_path: Ruta al archivo PKL con datos preprocesados
            random_state: Semilla para reproducibilidad
        """
        self.data_path = Path(data_path)
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
        # Directorios
        self.base_dir = Path(os.getcwd()).parent if 'notebooks' in os.getcwd() else Path(os.getcwd())
        self.models_dir = self.base_dir / "models"
        self.reports_dir = self.base_dir / "reports"
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.reports_dir.mkdir(exist_ok=True, parents=True)
        
    def load_preprocessed_data(self):
        """Cargar datos preprocesados desde PKL"""
        print("=" * 80)
        print("CARGANDO DATOS PREPROCESADOS")
        print("=" * 80)
        
        with open(self.data_path, 'rb') as f:
            datasets = pickle.load(f)
        
        self.X_train_original = datasets['original'][0]
        self.X_test = datasets['original'][1]
        self.y_train_original = datasets['original'][2]
        self.y_test = datasets['original'][3]
        self.scaler = datasets['scaler']
        self.feature_names = datasets['feature_names']
        
        print(f"\n✓ Datos cargados exitosamente")
        print(f"   Train: {self.X_train_original.shape}")
        print(f"   Test: {self.X_test.shape}")
        print(f"   Features: {len(self.feature_names)}")
        
        # Distribución de clases
        print(f"\n📊 Distribución de clases en Train:")
        train_dist = pd.Series(self.y_train_original).value_counts().sort_index()
        for cls, count in train_dist.items():
            percentage = (count / len(self.y_train_original)) * 100
            print(f"   Clase {int(cls)}: {count} ({percentage:.2f}%)")
        
        prevalence = (self.y_train_original == 1).sum() / len(self.y_train_original)
        print(f"\n   Prevalencia de ictus: {prevalence*100:.2f}%")
        print(f"   ⚠️ Dataset muy desbalanceado (realista para problema médico)")
        
        return self.X_train_original, self.X_test, self.y_train_original, self.y_test
    
    def apply_sampling_strategies(self, strategy='conservative'):
        """
        Aplicar diferentes estrategias de sampling
        
        Args:
            strategy: 'none', 'conservative', 'moderate', 'aggressive'
        """
        print("\n" + "=" * 80)
        print(f"APLICANDO ESTRATEGIA DE SAMPLING: {strategy.upper()}")
        print("=" * 80)
        
        self.sampling_datasets = {}
        
        # 1. Sin sampling (solo datos originales)
        self.sampling_datasets['NoSampling'] = (
            self.X_train_original, self.y_train_original
        )
        print(f"\n✓ NoSampling: {self.X_train_original.shape[0]} muestras")
        
        # 2. RandomUnderSampler (reducir clase mayoritaria)
        print(f"\n✓ RandomUnderSampler:")
        under_ratios = {'conservative': 0.2, 'moderate': 0.15, 'aggressive': 0.1}
        under_ratio = under_ratios.get(strategy, 0.15)
        
        undersampler = RandomUnderSampler(
            sampling_strategy=under_ratio,  # Ratio minoritaria/mayoritaria
            random_state=self.random_state
        )
        X_under, y_under = undersampler.fit_resample(self.X_train_original, self.y_train_original)
        self.sampling_datasets['UnderSampling'] = (X_under, y_under)
        print(f"   Muestras: {X_under.shape[0]}")
        print(f"   Ratio: {(y_under==0).sum()}:{(y_under==1).sum()}")
        print(f"   ⚠️ Perdemos {self.X_train_original.shape[0] - X_under.shape[0]} muestras de clase mayoritaria")
        
        if strategy in ['conservative', 'moderate', 'aggressive']:
            # 3. SMOTE conservador
            smote_ratio = {'conservative': 0.15, 'moderate': 0.3, 'aggressive': 0.5}[strategy]
            smote = SMOTE(sampling_strategy=smote_ratio, random_state=self.random_state, k_neighbors=3)
            X_smote, y_smote = smote.fit_resample(self.X_train_original, self.y_train_original)
            self.sampling_datasets['SMOTE'] = (X_smote, y_smote)
            print(f"\n✓ SMOTE ({strategy}): {X_smote.shape[0]} muestras")
            print(f"   Ratio: {(y_smote==0).sum()}:{(y_smote==1).sum()}")
        
        if strategy in ['moderate', 'aggressive']:
            # 4. BorderlineSMOTE
            borderline_ratio = {'moderate': 0.2, 'aggressive': 0.4}[strategy]
            borderline = BorderlineSMOTE(sampling_strategy=borderline_ratio, 
                                        random_state=self.random_state, k_neighbors=3)
            X_border, y_border = borderline.fit_resample(self.X_train_original, self.y_train_original)
            self.sampling_datasets['BorderlineSMOTE'] = (X_border, y_border)
            print(f"\n✓ BorderlineSMOTE: {X_border.shape[0]} muestras")
            print(f"   Ratio: {(y_border==0).sum()}:{(y_border==1).sum()}")
        
        if strategy in ['moderate', 'aggressive']:
            # 5. SMOTE + Undersampling (lo mejor de ambos mundos)
            print(f"\n✓ SMOTE + Undersampling (combinado):")
            over = SMOTE(sampling_strategy=0.3, random_state=self.random_state, k_neighbors=3)
            under = RandomUnderSampler(sampling_strategy=0.5, random_state=self.random_state)
            
            X_combined, y_combined = over.fit_resample(self.X_train_original, self.y_train_original)
            X_combined, y_combined = under.fit_resample(X_combined, y_combined)
            
            self.sampling_datasets['SMOTE+Under'] = (X_combined, y_combined)
            print(f"   Muestras finales: {X_combined.shape[0]}")
            print(f"   Ratio: {(y_combined==0).sum()}:{(y_combined==1).sum()}")
            print(f"   (Primero SMOTE para generar positivos, luego Under para reducir negativos)")
        
        if strategy == 'aggressive':
            # 6. Solo Undersampling agresivo (ratio casi 1:1)
            print(f"\n✓ Undersampling Agresivo:")
            under_aggressive = RandomUnderSampler(
                sampling_strategy=0.8,  # Ratio 1:1.25
                random_state=self.random_state
            )
            X_under_agg, y_under_agg = under_aggressive.fit_resample(
                self.X_train_original, self.y_train_original
            )
            self.sampling_datasets['UnderSampling_Aggressive'] = (X_under_agg, y_under_agg)
            print(f"   Muestras: {X_under_agg.shape[0]}")
            print(f"   Ratio: {(y_under_agg==0).sum()}:{(y_under_agg==1).sum()}")
            print(f"   ⚠️ Perdemos muchas muestras pero puede mejorar precision")
        
        print(f"\n✅ Total estrategias creadas: {len(self.sampling_datasets)}")
        
        return self.sampling_datasets
        
       
    def initialize_models(self):
        """Inicializar modelos optimizados para datos desbalanceados"""
        print("\n" + "=" * 80)
        print("INICIALIZANDO MODELOS")
        print("=" * 80)
        
        self.models = {
            # Modelos baseline rápidos
            'LogisticRegression': LogisticRegression(
                max_iter=1000, 
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=1
            ),
            
            # Modelos ensemble optimizados para desbalanceo
            'XGBoost_ScalePosWeight': XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.03,
                min_child_weight=3,
                gamma=0.2,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=19,  # Ratio de desbalanceo
                reg_alpha=0.3,
                reg_lambda=1.5,
                eval_metric='aucpr',
                random_state=self.random_state,
                n_jobs=1
            ),
            
            'RandomForest_Balanced': RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_split=15,
                min_samples_leaf=8,
                max_features='sqrt',
                class_weight='balanced_subsample',
                random_state=self.random_state,
                n_jobs=1
            ),
            
            'LightGBM_Unbalanced': LGBMClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.03,
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
            ),
            
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8,
                max_features='sqrt',
                random_state=self.random_state
            )
        }
        
        print(f"\n✓ {len(self.models)} modelos inicializados")
        print(f"   Configurados específicamente para datos desbalanceados")
        
        return self.models
    
    def train_and_evaluate(self, sampling_strategy='NoSampling'):
        """
        Entrenar y evaluar todos los modelos con una estrategia de sampling
        
        Args:
            sampling_strategy: Nombre de la estrategia de sampling a usar
        """
        print("\n" + "=" * 80)
        print(f"ENTRENANDO MODELOS - Sampling: {sampling_strategy}")
        print("=" * 80)
        
        X_train, y_train = self.sampling_datasets[sampling_strategy]
        
        results_key = f"{sampling_strategy}"
        if results_key not in self.results:
            self.results[results_key] = {}
        
        for name, model in self.models.items():
            print(f"\n{'─' * 80}")
            print(f"Entrenando: {name}")
            print(f"{'─' * 80}")
            
            try:
                # Entrenar modelo
                model.fit(X_train, y_train)
                
                # Predicciones
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(self.X_test)
                
                # Probabilidades (si disponible)
                try:
                    y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                except:
                    y_pred_proba = None
                
                # Métricas básicas
                train_acc = accuracy_score(y_train, y_pred_train)
                test_acc = accuracy_score(self.y_test, y_pred_test)
                precision = precision_score(self.y_test, y_pred_test, zero_division=0)
                recall = recall_score(self.y_test, y_pred_test, zero_division=0)
                f1 = f1_score(self.y_test, y_pred_test, zero_division=0)
                
                # Métricas avanzadas
                if y_pred_proba is not None:
                    auc_roc = roc_auc_score(self.y_test, y_pred_proba)
                    auc_pr = average_precision_score(self.y_test, y_pred_proba)
                else:
                    auc_roc = 0.0
                    auc_pr = 0.0
                
                # Validación cruzada
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall', n_jobs=1)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Matriz de confusión
                cm = confusion_matrix(self.y_test, y_pred_test)
                tn, fp, fn, tp = cm.ravel()
                
                # Métricas médicas
                sensibilidad = recall
                especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0
                vpp = precision
                vpn = tn / (tn + fn) if (tn + fn) > 0 else 0
                
                # Guardar resultados
                self.results[results_key][name] = {
                    'model': model,
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc_roc': auc_roc,
                    'auc_pr': auc_pr,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'y_pred': y_pred_test,
                    'y_pred_proba': y_pred_proba,
                    'confusion_matrix': cm,
                    'sensibilidad': sensibilidad,
                    'especificidad': especificidad,
                    'vpp': vpp,
                    'vpn': vpn,
                    'fn': int(fn),
                    'fp': int(fp),
                    'tp': int(tp),
                    'tn': int(tn)
                }
                
                # Imprimir resultados
                print(f"✓ Train Accuracy: {train_acc:.4f}")
                print(f"✓ Test Accuracy:  {test_acc:.4f}")
                print(f"✓ Precision:      {precision:.4f}")
                print(f"✓ Recall:         {recall:.4f} ⭐")
                print(f"✓ F1-Score:       {f1:.4f}")
                if auc_pr > 0:
                    print(f"✓ AUC-PR:         {auc_pr:.4f}")
                print(f"✓ CV Recall:      {cv_mean:.4f} (±{cv_std:.4f})")
                print(f"✓ FN (crítico):   {fn} casos no detectados")
                
            except Exception as e:
                print(f"❌ Error entrenando {name}: {str(e)}")
                continue
        
        return self.results[results_key]
    
    def get_best_model(self, metric='balanced'):
        """
        Obtener el mejor modelo
        
        Args:
            metric: 'recall', 'precision', 'f1', 'balanced' (recall*0.6 + precision*0.4)
        """
        print("\n" + "=" * 80)
        print("SELECCIONANDO MEJOR MODELO")
        print("=" * 80)
        
        # Compilar resultados de todas las estrategias
        all_results = []
        for strategy_name, models_results in self.results.items():
            for model_name, result in models_results.items():
                all_results.append({
                    'Strategy': strategy_name,
                    'Model': model_name,
                    'Recall': result['recall'],
                    'Precision': result['precision'],
                    'F1': result['f1_score'],
                    'AUC-PR': result['auc_pr'],
                    'CV_Recall': result['cv_mean'],
                    'FN': result['fn'],
                    'FP': result['fp']
                })
        
        df_comp = pd.DataFrame(all_results)
        
        # Calcular score según métrica
        if metric == 'balanced':
            df_comp['Score'] = df_comp['Recall'] * 0.6 + df_comp['Precision'] * 0.4
        else:
            df_comp['Score'] = df_comp[metric.capitalize()]
        
        df_comp = df_comp.sort_values('Score', ascending=False)
        
        print(f"\n📊 TOP 10 MODELOS (ordenados por {metric}):")
        print(df_comp.head(10).to_string(index=False))
        
        # Mejor modelo
        best_row = df_comp.iloc[0]
        self.best_strategy = best_row['Strategy']
        self.best_model_name = best_row['Model']
        self.best_model = self.results[self.best_strategy][self.best_model_name]['model']
        
        print(f"\n🏆 MEJOR MODELO: {self.best_model_name} ({self.best_strategy})")
        print(f"   Recall:    {best_row['Recall']:.4f}")
        print(f"   Precision: {best_row['Precision']:.4f}")
        print(f"   F1-Score:  {best_row['F1']:.4f}")
        print(f"   FN:        {int(best_row['FN'])} casos no detectados")
        
        return self.best_model, self.best_model_name, df_comp
    
    def save_results(self):
        """Guardar modelo y reportes"""
        print("\n" + "=" * 80)
        print("GUARDANDO RESULTADOS")
        print("=" * 80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar mejor modelo
        model_path = self.models_dir / "ictus_model_final.pkl"
        best_result = self.results[self.best_strategy][self.best_model_name]
        
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_name': self.best_model_name,
            'sampling_strategy': self.best_strategy,
            'metrics': {
                'recall': best_result['recall'],
                'precision': best_result['precision'],
                'f1': best_result['f1_score'],
                'auc_pr': best_result['auc_pr'],
                'sensibilidad': best_result['sensibilidad'],
                'especificidad': best_result['especificidad'],
                'vpp': best_result['vpp'],
                'vpn': best_result['vpn']
            },
            'confusion_matrix': {
                'tn': best_result['tn'],
                'fp': best_result['fp'],
                'fn': best_result['fn'],
                'tp': best_result['tp']
            }
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ Modelo guardado en: {model_path}")
        
        # Guardar comparación completa
        _, _, df_comp = self.get_best_model()
        comparison_path = self.reports_dir / f"model_comparison_{timestamp}.csv"
        df_comp.to_csv(comparison_path, index=False)
        print(f"✓ Comparación guardada en: {comparison_path}")
        
        # Guardar reporte detallado
        report_path = self.reports_dir / f"best_model_report_{timestamp}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"REPORTE - PREDICCIÓN DE ICTUS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Modelo: {self.best_model_name}\n")
            f.write(f"Estrategia de Sampling: {self.best_strategy}\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("MÉTRICAS DE RENDIMIENTO:\n")
            f.write(f"  Recall (Sensibilidad):  {best_result['recall']:.4f}\n")
            f.write(f"  Precision (VPP):        {best_result['precision']:.4f}\n")
            f.write(f"  F1-Score:               {best_result['f1_score']:.4f}\n")
            f.write(f"  AUC-PR:                 {best_result['auc_pr']:.4f}\n")
            f.write(f"  Especificidad:          {best_result['especificidad']:.4f}\n")
            f.write(f"  VPN:                    {best_result['vpn']:.4f}\n\n")
            
            f.write("MATRIZ DE CONFUSIÓN:\n")
            f.write(f"  TN: {best_result['tn']}  FP: {best_result['fp']}\n")
            f.write(f"  FN: {best_result['fn']}  TP: {best_result['tp']}\n\n")
            
            f.write("INTERPRETACIÓN CLÍNICA:\n")
            f.write(f"  • De 100 pacientes CON ictus → Detectamos {int(best_result['sensibilidad']*100)}\n")
            f.write(f"  • Si test POSITIVO → {best_result['vpp']*100:.1f}% probabilidad real de ictus\n")
            f.write(f"  • Casos NO detectados: {best_result['fn']} (CRÍTICO)\n")
        
        print(f"✓ Reporte guardado en: {report_path}")
        
        print("\n✓ Todos los resultados guardados exitosamente")


def main():
    """Función principal"""
    print("\n" + "=" * 80)
    print("ENTRENAMIENTO DE MODELOS - PREDICCIÓN DE ICTUS")
    print("=" * 80)
    
    # Configuración
    base_dir = Path(os.getcwd()).parent if 'notebooks' in os.getcwd() else Path(os.getcwd())
    data_path = base_dir / 'data' / 'preprocessed_dataSin.pkl'
    
    # Crear predictor
    predictor = IctusPredictor(data_path=data_path, random_state=42)
    
    # Pipeline completo
    predictor.load_preprocessed_data()
    predictor.apply_sampling_strategies(strategy='conservative')  # o 'moderate', 'aggressive'
    predictor.initialize_models()
    
    # Entrenar con diferentes estrategias
    for strategy in predictor.sampling_datasets.keys():
        predictor.train_and_evaluate(sampling_strategy=strategy)
    
    # Seleccionar mejor modelo
    predictor.get_best_model(metric='balanced')
    predictor.save_results()
    
    print("\n" + "=" * 80)
    print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 80)


if __name__ == "__main__":
    main()