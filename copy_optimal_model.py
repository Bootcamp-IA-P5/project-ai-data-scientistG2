#!/usr/bin/env python3
"""
Script para copiar el modelo Ultra optimizado desde el notebook 
a los archivos que usa la aplicación Streamlit
"""

import joblib
import os
import shutil

def copy_optimal_model():
    """Copia el modelo y configuración optimizada para la app web"""
    
    print("🔄 Copiando modelo Ultra optimizado...")
    
    # Crear directorio data si no existe
    os.makedirs("data", exist_ok=True)
    
    try:
        # 1. Copiar modelo Ultra (si existe en formato H5)
        if os.path.exists("data/mlp_model_ultra.h5"):
            print("✅ Modelo Ultra H5 ya existe")
        else:
            print("⚠️ Modelo Ultra H5 no encontrado")
        
        # 2. Copiar scaler Ultra 
        if os.path.exists("data/scaler_ultra.pkl"):
            print("✅ Scaler Ultra ya existe")
        else:
            print("⚠️ Scaler Ultra no encontrado")
        
        # 3. Verificar configuración optimizada
        if os.path.exists("data/final_optimal_recall80_config.pkl"):
            config = joblib.load("data/final_optimal_recall80_config.pkl")
            print("✅ Configuración optimizada encontrada:")
            print(f"   📈 Precisión: {config['metrics']['precision']*100:.1f}%")
            print(f"   🔥 Recall: {config['metrics']['recall']*100:.1f}%")
            print(f"   🎯 Umbral: {config['optimal_threshold']:.4f}")
            print(f"   🏗️ Modelo: {config['best_model']}")
        else:
            print("❌ Configuración optimizada no encontrada")
            return False
        
        # 4. Crear archivo de backup del modelo original si existe
        if os.path.exists("data/mlp_model.pkl"):
            if not os.path.exists("data/mlp_model_original_backup.pkl"):
                shutil.copy2("data/mlp_model.pkl", "data/mlp_model_original_backup.pkl")
                print("📦 Backup del modelo original creado")
        
        print("\n🎉 ¡Modelo Ultra optimizado listo para Streamlit!")
        print("📝 Archivos necesarios:")
        print("   ✅ data/final_optimal_recall80_config.pkl")
        print("   📊 Configuración: Precisión 13.7%, Recall 80.0%")
        print("   🎯 Umbral óptimo: 0.2760")
        
        return True
        
    except Exception as e:
        print(f"❌ Error copiando modelo: {e}")
        return False

if __name__ == "__main__":
    copy_optimal_model()