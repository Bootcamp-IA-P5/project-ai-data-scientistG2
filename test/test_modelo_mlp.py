#!/usr/bin/env python3
"""
🧪 Test del Modelo MLP - Predicción de Ictus
===========================================
Test específico para el modelo MLP desarrollado
"""

import os
import sys
import pickle
import pandas as pd


# Agregar el directorio de la app al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_modelo_mlp():
    """Test específico del modelo MLP"""
    
    print("🧪 INICIANDO TEST DEL MODELO MLP")
    print("="*50)
    
    # Test 1: Verificar archivos del modelo
    print("\n📁 Test 1: Verificando archivos del modelo MLP...")
    
    archivos_mlp = [
        'models/modelo_improved_info.pkl',
    ]
    
    for archivo in archivos_mlp:
        if os.path.exists(archivo):
            print(f"✅ {archivo} - EXISTE")
        else:
            print(f"❌ {archivo} - NO ENCONTRADO")
            return False
    
    # Test 2: Cargar el modelo MLP
    print("\n🤖 Test 2: Cargando modelo MLP...")
    
    try:
        with open('models/modelo_improved_info.pkl', 'rb') as f:
            model = pickle.load(f)
        
        print("✅ Modelo MLP cargado correctamente")
        print(f"   - Tipo de modelo: {type(model).__name__}")
        
        # Verificar métodos del modelo
        if hasattr(model, 'predict'):
            print("✅ Método 'predict' disponible")
        else:
            print("❌ Método 'predict' no encontrado")
            return False
            
        if hasattr(scaler, 'transform'):
            print("✅ Scaler funcional")
        else:
            print("❌ Scaler no funcional")
            return False
            
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return False
    
    # Test 3: Preprocesar datos de prueba
    print("\n🔧 Test 3: Probando preprocesamiento...")
    
    # Datos de prueba
    datos_test = pd.DataFrame([{
        'gender': 'Male',
        'age': 65.0,
        'hypertension': 1,
        'heart_disease': 0,
        'ever_married': 'Yes',
        'work_type': 'Private',
        'Residence_type': 'Urban',
        'avg_glucose_level': 200.0,
        'bmi': 28.5,
        'smoking_status': 'formerly smoked'
    }])
    
    try:
        # Mapear variables categóricas
        gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
        ever_married_map = {'No': 0, 'Yes': 1}
        work_type_map = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4}
        residence_map = {'Rural': 0, 'Urban': 1}
        smoking_map = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3}
        
        datos_procesados = datos_test.copy()
        datos_procesados['gender'] = datos_procesados['gender'].map(gender_map)
        datos_procesados['ever_married'] = datos_procesados['ever_married'].map(ever_married_map)
        datos_procesados['work_type'] = datos_procesados['work_type'].map(work_type_map)
        datos_procesados['Residence_type'] = datos_procesados['Residence_type'].map(residence_map)
        datos_procesados['smoking_status'] = datos_procesados['smoking_status'].map(smoking_map)
        
        print("✅ Preprocesamiento exitoso")
        print(f"   - Datos originales: {datos_test.shape}")
        print(f"   - Datos procesados: {datos_procesados.shape}")
        
    except Exception as e:
        print(f"❌ Error en preprocesamiento: {e}")
        return False
    
    # Test 4: Hacer predicción con el modelo MLP
    print("\n🔮 Test 4: Probando predicción del modelo MLP...")
    
    try:
        # Escalar datos
        X_scaled = scaler.transform(datos_procesados)
        
        # Hacer predicción
        if hasattr(model, 'predict_proba'):
            # Modelo sklearn
            probabilidad = model.predict_proba(X_scaled)[0][1]
            prediccion = model.predict(X_scaled)[0]
        elif hasattr(model, 'predict'):
            # Modelo TensorFlow/Keras
            prob_raw = model.predict(X_scaled, verbose=0)
            probabilidad = float(prob_raw[0][0]) if len(prob_raw[0]) == 1 else float(prob_raw[0][1])
            prediccion = 1 if probabilidad > 0.5 else 0
        else:
            print("❌ Modelo no tiene método de predicción válido")
            return False
        
        print("✅ Predicción exitosa")
        print(f"   - Predicción: {prediccion}")
        print(f"   - Probabilidad: {probabilidad:.4f}")
        print(f"   - Resultado: {'POSITIVO (Riesgo de Ictus)' if prediccion == 1 else 'NEGATIVO (Sin riesgo aparente)'}")
        
        # Validar rangos
        if 0 <= probabilidad <= 1:
            print("✅ Probabilidad en rango válido [0,1]")
        else:
            print(f"❌ Probabilidad fuera de rango: {probabilidad}")
            return False
            
        if prediccion in [0, 1]:
            print("✅ Predicción binaria válida")
        else:
            print(f"❌ Predicción inválida: {prediccion}")
            return False
        
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        return False
    
    # Test 5: Múltiples casos de prueba
    print("\n📋 Test 5: Probando múltiples casos...")
    
    casos_prueba = [
        {
            'nombre': 'Alto riesgo',
            'datos': {
                'gender': 'Male', 'age': 75, 'hypertension': 1, 'heart_disease': 1,
                'ever_married': 'Yes', 'work_type': 'Private', 'Residence_type': 'Urban',
                'avg_glucose_level': 250, 'bmi': 32, 'smoking_status': 'smokes'
            }
        },
        {
            'nombre': 'Bajo riesgo',
            'datos': {
                'gender': 'Female', 'age': 25, 'hypertension': 0, 'heart_disease': 0,
                'ever_married': 'No', 'work_type': 'Private', 'Residence_type': 'Urban',
                'avg_glucose_level': 85, 'bmi': 22, 'smoking_status': 'never smoked'
            }
        },
        {
            'nombre': 'Riesgo moderado',
            'datos': {
                'gender': 'Male', 'age': 55, 'hypertension': 1, 'heart_disease': 0,
                'ever_married': 'Yes', 'work_type': 'Self-employed', 'Residence_type': 'Rural',
                'avg_glucose_level': 180, 'bmi': 28, 'smoking_status': 'formerly smoked'
            }
        }
    ]
    
    try:
        for caso in casos_prueba:
            df_caso = pd.DataFrame([caso['datos']])
            
            # Preprocesar
            df_procesado = df_caso.copy()
            df_procesado['gender'] = df_procesado['gender'].map(gender_map)
            df_procesado['ever_married'] = df_procesado['ever_married'].map(ever_married_map)
            df_procesado['work_type'] = df_procesado['work_type'].map(work_type_map)
            df_procesado['Residence_type'] = df_procesado['Residence_type'].map(residence_map)
            df_procesado['smoking_status'] = df_procesado['smoking_status'].map(smoking_map)
            
            # Escalar y predecir
            X_scaled = scaler.transform(df_procesado)
            
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X_scaled)[0][1]
                pred = model.predict(X_scaled)[0]
            else:
                prob_raw = model.predict(X_scaled, verbose=0)
                prob = float(prob_raw[0][0]) if len(prob_raw[0]) == 1 else float(prob_raw[0][1])
                pred = 1 if prob > 0.5 else 0
            
            print(f"   {caso['nombre']}: Predicción={pred}, Probabilidad={prob:.3f}")
        
        print("✅ Múltiples casos procesados correctamente")
        
    except Exception as e:
        print(f"❌ Error en casos múltiples: {e}")
        return False
    
    # Test 6: Verificar información del modelo si existe
    print("\n📊 Test 6: Información adicional del modelo...")
    
    if os.path.exists('models/modelo_improved_info.pkl'):
        try:
            with open('models/modelo_improved_info.pkl', 'rb') as f:
                info = pickle.load(f)
            
            print("✅ Información del modelo encontrada:")
            for key, value in info.items():
                print(f"   - {key}: {value}")
                
        except Exception:
            print("⚠️ Archivo de información existe pero no se puede leer")
    else:
        print("⚠️ No se encontró información adicional del modelo")
    
    return True


def main():
    """Función principal"""
    
    # Verificar directorio
    if not os.path.exists('models/modelo_improved_info.pkl'):
        print("❌ Error: Este script debe ejecutarse desde el directorio raíz del proyecto")
        print("💡 Uso: cd /workspaces/project-ai-data-scientistG2 && python test_modelo_mlp.py")
        return False
    
    # Ejecutar tests
    exito = test_modelo_mlp()
    
    # Resumen final
    print("\n" + "="*50)
    print("📊 RESUMEN DEL TEST DEL MODELO MLP")
    print("="*50)
    
    if exito:
        print("🎉 ¡TEST DEL MODELO MLP EXITOSO!")
        print("✅ Tu modelo MLP funciona correctamente")
        print("✅ Predicciones validadas")
        print("✅ Preprocesamiento correcto")
        print("🚀 Modelo listo para usar")
    else:
        print("❌ TEST FALLÓ")
        print("🔧 Revisar configuración del modelo")
    
    print("="*50)
    return exito


if __name__ == "__main__":
    main()