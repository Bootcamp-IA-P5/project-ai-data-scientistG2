TÉCNICAS DE BALANCEO EXPLICADAS
🔵 1. OVERSAMPLING (Sobremuestreo de la Clase Minoritaria)
SMOTE (Synthetic Minority Over-sampling Technique)

¿Qué hace?
- Crea ejemplos SINTÉTICOS de la clase minoritaria
- No copia, sino que INTERPOLA entre ejemplos cercanos
- Genera diversidad en lugar de duplicados

¿Cómo funciona?
1. Toma un ejemplo de la clase minoritaria
2. Encuentra sus 5 vecinos más cercanos
3. Elige uno aleatoriamente
4. Crea un nuevo ejemplo en la línea entre ambos

Ventajas:
✅ Aumenta ejemplos sin duplicar
✅ Genera diversidad realista
✅ Reduce overfitting vs simple duplicación

Desventajas:
⚠️ Puede generar ruido si hay outliers
⚠️ No considera la distribución de la clase mayoritaria

Original:  ⚫ (casos con ictus escasos)
           ⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪ (casos sin ictus)

Con SMOTE: ⚫⚫⚫⚫⚫⚫⚫⚫⚫ (generados sintéticamente)
           ⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪⚪


ADASYN (Adaptive Synthetic Sampling)
¿Qué hace?
- Versión ADAPTATIVA de SMOTE
- Genera MÁS ejemplos en zonas difíciles de aprender
- Se enfoca en la frontera de decisión

¿Cómo funciona?
1. Identifica ejemplos minoritarios rodeados de mayoritarios
2. Genera MÁS ejemplos sintéticos en esas zonas difíciles
3. Genera MENOS en zonas ya bien representadas

Ventajas:
✅ Se enfoca en casos difíciles (frontera de decisión)
✅ Mejora la precisión en zonas problemáticas
✅ Más inteligente que SMOTE

Desventajas:
⚠️ Puede generar más ruido que SMOTE
⚠️ Más sensible a outliers

¿Cuándo usar?

Dataset con zonas de solapamiento entre clases
Cuando SMOTE simple no da buenos resultados

BorderlineSMOTE
¿Qué hace?
- Genera ejemplos sintéticos SOLO en la frontera de decisión
- Ignora ejemplos "fáciles" en el interior

Casos considerados:
- DANGER: Ejemplos minoritarios rodeados de mayoritarios
- SAFE: Ejemplos minoritarios rodeados de otros minoritarios (IGNORADOS)
- NOISE: Outliers (IGNORADOS)

Ventajas:
✅ Se enfoca en la frontera (zona más importante)
✅ Reduce ruido vs SMOTE normal
✅ Más eficiente computacionalmente

Desventajas:
⚠️ Puede perder información de zonas "seguras"

2. UNDERSAMPLING (Submuestreo de la Clase Mayoritaria)
RandomUnderSampler
¿Qué hace?
- ELIMINA aleatoriamente ejemplos de la clase mayoritaria
- Balancea reduciendo, no aumentando

Ventajas:
✅ Rápido y simple
✅ Reduce tiempo de entrenamiento
✅ No genera datos sintéticos (más "real")

Desventajas:
❌ PIERDE INFORMACIÓN valiosa
❌ Puede eliminar ejemplos importantes
❌ Reduce tamaño del dataset
¿Cuándo usar?

Dataset GRANDE donde puedes permitirte perder ejemplos
Cuando el oversampling no funciona (genera demasiado ruido)

3. MÉTODOS HÍBRIDOS (Combinan Over + Under)
¿Qué hace?
1. Aplica SMOTE (genera ejemplos sintéticos)
2. Aplica Tomek Links (elimina ejemplos ambiguos)

Tomek Links:
- Identifica pares de ejemplos (uno de cada clase) muy cercanos
- Son candidatos a ser ruido o estar en la frontera
- Los elimina para "limpiar" la frontera

Ventajas:
✅ Balancea + Limpia
✅ Reduce ambigüedad en la frontera
✅ Mejor que SMOTE solo

Desventajas:
⚠️ Más lento que SMOTE
⚠️ Puede eliminar información útil

SMOTEENN
¿Qué hace?
1. Aplica SMOTE (genera ejemplos)
2. Aplica Edited Nearest Neighbors (limpia ejemplos mal clasificados)

ENN (Edited Nearest Neighbors):
- Encuentra ejemplos cuya clase difiere de sus vecinos
- Los considera ruido y los elimina

Ventajas:
✅ Limpieza más agresiva que SMOTETomek
✅ Reduce ruido significativamente
✅ Mejora precisión del modelo

Desventajas:
⚠️ Puede ser demasiado agresivo
⚠️ Pierde más información que SMOTETomek


4. MÉTODOS BASADOS EN ALGORITMOS
XGBoost con scale_pos_weight
¿Qué hace?
- NO modifica el dataset
- Ajusta internamente los PESOS de las clases
- Da más importancia a la clase minoritaria durante el entrenamiento

scale_pos_weight = (nº casos negativos) / (nº casos positivos)

Ejemplo:
950 negativos / 50 positivos = 19
→ scale_pos_weight = 19

Ventajas:
✅ No genera datos sintéticos (más "puro")
✅ Muy eficiente (no aumenta dataset)
✅ Integrado en XGBoost, LightGBM
✅ No requiere preprocesamiento adicional

Desventajas:
⚠️ Solo funciona con algoritmos que lo soporten
⚠️ Puede requerir ajuste fino del peso
¿Cuándo usar?

Primera opción con XGBoost/LightGBM
Cuando quieres solución rápida sin modificar datos

BalancedRandomForest
¿Qué hace?
- Random Forest modificado para datasets desbalanceados
- Hace undersampling automático en cada árbol
- Cada árbol ve un subset balanceado diferente

Funcionamiento:
1. Para cada árbol:
   - Toma TODOS los ejemplos minoritarios
   - Muestrea ALEATORIAMENTE el mismo nº de mayoritarios
   - Entrena el árbol con ese subset balanceado
2. Combina las predicciones (ensemble)

Ventajas:
✅ Aprovecha poder del ensemble
✅ Cada árbol ve datos balanceados
✅ No pierde información (cada árbol ve diferentes mayoritarios)
✅ Built-in en imbalanced-learn

Desventajas:
⚠️ Más lento que Random Forest normal
⚠️ Puede tener menos accuracy general que XGBoost

 COMPARACIÓN RÁPIDA
Técnica	Modifica Dataset	Velocidad	Mejor Para
SMOTE	✅ Aumenta	⚡⚡ Media	Desbalanceo moderado (5:1 a 10:1)
ADASYN	✅ Aumenta	⚡ Lenta	Fronteras complejas, solapamiento
BorderlineSMOTE	✅ Aumenta	⚡⚡ Media	Enfoque en frontera de decisión
SMOTETomek	✅ Aumenta + Reduce	⚡ Lenta	Limpieza de frontera
SMOTEENN	✅ Aumenta + Reduce	⚡ Muy lenta	Limpieza agresiva de ruido
RandomUnderSampler	❌ Reduce	⚡⚡⚡ Rápida	Datasets grandes, prioridad velocidad
scale_pos_weight	❌ No modifica	⚡⚡⚡ Muy rápida	XGBoost/LightGBM, solución rápida
BalancedRandomForest	❌ No modifica	⚡⚡ Media	Ensemble, robustez