
## 🎚️ Optimización de Threshold - Modelo Balanceado

### Estrategia de Balanceo: XGBoost (weight=19.12)
### Threshold Óptimo: 0.15

| Métrica | Valor |
|---------|-------|
| Accuracy | 0.7362 |
| Precision | 0.1340 |
| Recall | 0.7800 |
| F1-Score | 0.2287 |
| G-Mean | 0.7566 |
| Specificity | 0.7339 |
| Falsos Negativos | 11 / 50 (22.0%) |
| Falsos Positivos | 252 / 947 (26.6%) |

### Mejora vs Baseline:
- ✅ Recall: 32% → 78.0% (+46.0 puntos)
- ✅ F1-Score: 0.19 → 0.23 (+0.04)
- ✅ Falsos Negativos: 34 → 11 (reducción de 23 casos)

Este threshold optimizado:
- ✅ Maximiza F1-Score (balance Precision-Recall)
- ✅ Detecta 78.0% de los casos reales
- ✅ Reduce significativamente falsos negativos
- ✅ Adecuado para screening médico de ictus
