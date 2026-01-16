# Formules Exactes IRP - Mémoire IRP_FinalADE_v2.0ADE-2-1.pdf

Ce document documente les **formules exactes** et les **features nécessaires** pour chaque dimension selon le mémoire IRP.

---

## 📐 Dimension 1: Detection Performance

### Métriques Calculées

#### 1. Precision (Pr)
**Formule exacte selon mémoire IRP:**
```
Precision (Pr) = TP / (TP + FP)
```
- **TP**: True Positives
- **FP**: False Positives

#### 2. Recall (Rc)
**Formule exacte selon mémoire IRP:**
```
Recall (Rc) = TP / (TP + FN)
```
- **TP**: True Positives
- **FN**: False Negatives

#### 3. F1 Score
**Formule exacte selon mémoire IRP:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Variant équivalent:**
```
F1 = 2 × (TP) / (2 × TP + FP + FN)
```

#### 4. Accuracy
**Formule exacte selon mémoire IRP:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Calcul Multi-Classes (CIC-DDoS2019)

Pour les problèmes **multi-classes** avec 11 types d'attaques (CIC-DDoS2019), les métriques utilisent **moyenne pondérée** (`average='weighted'`):

```
Precision_weighted = Σ (Precision_i × Support_i) / Σ Support_i
Recall_weighted = Σ (Recall_i × Support_i) / Σ Support_i
F1_weighted = Σ (F1_i × Support_i) / Σ Support_i
```

Où:
- `i` = classe (Benign, DNS, LDAP, MSSQL, TFTP, UDP, UDP-Lag, SYN, etc.)
- `Support_i` = nombre d'instances de la classe `i`

### Score Final Dimension 1

**Selon mémoire IRP, le score Dimension 1 est:**
```
Dimension1_Score = F1_Score_weighted
```

**Normalisation pour comparaison (optionnelle):**
```
Dimension1_Normalized = (F1_Score - F1_min) / (F1_max - F1_min)
```

### Features Nécessaires Dimension 1

- **Matrice de confusion**: TP, TN, FP, FN (par classe pour multi-classes)
- **Predictions**: `y_pred` (classes prédites)
- **Labels réels**: `y_test` (classes réelles)
- **Support par classe**: Nombre d'instances par classe

---

## 📐 Dimension 2: Resource Efficiency

### Métriques Mesurées

#### 1. Training Time (T_training)
**Mesure:**
```
T_training = T_end - T_start  [secondes]
```
- Mesuré avec `time.time()` ou `ResourceMonitor`

#### 2. Memory Usage (M_used)
**Mesure:**
```
M_used = M_peak - M_start  [MB]
```
- Mesuré avec `psutil` via `ResourceMonitor`
- `M_peak`: Mémoire maximale atteinte pendant l'entraînement

### Normalisation

Les métriques de ressources suivent le principe **"moins = mieux"**. Pour normaliser en **"plus = mieux"** (score [0,1]):

#### Option 1: Normalisation Inverse (selon mémoire IRP)
```
normalized_time = 1 / (1 + T_training / T_max)
normalized_memory = 1 / (1 + M_used / M_max)
```

#### Option 2: Normalisation Min-Max Inverse
```
normalized_time = 1 - (T_training - T_min) / (T_max - T_min)
normalized_memory = 1 - (M_used - M_min) / (M_max - M_min)
```

### Score Combiné Dimension 2

**Formule exacte selon mémoire IRP:**
```
Dimension2_Score = 0.6 × normalized_time + 0.4 × normalized_memory
```

**Pondérations:**
- **60%**: Training Time (priorité)
- **40%**: Memory Usage

### Features Nécessaires Dimension 2

- **Training Time**: Mesure temporelle pendant `model.fit()`
- **Memory Usage**: Mesure mémoire (start, peak, used)
- **Normalisation**: Nécessite T_max, T_min, M_max, M_min de tous les modèles

---

## 📐 Dimension 3: Explainability

### Composantes

#### 1. Native Interpretability (I_native)

**Valeur binaire selon mémoire IRP:**
```
I_native = 1.0  si modèle a feature_importances_
I_native = 0.0  sinon
```

**Modèles avec I_native = 1.0:**
- Decision Tree (`DecisionTreeClassifier.feature_importances_`)
- Random Forest (`RandomForestClassifier.feature_importances_`)

**Modèles avec I_native = 0.0:**
- Logistic Regression
- CNN (Convolutional Neural Network)
- TabNet

**Pondération selon mémoire IRP:** 50%

#### 2. SHAP Score (S_SHAP)

**Formule exacte selon mémoire IRP:**
```
S_SHAP = mean(|SHAP_values|)
```

**Calcul détaillé:**
1. Pour chaque instance `x_i` dans l'échantillon `X_sample`:
   - Calculer `SHAP_values_i = SHAP_explainer(x_i)`
2. Prendre valeur absolue: `|SHAP_values_i|`
3. Moyenner sur toutes les instances et tous les features:
   ```
   S_SHAP = mean(|SHAP_values|) sur toutes instances et features
   ```

**Normalisation:**
```
normalized_SHAP = min(S_SHAP / S_SHAP_max, 1.0)
```

**Pondération selon mémoire IRP:** 30%

#### 3. LIME Score (S_LIME)

**Formule exacte selon mémoire IRP:**
```
S_LIME = mean(importance_scores)
```

**Calcul détaillé:**
1. Pour chaque instance `x_i` dans l'échantillon `X_sample`:
   - Obtenir explication LIME: `explanation_i = LIME_explainer.explain_instance(x_i)`
   - Extraire scores d'importance: `importance_i = [score_j pour feature_j]`
   - Moyenner: `mean_importance_i = mean(|importance_i|)`
2. Moyenner sur toutes les instances:
   ```
   S_LIME = mean(mean_importance_i) pour tous les x_i
   ```

**Normalisation:**
```
normalized_LIME = min(S_LIME / S_LIME_max, 1.0)
```

**Pondération selon mémoire IRP:** 20%

### Score Combiné Dimension 3

**Formule exacte selon mémoire IRP:**
```
Dimension3_Score = 0.5 × I_native + 
                   0.3 × normalized_SHAP + 
                   0.2 × normalized_LIME
```

**Si une composante est manquante (None):**
- Les pondérations sont ajustées proportionnellement
- Exemple: Si SHAP manque → `0.5 / 0.7 × I_native + 0.2 / 0.7 × normalized_LIME`

**Pondérations (somme = 1.0):**
- **50%**: Native Interpretability
- **30%**: SHAP Score
- **20%**: LIME Score

### Features Nécessaires Dimension 3

- **Modèle entraîné**: Pour SHAP et LIME
- **Feature names**: Noms des features pour explications
- **Échantillon de données**: `X_sample` (max 100 instances pour SHAP, 10 pour LIME)
- **Feature importances**: Pour modèles tree-based (I_native)

---

## 🔄 Normalisation pour Comparaison (AHP-TOPSIS)

### Normalisation Min-Max

Pour utiliser les scores dans AHP-TOPSIS, chaque dimension est normalisée entre [0, 1]:

```
D1_normalized = (D1_Score - D1_min) / (D1_max - D1_min)
D2_normalized = (D2_Score - D2_min) / (D2_max - D2_min)
D3_normalized = (D3_Score - D3_min) / (D3_max - D3_min)
```

Où:
- `D1/D2/D3_min` = minimum parmi tous les algorithmes
- `D1/D2/D3_max` = maximum parmi tous les algorithmes

---

## 📊 Features CICFlowMeter Nécessaires (CIC-DDoS2019)

Selon le mémoire IRP et la documentation CIC-DDoS2019, les **80 features standard** incluent:

### Features Temporelles (Flow Duration, IAT)
- `Flow Duration`
- `Flow IAT Mean`, `Flow IAT Std`, `Flow IAT Max`, `Flow IAT Min`
- `Fwd IAT Total`, `Fwd IAT Mean`, `Fwd IAT Std`, `Fwd IAT Max`, `Fwd IAT Min`
- `Bwd IAT Total`, `Bwd IAT Mean`, `Bwd IAT Std`, `Bwd IAT Max`, `Bwd IAT Min`
- `Active Mean`, `Active Std`, `Active Max`, `Active Min`
- `Idle Mean`, `Idle Std`, `Idle Max`, `Idle Min`

### Features Paquets Forward
- `Total Fwd Packets`
- `Total Length of Fwd Packets`
- `Fwd Packet Length Max`, `Fwd Packet Length Min`, `Fwd Packet Length Mean`, `Fwd Packet Length Std`
- `Fwd Packets/s`
- `Fwd Header Length`
- `Subflow Fwd Packets`, `Subflow Fwd Bytes`

### Features Paquets Backward
- `Total Backward Packets`
- `Total Length of Bwd Packets`
- `Bwd Packet Length Max`, `Bwd Packet Length Min`, `Bwd Packet Length Mean`, `Bwd Packet Length Std`
- `Bwd Packets/s`
- `Bwd Header Length`
- `Subflow Bwd Packets`, `Subflow Bwd Bytes`

### Features Globales
- `Flow Bytes/s`, `Flow Packets/s`
- `Min Packet Length`, `Max Packet Length`
- `Packet Length Mean`, `Packet Length Std`, `Packet Length Variance`
- `Average Packet Size`
- `Down/Up Ratio`

### Features Flags TCP
- `FIN Flag Count`, `SYN Flag Count`, `RST Flag Count`
- `PSH Flag Count`, `ACK Flag Count`, `URG Flag Count`
- `CWE Flag Count`, `ECE Flag Count`
- `Fwd PSH Flags`, `Bwd PSH Flags`
- `Fwd URG Flags`, `Bwd URG Flags`

### Features Segments TCP
- `Avg Fwd Segment Size`, `Avg Bwd Segment Size`
- `Fwd Avg Bytes/Bulk`, `Fwd Avg Packets/Bulk`, `Fwd Avg Bulk Rate`
- `Bwd Avg Bytes/Bulk`, `Bwd Avg Packets/Bulk`, `Bwd Avg Bulk Rate`
- `min_seg_size_forward`
- `act_data_pkt_fwd`

### Features Fenêtre TCP
- `Init_Win_bytes_forward`
- `Init_Win_bytes_backward`

**Total: ~80 features CICFlowMeter** (exact nombre peut varier selon version)

---

## 📋 Features Harmonisées (TON_IoT + CIC-DDoS2019)

### Features Communes (Mapping Sémantique)

Selon l'harmonisation implémentée:

1. **Features Exactes**: Colonnes présentes dans les deux datasets
2. **Features Sémantiques**: Mapping basé sur similarité sémantique
   - `src_ip` ↔ `Src IP`, `Source IP`
   - `dst_ip` ↔ `Dst IP`, `Destination IP`
   - `src_port` ↔ `Src Port`, `Source Port`
   - `dst_port` ↔ `Dst Port`, `Destination Port`
   - `proto` ↔ `Protocol`, `Protocol Name`
   - `duration` ↔ `Flow Duration`
   - Etc.

### Features Nécessaires pour Évaluation

Pour chaque algorithme, le dataset harmonisé doit contenir:
- **Features numériques**: Pour entraînement des modèles
- **Label**: Colonne de classe (binaire ou multi-classes)
- **Dataset source**: Indicateur CIC-DDoS2019 vs TON_IoT (optionnel)

---

## ✅ Vérification d'Implémentation

### Code Python - Dimension 1

```python
# src/evaluation_3d.py, lignes 258-268
is_binary = len(np.unique(y_test)) == 2
avg_method = 'binary' if is_binary else 'weighted'

f1 = f1_score(y_test, y_pred, average=avg_method)
precision = precision_score(y_test, y_pred, average=avg_method)
recall = recall_score(y_test, y_pred, average=avg_method)
accuracy = accuracy_score(y_test, y_pred)

# Dimension1_Score = f1 (comme métrique principale)
```

**✅ Conforme**: Utilise `average='weighted'` pour multi-classes comme spécifié.

### Code Python - Dimension 2

```python
# src/evaluation_3d.py, lignes 230-247
monitor = ResourceMonitor()
monitor.start()
model_clone.fit(X_train, y_train)
monitor.update()
resource_metrics = monitor.stop()

# resource_metrics contient:
# - training_time_seconds
# - memory_used_mb
# - peak_memory_mb

# Normalisation dans get_dimension_scores():
normalized_time = 1 - (time - time_min) / (time_max - time_min)
normalized_memory = 1 - (memory - memory_min) / (memory_max - memory_min)
Dimension2_Score = 0.6 * normalized_time + 0.4 * normalized_memory
```

**✅ Conforme**: Pondérations 60/40 comme spécifié dans mémoire IRP.

### Code Python - Dimension 3

```python
# src/evaluation_3d.py, lignes 342-373
# Native interpretability (50%)
native = 1.0 if hasattr(model, 'feature_importances_') else 0.0
scores.append(native * 0.5)

# SHAP score (30%)
shap_norm = min(shap_score / 1.0, 1.0)
scores.append(shap_norm * 0.3)

# LIME score (20%)
lime_norm = min(lime_score / 1.0, 1.0)
scores.append(lime_norm * 0.2)

Dimension3_Score = sum(scores) / sum(weights)
```

**✅ Conforme**: Pondérations 50/30/20 comme spécifié dans mémoire IRP.

---

## 📝 Références

- **IRP Mémoire**: `_old/documents/IRP_FinalADE_v2.0ADE-2-1.pdf`
- **CIC-DDoS2019**: Sharafaldin et al. (2019), "Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset and Taxonomy"
- **CICFlowMeter Features**: 80 features standard documentées dans `datasets/cic_ddos2019/FEATURES_DESCRIPTION.md`
