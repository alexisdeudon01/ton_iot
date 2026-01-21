# Documentation des Calculs des 3 Dimensions d'Évaluation

Ce document décrit en détail comment les 3 dimensions d'évaluation sont calculées pour chaque algorithme dans le pipeline IRP.

## Vue d'ensemble

Les algorithmes sont évalués selon 3 dimensions :
1. **Dimension 1 : Detection Performance** (Performance de détection)
2. **Dimension 2 : Resource Efficiency** (Efficacité des ressources)
3. **Dimension 3 : Explainability** (Expliquabilité)

---

## Dimension 1 : Detection Performance

### Métriques utilisées

#### 1. Precision (Pr)
**Formule :**
```
Precision = TP / (TP + FP)
```
- **TP** : True Positives (vrais positifs)
- **FP** : False Positives (faux positifs)

**Interprétation :** Proportion de prédictions positives qui sont correctes.

#### 2. Recall (Rc)
**Formule :**
```
Recall = TP / (TP + FN)
```
- **TP** : True Positives
- **FN** : False Negatives (faux négatifs)

**Interprétation :** Proportion de cas positifs réellement détectés.

#### 3. F1 Score
**Formule :**
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Interprétation :** Moyenne harmonique de Precision et Recall. Le F1 Score est utilisé comme métrique principale pour la Dimension 1.

#### 4. Accuracy
**Formule :**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- **TN** : True Negatives (vrais négatifs)

**Interprétation :** Proportion totale de prédictions correctes.

### Calcul du Score Dimension 1

Pour les problèmes **multi-classes** (comme CIC-DDoS2019 avec 11 types d'attaques), les métriques utilisent `average='weighted'` selon la méthodologie CIC-DDoS2019.

**Score final Dimension 1 :**
- **Métrique principale** : F1 Score (normalisé entre [0, 1])
- Si nécessaire pour comparaison : normalisation `(f1_score - min) / (max - min)`

### Représentations visuelles

1. **Bar Chart** : F1, Precision, Recall par algorithme
2. **Matrice de confusion** : Visualisation des erreurs de classification
3. **Courbe ROC** : Pour problèmes binaires
4. **Heatmap** : Corrélation entre métriques

### Code Python (exemple)

```python
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Pour multi-classes (CIC-DDoS2019)
is_binary = len(np.unique(y_test)) == 2
avg_method = 'binary' if is_binary else 'weighted'

f1 = f1_score(y_test, y_pred, average=avg_method)
precision = precision_score(y_test, y_pred, average=avg_method)
recall = recall_score(y_test, y_pred, average=avg_method)
accuracy = accuracy_score(y_test, y_pred)

# Score Dimension 1 (F1 comme métrique principale)
dimension1_score = f1
```

---

## Dimension 2 : Resource Efficiency

### Métriques utilisées

#### 1. Training Time (Temps d'entraînement)
**Mesure :** Temps en secondes mesuré avec `time.time()` ou `ResourceMonitor`

#### 2. Memory Usage (Utilisation mémoire)
**Mesure :** Mémoire utilisée en MB mesurée avec `psutil` via `ResourceMonitor`

### Calcul du Score Dimension 2

Les métriques de ressources suivent le principe "moins = mieux". Pour créer un score normalisé où "plus = mieux", on applique des transformations inverses.

#### Normalisation Training Time

**Formule :**
```
normalized_time = 1 / (1 + training_time / max_training_time)
```

Cette formule transforme un temps de 0 secondes → 1.0, et un temps très long → proche de 0.

#### Normalisation Memory Usage

**Formule :**
```
normalized_memory = 1 / (1 + memory_usage / max_memory_usage)
```

#### Score combiné Dimension 2

**Formule :**
```
Dimension2_Score = 0.6 * normalized_time + 0.4 * normalized_memory
```

Ou avec normalisation Min-Max alternative :
```
normalized_time = 1 - (time - min_time) / (max_time - min_time)
normalized_memory = 1 - (memory - min_memory) / (max_memory - min_memory)
Dimension2_Score = 0.6 * normalized_time + 0.4 * normalized_memory
```

**Pondérations :** 
- 60% pour le temps d'entraînement (plus important)
- 40% pour la mémoire (secondaire)

### Représentations visuelles

1. **Bar Chart** : Training time (seconds) par algorithme
2. **Bar Chart** : Memory usage (MB) par algorithme
3. **Scatter Plot** : Time vs Memory (un point par algorithme)
4. **Graphique Radar** : Visualisation multi-métriques (time, memory)

### Code Python (exemple)

```python
import time
import psutil

# Mesure du temps
start_time = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start_time

# Mesure de la mémoire (via ResourceMonitor)
# Voir ResourceMonitor dans evaluation_3d.py

# Normalisation
max_time = max(all_training_times)
normalized_time = 1 / (1 + training_time / max_time)

max_memory = max(all_memory_usages)
normalized_memory = 1 / (1 + memory_usage / max_memory)

# Score Dimension 2
dimension2_score = 0.6 * normalized_time + 0.4 * normalized_memory
```

---

## Dimension 3 : Explainability

### Composantes utilisées

#### 1. SHAP Score
**Formule :**
```
SHAP_Score = mean(|SHAP_values|)
```

**Calcul :**
- Calculer les SHAP values pour un échantillon de données
- Prendre la valeur absolue de chaque SHAP value
- Calculer la moyenne sur tous les features et toutes les instances

**Interprétation :** Plus élevé = plus d'impact des features = meilleure explicabilité

#### 2. LIME Score
**Formule :**
```
LIME_Score = mean(importance_scores)
```

**Calcul :**
- Pour chaque instance dans l'échantillon :
  - Obtenir l'explication LIME
  - Extraire les scores d'importance des features
  - Prendre la valeur absolue de chaque score
  - Moyenner les scores
- Moyenner sur toutes les instances

#### 3. Native Interpretability
**Valeur :** Binaire
- `1.0` : Le modèle a `feature_importances_` (tree-based models : Decision Tree, Random Forest)
- `0.0` : Le modèle n'a pas d'interprétabilité native (Logistic Regression, CNN, TabNet)

### Calcul du Score Dimension 3

#### Normalisation des composantes

Chaque composante est normalisée entre [0, 1] si nécessaire :

```
normalized_shap = min(SHAP_score / max_shap_value, 1.0)
normalized_lime = min(LIME_score / max_lime_value, 1.0)
native_interpretability = 1.0 si disponible, 0.0 sinon
```

#### Score combiné Dimension 3

**Formule :**
```
Explainability = 0.5 * native_interpretability + 
                 0.3 * normalized_shap + 
                 0.2 * normalized_lime
```

**Pondérations :**
- **50%** : Native interpretability (le plus important - disponibilité immédiate)
- **30%** : SHAP score (explicabilité globale)
- **20%** : LIME score (explicabilité locale)

Si une composante est manquante (None), elle n'est pas incluse dans la somme, et les pondérations sont ajustées proportionnellement.

### Représentations visuelles

1. **Bar Chart** : SHAP score, LIME score, Native interpretability par algorithme
2. **Heatmap** : Features importantes par algorithme (si disponible)
3. **Graphique Radar** : Comparaison des 3 composantes d'explicabilité

### Code Python (exemple)

```python
# Native interpretability
native = 1.0 if hasattr(model, 'feature_importances_') else 0.0

# SHAP score (si disponible)
if SHAP_AVAILABLE:
    shap_values = explainer.shap_values(X_sample)
    shap_score = np.mean(np.abs(shap_values))
    normalized_shap = min(shap_score / 1.0, 1.0)  # Ajuster selon valeurs typiques
else:
    normalized_shap = 0.0

# LIME score (si disponible)
if LIME_AVAILABLE:
    lime_scores = [compute_lime_for_instance(x) for x in X_sample]
    lime_score = np.mean(lime_scores)
    normalized_lime = min(lime_score / 1.0, 1.0)  # Ajuster selon valeurs typiques
else:
    normalized_lime = 0.0

# Score Dimension 3 combiné
dimension3_score = (0.5 * native + 
                    0.3 * normalized_shap + 
                    0.2 * normalized_lime)
```

---

## Scores combinés 3D et normalisation

### Normalisation finale pour comparaison

Pour comparer les algorithmes, chaque dimension est normalisée entre [0, 1] :

```
dimension1_normalized = (f1_score - min_f1) / (max_f1 - min_f1)
dimension2_normalized = (resource_score - min_resource) / (max_resource - min_resource)
dimension3_normalized = (explainability_score - min_explainability) / (max_explainability - min_explainability)
```

### Représentations 3D combinées

1. **Scatter Plot 3D** : X=Performance, Y=Efficiency, Z=Explainability (un point par algorithme)
2. **Graphique Radar/Spider** : Les 3 dimensions normalisées par algorithme
3. **Heatmap** : Matrice des scores 3D (lignes=algorithmes, colonnes=dimensions)

---

## Interprétation des scores

### Dimension 1 (Detection Performance)
- **Score > 0.9** : Excellente performance de détection
- **Score 0.7-0.9** : Bonne performance
- **Score 0.5-0.7** : Performance acceptable
- **Score < 0.5** : Performance insuffisante

### Dimension 2 (Resource Efficiency)
- **Score > 0.8** : Très efficace (rapide et peu de mémoire)
- **Score 0.5-0.8** : Efficace
- **Score 0.3-0.5** : Efficacité modérée
- **Score < 0.3** : Peu efficace (lent ou gourmand en mémoire)

### Dimension 3 (Explainability)
- **Score > 0.7** : Très explicable
- **Score 0.4-0.7** : Modérément explicable
- **Score < 0.4** : Peu explicable (boîte noire)

---

## Références

- **CIC-DDoS2019** : "Developing Realistic Distributed Denial of Service (DDoS) Attack Dataset and Taxonomy" (Sharafaldin et al., 2019)
- **IRP Pipeline** : "AI-Powered Log Analysis for Smarter Threat Detection"
- **SHAP** : Lundberg & Lee (2017), "A Unified Approach to Interpreting Model Predictions"
- **LIME** : Ribeiro et al. (2016), "Why Should I Trust You?"
