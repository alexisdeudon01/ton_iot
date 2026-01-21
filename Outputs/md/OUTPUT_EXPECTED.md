# Documentation des Outputs Attendus

Ce document décrit les fichiers de sortie générés par chaque phase du pipeline IRP et comment les interpréter.

## Structure des répertoires

```
output/
├── phase1_preprocessing/
│   ├── preprocessed_data.csv
│   └── harmonization_stats.json
├── phase3_evaluation/
│   ├── evaluation_results.csv
│   ├── dimension_scores.csv
│   ├── algorithm_reports/
│   │   ├── Logistic_Regression_report.md
│   │   ├── Decision_Tree_report.md
│   │   ├── Random_Forest_report.md
│   │   ├── CNN_report.md
│   │   └── TabNet_report.md
│   ├── dimension_explanation.md
│   └── visualizations/
│       ├── dimension1_performance.png
│       ├── dimension2_resources.png
│       ├── dimension3_explainability.png
│       └── combined_3d_radar.png
├── phase5_ranking/
│   ├── ahp_weights.csv
│   ├── ranking_results.csv
│   └── decision_matrix.csv
└── logs/
    ├── phase1.log
    ├── phase3.log
    ├── phase5.log
    └── main.log
```

---

## Phase 1 : Preprocessing Configuration Selection

### Workflow détaillé avec sous-étapes

La Phase 1 suit un workflow structuré en 6 étapes principales :

#### 1. Harmonisation des Caractéristiques (Features)

**Objectif :** Aligner les features des deux datasets (TON_IoT et CIC-DDoS2019) sur un schéma commun.

**Méthodes :**
- **CIC-DDoS2019** : Utilise 80+ features extraites par CICFlowMeter
- **TON_IoT** : Features originales ou version CIC-ToN-IoT (si disponible)
- **Matching** : 
  - Correspondances exactes (même nom de colonne)
  - Correspondances sémantiques (similarité de sens, unités, catégories)
- **Output** : Datasets harmonisés avec schéma unifié

#### 2. Alignement des Étiquettes (Labels) - Classification Binaire

**Objectif :** Standardiser les labels pour classification binaire (Normal/Benign = 0, Attack/DDoS = 1).

**CIC-DDoS2019 :**
- Colonne label : **Dernière colonne** du CSV
- Mapping : `Benign = 0`, toutes les attaques (non-Benign) = `1`
- Types d'attaques : DNS, LDAP, MSSQL, TFTP, UDP, UDP-Lag, SYN, etc.

**TON_IoT :**
- Colonne label : **Dernière colonne** du CSV
- Filtrage : Conserver uniquement les lignes avec `type='normal'` ou `type='ddos'`
- Mapping : `normal = 0`, `ddos = 1`
- Autres types (backdoor, injection, etc.) : **Exclus** du dataset

**Output :** Labels binaires standardisés (0 = Normal/Benign, 1 = Attack/DDoS)

#### 3. Prétraitement Unifié

Une fois les datasets fusionnés (via `pandas.concat`), les étapes suivantes sont appliquées :

##### 3.1. Data Cleaning (Nettoyage)
- **Suppression NaN et Infinity** : Remplacement par NaN puis imputation
- **Conversion numérique** : Toutes les colonnes converties en numérique
- **Suppression colonnes vides** : Colonnes entièrement NaN supprimées
- **Imputation** : Valeurs NaN restantes remplacées par médiane

##### 3.2. Encoding (Encodage)
- **Features catégorielles** : Encodage avec LabelEncoder
- **Gestion valeurs manquantes** : Remplacées par 'unknown' avant encodage

##### 3.3. Feature Selection (Sélection de Features)
- **Méthode** : Information Mutuelle (Mutual Information)
- **Nombre de features** : Top K features (défaut : 20)
- **Objectif** : Réduire la dimensionnalité et le temps d'entraînement
- **Critère** : Features les plus discriminantes pour la classification

##### 3.4. Scaling (Mise à l'échelle)
- **Méthode** : RobustScaler (basé sur médiane et IQR)
- **Avantage** : Robuste aux outliers (meilleur que StandardScaler pour trafic réseau)
- **Plage typique** : Valeurs dans [-3, 3] (peut être plus large pour outliers)
- **Objectif** : Normaliser les différences d'échelle entre trafic IoT (léger) et DDoS (massif)

##### 3.5. Resampling (Rééchantillonnage)
- **Méthode** : SMOTE (Synthetic Minority Over-sampling Technique)
- **Objectif** : Équilibrer les classes (éviter biais vers classe majoritaire)
- **Technique** : Création d'échantillons synthétiques pour classe minoritaire
- **Note** : CIC-DDoS2019 souvent très volumineux par rapport à TON_IoT

#### 4. Division et Test (Splitting)

##### 4.1. Stratification
- **Méthode** : `train_test_split` avec `stratify=y`
- **Objectif** : Assurer représentation proportionnelle de TON_IoT et CIC-DDoS2019 dans chaque split
- **Distribution des classes** : Maintenue dans chaque split

##### 4.2. Division en 3 parties
- **Training Set** : 70% (défaut) - Pour entraînement des modèles
- **Validation Set** : 15% (défaut) - Pour validation et hyperparamètres
- **Test Set** : 15% (défaut) - Pour évaluation finale

##### 4.3. Cross-Validation
- **Méthode** : 5-fold Stratified Cross-Validation
- **Objectif** : Estimation robuste de la performance
- **Utilisation** : Évaluation des algorithmes en Phase 3

### Fichiers générés

#### `output/phase1_preprocessing/preprocessed_data.csv`

**Description :** Données préprocessées après toutes les étapes (harmonisation, early fusion, cleaning, encoding, feature selection, scaling, resampling, splitting).

**Colonnes :**
- Colonnes numériques : Features sélectionnées et normalisées (RobustScaler)
- `label` : Labels binaires (0 = normal/benign, 1 = attack/ddos)

**Format :** CSV avec header

**Exemple :**
```csv
feature_1,feature_2,...,feature_n,label
0.234,-1.456,...,0.789,0
-0.123,2.345,...,-0.567,1
...
```

**Utilisation :** Input pour Phase 3 (évaluation des algorithmes)

**Note :** Les données sont déjà divisées en train/validation/test et peuvent être utilisées directement pour l'entraînement.

---

#### `output/phase1_preprocessing/harmonization_stats.json` (optionnel)

**Description :** Statistiques sur le processus d'harmonisation entre TON_IoT et CIC-DDoS2019.

**Contenu :**
```json
{
  "num_harmonized_features": 45,
  "num_common_features": 45,
  "cic_shape": [10000, 82],
  "ton_shape": [5000, 43],
  "fused_shape": [15000, 46],
  "compatible_features_ks": 38,
  "total_features_tested": 45
}
```

**Utilisation :** Analyse de la qualité de l'harmonisation

---

## Phase 3 : Multi-Dimensional Algorithm Evaluation

### Fichiers générés

#### `output/phase3_evaluation/evaluation_results.csv`

**Description :** Résultats complets d'évaluation pour tous les algorithmes.

**Colonnes :**
- `model_name` : Nom de l'algorithme (Logistic Regression, Decision Tree, Random Forest, CNN, TabNet)
- `f1_score` : F1 Score (Dimension 1 - métrique principale)
- `accuracy` : Accuracy
- `precision` : Precision (Pr)
- `recall` : Recall (Rc)
- `training_time_seconds` : Temps d'entraînement moyen (5-fold CV) en secondes
- `memory_used_mb` : Mémoire utilisée en MB
- `peak_memory_mb` : Pic de mémoire en MB
- `shap_score` : Score SHAP (si disponible)
- `lime_score` : Score LIME (si disponible)
- `native_interpretability` : 1.0 si interprétabilité native, 0.0 sinon
- `explainability_score` : Score combiné Dimension 3

**Format :** CSV avec header

**Exemple :**
```csv
model_name,f1_score,accuracy,precision,recall,training_time_seconds,memory_used_mb,peak_memory_mb,explainability_score
Logistic Regression,0.8234,0.8567,0.8123,0.8345,12.45,245.67,512.34,0.3456
Decision Tree,0.8912,0.9123,0.8765,0.9067,3.23,123.45,256.78,1.0000
...
```

**Utilisation :** Input pour Phase 5 (AHP-TOPSIS ranking)

---

#### `output/phase3_evaluation/dimension_scores.csv`

**Description :** Scores normalisés des 3 dimensions pour chaque algorithme.

**Colonnes :**
- `model_name` : Nom de l'algorithme
- `detection_performance` : Score Dimension 1 normalisé [0, 1]
- `resource_efficiency` : Score Dimension 2 normalisé [0, 1]
- `explainability` : Score Dimension 3 normalisé [0, 1]

**Format :** CSV avec header

**Exemple :**
```csv
model_name,detection_performance,resource_efficiency,explainability
Logistic Regression,0.7823,0.9123,0.3456
Decision Tree,0.9456,0.9876,1.0000
...
```

**Utilisation :** Visualisation et comparaison des algorithmes sur les 3 dimensions

---

#### `output/phase3_evaluation/algorithm_reports/{Algorithm}_report.md`

**Description :** Rapport détaillé par algorithme expliquant les 3 dimensions.

**Contenu pour chaque algorithme :**
- Résumé des performances
- Dimension 1 : Détail des métriques (F1, Precision, Recall)
- Dimension 2 : Temps d'entraînement et utilisation mémoire
- Dimension 3 : Scores SHAP/LIME et interprétabilité native
- Forces et faiblesses de l'algorithme

**Format :** Markdown

**Exemple :**
```markdown
# Rapport d'Évaluation : Logistic Regression

## Résumé
F1 Score: 0.8234 | Resource Efficiency: 0.9123 | Explainability: 0.3456

## Dimension 1: Detection Performance
- F1 Score: 0.8234
- Precision: 0.8123
- Recall: 0.8345
- Accuracy: 0.8567

Interprétation: Bonne performance de détection...

## Dimension 2: Resource Efficiency
- Training Time: 12.45 seconds
- Memory Usage: 245.67 MB
...
```

---

#### `output/phase3_evaluation/dimension_explanation.md`

**Description :** Explication générale des 3 dimensions et leur calcul.

**Contenu :**
- Explication de chaque dimension
- Comment les scores sont calculés
- Interprétation des valeurs

**Format :** Markdown

---

#### `output/phase3_evaluation/visualizations/`

**Description :** Graphiques et visualisations des résultats.

**Fichiers PNG générés :**
- `dimension1_performance.png` : Bar chart F1/Precision/Recall par algorithme
- `dimension2_resources.png` : Bar chart temps/mémoire par algorithme
- `dimension3_explainability.png` : Bar chart SHAP/LIME/Native par algorithme
- `combined_3d_radar.png` : Graphique radar des 3 dimensions par algorithme
- `scatter_3d.png` : Scatter plot 3D (Performance vs Efficiency vs Explainability)

**Format :** PNG (150 DPI)

---

## Phase 5 : AHP-TOPSIS Ranking

### Fichiers générés

#### `output/phase5_ranking/ahp_weights.csv`

**Description :** Poids AHP calculés pour les 3 dimensions.

**Colonnes :**
- `criterion` : Nom de la dimension (Detection Performance, Resource Efficiency, Explainability)
- `weight` : Poids normalisé [0, 1] (somme = 1.0)
- `consistency_ratio` : Ratio de cohérence AHP

**Format :** CSV avec header

**Exemple :**
```csv
criterion,weight,consistency_ratio
Detection Performance,0.6234,0.0234
Resource Efficiency,0.1876,0.0234
Explainability,0.1890,0.0234
```

**Utilisation :** Compréhension de l'importance relative des dimensions

---

#### `output/phase5_ranking/decision_matrix.csv`

**Description :** Matrice de décision utilisée pour TOPSIS.

**Colonnes :**
- `alternative` : Nom de l'algorithme
- `dimension1` : Score Dimension 1 normalisé
- `dimension2` : Score Dimension 2 normalisé
- `dimension3` : Score Dimension 3 normalisé

**Format :** CSV avec header

---

#### `output/phase5_ranking/ranking_results.csv`

**Description :** Classement final des algorithmes selon TOPSIS.

**Colonnes :**
- `rank` : Position dans le classement (1 = meilleur)
- `alternative` : Nom de l'algorithme
- `topsis_score` : Score TOPSIS (plus élevé = meilleur)
- `distance_positive_ideal` : Distance à la solution idéale positive
- `distance_negative_ideal` : Distance à la solution idéale négative
- `relative_closeness` : Proximité relative (utilisée pour le classement)

**Format :** CSV avec header, trié par rank

**Exemple :**
```csv
rank,alternative,topsis_score,distance_positive_ideal,distance_negative_ideal,relative_closeness
1,Random Forest,0.8234,0.1234,0.5678,0.8217
2,Decision Tree,0.7567,0.2345,0.4567,0.6608
...
```

**Utilisation :** Classement final recommandé des algorithmes

---

## Logs

### Fichiers de logs

#### `output/logs/phase1.log`, `phase3.log`, `phase5.log`, `main.log`

**Description :** Logs détaillés de chaque phase avec timestamps.

**Format :** Fichier texte avec format :
```
2024-01-16 10:23:45,123 - INFO - dataset_loader - Loading TON_IoT dataset...
2024-01-16 10:23:47,456 - INFO - dataset_loader - TON_IoT loaded: (5000, 43)
2024-01-16 10:23:48,789 - WARNING - data_harmonization - Feature X not compatible (p-value < 0.05)
...
```

**Niveaux :** DEBUG, INFO, WARNING, ERROR, CRITICAL

**Utilisation :** Débogage et suivi de l'exécution

---

## Interprétation des résultats

### Comment lire `evaluation_results.csv`

1. **F1 Score** : Plus élevé = meilleure performance de détection
2. **Training Time** : Plus bas = plus rapide
3. **Memory Usage** : Plus bas = plus efficace
4. **Explainability Score** : Plus élevé = plus explicable

### Comment lire `ranking_results.csv`

- **Rank 1** : Algorithme recommandé (meilleur compromis sur les 3 dimensions)
- **Topsis Score** : Score composite pondéré (plus élevé = meilleur)
- Les algorithmes sont classés du meilleur au moins bon

### Recommandations d'utilisation

1. **Pour performance maximale** : Choisir l'algorithme avec le plus haut F1 Score
2. **Pour efficacité** : Choisir l'algorithme avec le plus haut `resource_efficiency`
3. **Pour explicabilité** : Choisir l'algorithme avec le plus haut `explainability_score`
4. **Pour compromis optimal** : Utiliser le classement AHP-TOPSIS (Rank 1)

---

## Notes importantes

- Tous les fichiers CSV utilisent la virgule (`,`) comme séparateur
- Les valeurs manquantes sont représentées par `NaN` ou laissées vides
- Les scores sont normalisés entre [0, 1] sauf indication contraire
- Les temps sont en secondes, les mémoires en MB
- Les logs peuvent être volumineux selon le niveau de verbosité
