# Critères de Calcul des Dimensions et Métriques

Ce document détaille les formules et critères utilisés pour le calcul des dimensions (caractéristiques alignées) et des métriques de performance dans le pipeline DDoS.

## 1. Alignement des Caractéristiques (Feature Alignment)

L'alignement entre CICDDoS2019 et TON_IoT repose sur la similarité statistique de leurs distributions.

### A. Similarité Cosinus (Cosine Similarity)
Utilisée pour comparer les vecteurs de descripteurs (moyenne, écart-type, min, max, quartiles) de deux colonnes.
$$ \text{similarity} = \frac{A \cdot B}{\|A\| \|B\|} $$
*   **Critère** : `cosine_min` (défaut 0.95). Une valeur proche de 1 indique des profils statistiques identiques.

### B. Test de Kolmogorov-Smirnov (KS Test)
Évalue si deux échantillons proviennent de la même distribution continue.
*   **Calcul** : Distance maximale entre les fonctions de répartition empiriques.
*   **Critère** : `ks_p_min` (défaut 0.05). Si la p-value est supérieure au seuil, on ne rejette pas l'hypothèse que les distributions sont identiques.

### C. Distance de Wasserstein (Earth Mover's Distance)
Mesure l'effort minimal pour transformer une distribution en une autre.
*   **Critère** : `wasserstein_max`. Plus la distance est faible, plus les distributions sont proches.

---

## 2. Profilage des Données (Data Profiling)

Chaque colonne est analysée selon les dimensions suivantes :

| Dimension | Formule / Calcul |
| :--- | :--- |
| **Missing Rate** | $\frac{n_{null}}{n_{total}}$ |
| **Mean** | $\frac{1}{n} \sum_{i=1}^{n} x_i$ |
| **Standard Deviation** | $\sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2}$ |
| **Label Balance** | Ratio $\frac{y=1}{y=0}$ pour mesurer le déséquilibre des classes. |

---

## 3. Évaluation de la Performance (Evaluation Metrics)

Les métriques sont calculées sur les prédictions fusionnées (Late Fusion).

### A. F1-Score
Moyenne harmonique de la Précision et du Rappel. Crucial pour les datasets DDoS souvent déséquilibrés.
$$ F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} $$

### B. Précision (Precision)
Capacité à ne pas marquer comme attaque un flux légitime.
$$ \text{Precision} = \frac{TP}{TP + FP} $$

### C. Rappel (Recall / Sensitivity)
Capacité à détecter toutes les attaques DDoS.
$$ \text{Recall} = \frac{TP}{TP + FN} $$

---

## 4. Dimension Performance

La performance est évaluée à deux niveaux : système et prédictif, **indépendamment pour chaque algorithme** (LR, DT, RF, CNN, TabNet).

### A. Performance Système (Efficacité)
*   **Latence de Traitement** : Temps d'exécution par tâche ($T_{task}$ en secondes).
*   **Débit (Throughput)** : Nombre de lignes traitées par seconde ($\frac{N_{rows}}{T_{task}}$).
*   **Empreinte Mémoire** : Utilisation de la RAM (Peak RSS en MB) capturée par le `ResourceMonitor`.
*   **Charge CPU** : Pourcentage d'utilisation CPU moyen durant le calcul.

### B. Performance Prédictive (Métriques ML)
*   **F1-Score** : Métrique principale pour le déséquilibre de classe.
*   **Précision / Rappel** : Équilibre entre faux positifs et détection exhaustive.

---

## 5. Dimension Explicabilité (Explainability)

L'explicabilité est traitée **spécifiquement pour chaque algorithme** afin de comparer leurs logiques de décision.

### A. Importance des Caractéristiques (Feature Importance)
*   **Modèles Intrinsèques** : Pour RF et DT, calcul basé sur la réduction de l'impureté de Gini.
*   **Coefficients** : Pour LR, poids normalisés attribués à chaque dimension d'entrée.
*   **Attribution Globale** : Moyenne des importances à travers les 5 algorithmes pour identifier les "Top Dimensons" critiques (ex: `Source Port`, `Flow Duration`).

### B. Traçabilité de la Fusion (Late Fusion Traceability)
*   **Contribution par Modèle** : Chaque prédiction finale est décomposable en $M$ sous-prédictions.
*   **Score de Consensus** : Écart-type entre les probabilités des modèles ($\sigma_{probas}$). Un $\sigma$ faible indique une haute confiance inter-modèles.

---

## 6. Late Fusion (Fusion Tardive)

Le calcul de la probabilité finale pour un échantillon $i$ est la moyenne arithmétique des sorties des modèles :
$$ P_{final}(i) = \frac{1}{M} \sum_{m=1}^{M} P_m(i) $$
Où $M$ est le nombre de modèles (LR, DT, RF, CNN, TabNet) et $P_m$ la probabilité prédite par le modèle $m$.
