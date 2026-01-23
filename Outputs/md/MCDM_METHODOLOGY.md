# Méthodologie de Décision Multicritère (MCDM) pour la Sélection d'Algorithmes DDoS

Ce document détaille la méthodologie implémentée pour évaluer et classer les algorithmes d'IA dans le cadre de la détection DDoS sur les datasets ToN-IoT et CIC-DDoS2019.

## 1. Problématique
L'objectif est de sélectionner l'algorithme offrant le meilleur compromis entre trois piliers souvent contradictoires :
1.  **Performance** : Capacité à détecter précisément les attaques.
2.  **Explicabilité (XAI)** : Capacité à justifier les décisions du modèle.
3.  **Ressources** : Sobriété numérique et rapidité d'exécution.

## 2. Nécessité de la Mesure des Ressources
La mesure de la consommation de ressources n'est pas une option, mais une nécessité opérationnelle pour deux raisons majeures :
*   **Contraintes IoT (ToN-IoT)** : Les dispositifs IoT ont des capacités CPU et RAM extrêmement limitées. Un modèle performant mais trop gourmand provoquerait un déni de service par épuisement des ressources locales.
*   **Détection Temps Réel (CIC-DDoS2019)** : Sur des réseaux à haut débit, la latence d'inférence doit être minimale pour éviter la congestion du trafic légitime.

## 3. Fondements de la Mesure de Performance
Pour répondre à votre question sur la validité de la mesure, la performance de détection n'est pas une estimation, mais un calcul exact basé sur la **Vérité Terrain (Ground Truth)** :

*   **Source de Vérité** : Le programme se base sur les étiquettes réelles fournies dans les datasets originaux (colonne `label` pour ToN-IoT et `Label` pour CIC-DDoS2019).
*   **Mécanisme de Comparaison** : Pour chaque paquet réseau, le programme compare la **prédiction** de l'IA avec le **label réel**.
*   **Métriques Dérivées** : À partir de cette comparaison, on construit une **Matrice de Confusion** (Vrais Positifs, Faux Positifs, etc.) qui sert de base mathématique unique pour calculer le F1-Score, la Précision et le Rappel.

## 4. Analyse du Mixage des Datasets (ToN-IoT & CIC-DDoS2019)
Le programme permet d'étudier l'impact de la fusion ou du mixage de ces deux sources de données hétérogènes.

### Avantages du Mixage :
*   **Généralisation Accrue** : Le modèle apprend des motifs d'attaques provenant de deux mondes différents (IoT vs Réseau classique).
*   **Couverture Holistique** : ToN-IoT apporte la spécificité IoT, tandis que CIC-DDoS2019 apporte la volumétrie réseau.

### Désavantages et Défis :
*   **Désalignement des Caractéristiques** : Nécessite une étape de projection qui peut entraîner une perte d'information.
*   **Bruit Statistique (Domain Shift)** : Les distributions de trafic diffèrent, ce qui peut "confondre" certains algorithmes.

## 4. Hiérarchie des Critères
Nous avons structuré la décision en 3 piliers et 9 critères techniques.

### Pilier 1 : Performance (Poids AHP : 45%)
*   **F1-Score** : Équilibre entre précision et rappel.
*   **Rappel** : Capacité à ne manquer aucune attaque.
*   **Précision** : Capacité à ne pas bloquer de trafic légitime.
*   **Accuracy** : Performance globale.

### Pilier 2 : Explicabilité (Poids AHP : 30%)
*   **Fidélité (Faithfulness)** : Corrélation entre l'explication SHAP et le comportement réel du modèle.
*   **Stabilité** : Robustesse de l'explication face au bruit.
*   **Complexité** : Simplicité de l'explication (mesurée par l'entropie).

### Pilier 3 : Ressources (Poids AHP : 25%)
*   **Latence** : Temps de réponse par prédiction.
*   **Usage CPU** : Charge processeur.
*   **Usage RAM** : Empreinte mémoire.

## 5. Processus de Décision (Banc d'essai Multi-Bibliothèques)
Le pipeline utilise les bibliothèques de référence **PyMCDM** et **pymoo** pour garantir la robustesse du choix :
*   **Banc d'essai MCDM** : Comparaison de 10 méthodes (**TOPSIS, VIKOR, COPRAS, PROMETHEE II, ARAS, EDAS, MABAC, MOORA, WSM et WPM**).
*   **Analyse de Pareto (pymoo)** : Identification des solutions **non-dominées** (efficientes) et **dominées** (sous-optimales).

## 6. Validation de la Définition des Métriques
Pour garantir qu'une mesure de performance est "bien définie" et fiable, le programme s'appuie sur quatre critères :
1.  **Objectivité Mathématique** : Calcul via des formules standardisées (ex: F1-Score, Entropie SHAP).
2.  **Monotonicité** : Une amélioration réelle doit se traduire par une augmentation du score.
3.  **Sensibilité** : Évaluation sur des jeux de test distincts pour éviter le sur-apprentissage.
4.  **Analyse de Corrélation** : Vérification via Heatmap que les métriques ne sont pas redondantes.

## 7. Visualisations Fournies
Le programme génère automatiquement :
*   **Heatmap de la Matrice de Décision** : Performances normalisées de tous les modèles.
*   **Bar Chart des Poids Globaux** : Détail du calcul AHP x SWARA.
*   **Radar Chart Exhaustif** : Profil multidimensionnel de tous les modèles.
*   **Heatmap de Corrélation des Méthodes** : Convergence des algorithmes MCDM.

## 8. Formules Clés
*   **Normalisation Bénéfice** : $x / max(x)$
*   **Normalisation Coût** : $min(x) / x$
*   **Score TOPSIS** : $D^- / (D^+ + D^-)$
