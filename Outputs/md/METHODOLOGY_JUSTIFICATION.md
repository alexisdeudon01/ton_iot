# Rapport de Justification : Méthodologie, Design, Implémentation et Protocole d'Évaluation

**Projet :** Détection de DDoS dans l'IoT (ToN-IoT & CIC-DDoS2019)  
**Version :** 2.0  
**Date :** 22 Janvier 2026

---

## 1. Mise en œuvre technique (Implementation)

Cette section décrit l'exécution pratique du pipeline de détection et le traitement des données, tel qu'implémenté dans l'architecture logicielle du projet.

### 1.1 Architecture du Pipeline et Reproductibilité
Le projet utilise un pipeline modulaire orchestré par la classe `PipelineRunner`, garantissant une **reproductibilité totale** grâce à plusieurs mécanismes clés :
- **Gestion des Graines (Seeds) :** Utilisation systématique d'un `random_state` global (défini dans `PipelineConfig`) pour toutes les opérations stochastiques (splits, SMOTE, initialisation des modèles).
- **Ordre des Caractéristiques :** La classe `PreprocessBuilder` impose un `feature_order` strict et immuable après le prétraitement, évitant tout décalage de colonnes entre l'entraînement et l'inférence.
- **Isolation des Phases :** Chaque phase (1 à 5) produit des artefacts versionnés (JSON, Parquet, modèles sérialisés), permettant de rejouer n'importe quelle étape avec les mêmes entrées.

### 1.2 Pipeline en cinq phases : La réponse au compromis Performance/Explicabilité/Ressources
Le pipeline est spécifiquement conçu pour répondre à la problématique : *"Comment détecter des attaques DDoS en maximisant la performance et l'explicabilité tout en minimisant les ressources ?"*

1.  **Phase 1 (Optimisation) :** Exploration systématique via `PreprocessBuilder`. On cherche la configuration de prétraitement qui offre le meilleur compromis initial.
2.  **Phase 2 (Prétraitement) :** Application de la configuration optimale avec **Polars**, garantissant une exécution rapide et une faible empreinte mémoire dès le début du flux.
3.  **Phase 3 (Évaluation 3D) :** C'est le cœur de la réponse. Chaque modèle (LR, DT, RF, CNN, TabNet) est évalué simultanément sur trois dimensions :
    - **Performance :** F1-Score et AUC (Maximisation de la détection).
    - **Ressources :** Temps CPU et occupation RAM (Minimisation du coût).
    - **Explicabilité :** Stabilité et fidélité des explications SHAP/LIME (Maximisation de la transparence).
4.  **Phase 4 (Pondération AHP) :** La classe `AHP` permet de quantifier mathématiquement l'importance relative de ces trois objectifs. En ajustant les poids, on peut privilégier la performance pure ou, au contraire, favoriser un modèle léger et explicable pour un déploiement sur sonde IoT.
5.  **Phase 5 (Classement TOPSIS) :** L'algorithme `TOPSIS` synthétise ces mesures contradictoires. En calculant la distance par rapport à une "solution idéale" (qui aurait 100% de performance, 100% d'explicabilité et 0 ressource consommée), il désigne mathématiquement le modèle qui s'en rapproche le plus.

### 1.3 Optimisation Multi-Objectifs : Le Front de Pareto dans le Code
L'optimisation est gérée par le module dédié `pareto/`, qui agit comme le moteur de décision du système :
- **Localisation :** La logique réside dans `pareto/front.py`. Elle est invoquée après la Phase 3 (Évaluation 3D) pour filtrer les résultats bruts.
- **Identification des Solutions Non-Dominées :** La méthode `ParetoFront.get_pareto_front()` analyse la matrice des métriques. Elle élimine mathématiquement les algorithmes "dominés" (ceux qui sont moins performants, plus coûteux et moins explicables qu'un autre modèle existant).
- **Réduction de l'Espace de Décision :** En ne conservant que le **Front de Pareto**, le pipeline garantit que l'étape finale de classement (TOPSIS) ne porte que sur les meilleurs compromis techniquement possibles. Cela assure une réponse rigoureuse à la problématique : toute augmentation de la performance ou de l'explicabilité est validée par rapport au coût en ressources qu'elle engendre.

---

## 2. Protocole d'Évaluation (Protocol)

Cette section définit les règles de validation et de test pour garantir la rigueur scientifique et la fiabilité des résultats.

### 2.1 Validation du prétraitement
L'efficacité du prétraitement est validée par un score composite mesurant la capacité de généralisation :
$$Score = 0.6 \times F1_{cross-dataset} + 0.4 \times F1_{in-distribution}$$
Ce score assure que le pipeline de prétraitement favorise les caractéristiques qui restent discriminantes d'un dataset à l'autre, évitant ainsi le sur-apprentissage sur des artefacts spécifiques à une seule source.

### 2.2 Validation croisée et Tests de généralisation
- **Validation croisée :** Le projet implémente une **validation croisée stratifiée à 5 plis** (5-fold stratified cross-validation). Cela garantit que chaque pli conserve la même proportion de classes que le dataset original, fournissant une estimation stable de la performance.
- **Protocole de test de généralisation :** Pour contrer le "biais de laboratoire", le système est systématiquement testé sur un jeu de données différent de celui utilisé pour l'entraînement. Cette séparation stricte valide la capacité du modèle à détecter des attaques dans des environnements réseau qu'il n'a jamais rencontrés.

### 2.3 Validation de la cohérence (AHP)
La fiabilité du processus décisionnel est garantie par le calcul du **Ratio de Cohérence (Consistency Ratio)** dans la classe `AHP`. 
- Le système rejette toute matrice de comparaison par paires dont le ratio est supérieur à **0.1**. 
- Cette validation mathématique assure que les poids attribués aux critères (Performance, Ressources, Explicabilité) ne sont pas le fruit du hasard mais d'une logique experte consistante.

---

**Conclusion :** L'intégration des méthodes multicritères AHP et TOPSIS au sein d'un pipeline de traitement haute performance (Polars/Dask) permet d'offrir une solution de détection DDoS à la fois scientifiquement rigoureuse et adaptable aux besoins opérationnels.
