# RAPPORT DE JUSTIFICATION DU CHOIX DE L'ALGORITHME

## Étape 1 : Identification de l'Objectif et des Alternatives
Objectif : Maximiser Performance & Explicabilité tout en minimisant les Ressources.
Alternatives testées : RF, CNN, TabNet, DT, LR

## Étape 2 : Détermination des Critères et Poids (AHP + SWARA)
Les poids globaux calculés pour chaque critère sont :
- f1 : 0.1418
- faithfulness : 0.1245
- recall : 0.1182
- latency : 0.1104
- precision : 0.1074
- stability : 0.0957
- accuracy : 0.0826
- complexity : 0.0798
- cpu_percent : 0.0789
- ram_percent : 0.0607

## Étape 3 : Analyse d'Efficience de Pareto
- **Solutions NON-DOMINÉES (Efficientes)** : TabNet, CNN, RF, DT, LR
- **Solutions DOMINÉES (Sous-optimales)** : Aucune
Note : Seules les solutions non-dominées sont considérées comme des choix rationnels.

## Étape 4 : Classement Final (TOPSIS)
Le score TOPSIS mesure la proximité à la solution idéale (1.0 = Parfait) :
- TabNet : 0.9405 (Efficient)
- CNN : 0.5868 (Efficient)
- RF : 0.1945 (Efficient)
- DT : 0.1434 (Efficient)
- LR : 0.0956 (Efficient)

## Étape 5 : Conclusion et Justification du Gagnant
L'algorithme sélectionné est : **TabNet**
Justification : Avec un score de 0.9405, ce modèle offre le meilleur compromis.
Il est classé comme efficient (Pareto) et surpasse les autres modèles sur l'agrégation pondérée des 3 dimensions.