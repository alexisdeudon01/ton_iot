# Contrainte Ressource - Seuil 90.0

- Seuil applique: 90.0
- Formule: (CPU% + RAM%) / 2 <= Seuil
- Solutions admissibles: 1/5

## Solutions admissibles
- CNN

## Solutions rejetees
- RF
- TabNet
- LR
- DT

## Interpretation
Seuil intermediaire: la contrainte filtre une partie des solutions. Cela illustre le compromis entre performances, explicabilite et ressources.

## Lien avec criteres du prof
Poids des piliers: Performance: 0.70, Explicabilité: 0.15, Ressources: 0.15
