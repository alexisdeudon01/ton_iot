# Performance - Seuil 60.0

- Seuil applique: 60.0
- Formule: dim_performance >= Seuil
- Solutions admissibles: 3/5

## Solutions admissibles
- fused_RF
- fused_DT
- fused_TabNet

## Solutions rejetees
- fused_LR
- fused_CNN

## Interpretation
Seuil intermediaire: la contrainte filtre une partie des solutions. Cela illustre le compromis entre performances, explicabilite et ressources.

## Lien avec criteres du prof
Poids des piliers: Performance: 0.70, Explicabilité: 0.15, Ressources: 0.15
