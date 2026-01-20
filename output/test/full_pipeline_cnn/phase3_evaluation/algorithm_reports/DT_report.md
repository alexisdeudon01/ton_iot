# Rapport d'Évaluation : DT

## Résumé

- **F1 Score**: 0.4455
- **Resource Efficiency Score**: 0.9674
- **Explainability Score**: 1.0000

## Dimension 1: Detection Performance

- **F1 Score**: 0.4455 (métrique principale)
- **Precision (Pr)**: 0.2868
- **Recall (Rc)**: 1.0000
- **Accuracy**: 0.9932

**Interprétation**: Performance insuffisante. Le modèle a un bon équilibre entre Precision et Recall.

## Dimension 2: Resource Efficiency

- **Training Time**: 0.04 seconds
- **Memory Usage**: -673.13 MB
- **Peak Memory**: 0.00 MB

**Interprétation**: Très efficace (rapide et peu de mémoire).

## Dimension 3: Explainability

- **Native Interpretability**: 0.0 (Pas d'interprétabilité native - boîte noire)
- **SHAP Score**: N/A
- **LIME Score**: N/A
- **Combined Explainability Score**: 1.0000

**Interprétation**: Très explicable.

## Forces et Faiblesses

**Forces**:
- Entraînement rapide et efficace en mémoire
- Modèle interprétable

**Faiblesses**:
- Performance de détection modérée (F1 < 0.7)

