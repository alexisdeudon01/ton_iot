# Rapport d'Évaluation : RF

## Résumé

- **F1 Score**: 0.4455
- **Resource Efficiency Score**: 0.1696
- **Explainability Score**: 1.0000

## Dimension 1: Detection Performance

- **F1 Score**: 0.4455 (métrique principale)
- **Precision (Pr)**: 0.2868
- **Recall (Rc)**: 1.0000
- **Accuracy**: 0.9932

**Interprétation**: Performance insuffisante. Le modèle a un bon équilibre entre Precision et Recall.

## Dimension 2: Resource Efficiency

- **Training Time**: 0.84 seconds
- **Memory Usage**: -569.11 MB
- **Peak Memory**: 0.00 MB

**Interprétation**: Peu efficace (lent ou gourmand en mémoire).

## Dimension 3: Explainability

- **Native Interpretability**: 0.0 (Pas d'interprétabilité native - boîte noire)
- **SHAP Score**: N/A
- **LIME Score**: N/A
- **Combined Explainability Score**: 1.0000

**Interprétation**: Très explicable.

## Forces et Faiblesses

**Forces**:
- Modèle interprétable

**Faiblesses**:
- Performance de détection modérée (F1 < 0.7)
- Entraînement lent ou gourmand en ressources

