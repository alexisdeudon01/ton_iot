#!/usr/bin/env python3
"""
IRP Features Requirements - Based on IRP_EXACT_FORMULAS.md
Documents which features are necessary for calculating the 3 dimensions according to IRP methodology
"""
from typing import List, Set, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class IRPFeaturesRequirements:
    """Defines features requirements based on IRP calculations"""
    
    @staticmethod
    def get_features_for_dimension1() -> Dict[str, str]:
        """
        Features necessary for Dimension 1: Detection Performance
        
        According to IRP_EXACT_FORMULAS.md:
        - Matrice de confusion: TP, TN, FP, FN (calculated from predictions)
        - Predictions: y_pred (from model predictions)
        - Labels réels: y_test (label column)
        - Support par classe: Calculated from labels
        
        Returns:
            Dict mapping requirement to description
        """
        return {
            'label_column': 'Required: Label column (y_test) for calculating TP, TN, FP, FN',
            'all_numeric_features': 'Required: All numeric features harmonized (to train models for predictions)',
            'feature_names': 'Required: Feature names for model training'
        }
    
    @staticmethod
    def get_features_for_dimension2() -> Dict[str, str]:
        """
        Features necessary for Dimension 2: Resource Efficiency
        
        According to IRP_EXACT_FORMULAS.md:
        - Training Time: Measured during model.fit() (no specific features needed)
        - Memory Usage: Measured with ResourceMonitor (no specific features needed)
        
        Note: No dataset features required, only metrics during training
        
        Returns:
            Dict mapping requirement to description
        """
        return {
            'training_time': 'Measured: Time during model.fit() - no dataset features needed',
            'memory_usage': 'Measured: Memory via ResourceMonitor - no dataset features needed',
            'note': 'No dataset features required for Dimension 2, only runtime metrics'
        }
    
    @staticmethod
    def get_features_for_dimension3() -> Dict[str, str]:
        """
        Features necessary for Dimension 3: Explainability
        
        According to IRP_EXACT_FORMULAS.md:
        - Modèle entraîné: Trained model (requires all numeric features)
        - Feature names: Names of features for SHAP/LIME explanations
        - Échantillon de données: X_sample (100 instances for SHAP, 10 for LIME)
        - Feature importances: For tree-based models (native interpretability)
        
        Returns:
            Dict mapping requirement to description
        """
        return {
            'all_numeric_features': 'Required: All numeric features harmonized (to train models)',
            'feature_names': 'Required: Feature names for SHAP/LIME explanations',
            'x_sample_shap': 'Required: Sample of 100 instances for SHAP calculations',
            'x_sample_lime': 'Required: Sample of 10 instances for LIME calculations',
            'feature_importances': 'Optional: For tree-based models (Decision Tree, Random Forest)'
        }
    
    @staticmethod
    def get_all_required_features() -> Dict[str, List[str]]:
        """
        Get all features required for IRP 3-dimensional evaluation
        
        Summary:
        - Dimension 1: All numeric features + label column
        - Dimension 2: No dataset features (runtime metrics only)
        - Dimension 3: All numeric features + feature names + samples
        
        Returns:
            Dict summarizing requirements by dimension
        """
        return {
            'dimension_1': [
                'All numeric features harmonized (for model training)',
                'Label column (for y_test and confusion matrix)',
                'Feature names (for model input)'
            ],
            'dimension_2': [
                'No dataset features required',
                'Runtime metrics: Training time, Memory usage'
            ],
            'dimension_3': [
                'All numeric features harmonized (for model training)',
                'Feature names (for SHAP/LIME explanations)',
                'Sample instances: 100 for SHAP, 10 for LIME'
            ],
            'common_requirements': [
                'All numeric features common between TON_IoT and CIC-DDoS2019',
                'Label column (standardized as "label")',
                'Features must be harmonized (same schema, normalized)'
            ]
        }
    
    @staticmethod
    def filter_features_for_irp(common_features: List[Dict], 
                                ton_columns: List[str],
                                cic_columns: List[str]) -> List[str]:
        """
        Filter common features to keep only those necessary for IRP calculations
        
        According to IRP_EXACT_FORMULAS.md, we need:
        - All numeric features (harmonized) for model training
        - Label column (excluded from features, handled separately)
        - Feature names for explanations
        
        Args:
            common_features: List of common feature mappings from FeatureAnalyzer
            ton_columns: List of TON_IoT column names
            cic_columns: List of CIC-DDoS2019 column names
            
        Returns:
            List of unified feature names that should be used for IRP evaluation
        """
        # Extract unified names from common features
        unified_features = []
        label_candidates = {'label', 'Label', 'Attack', 'attack', 'Class', 'class'}
        
        for feat in common_features:
            unified_name = feat.get('unified_name')
            if unified_name and unified_name not in label_candidates:
                # Check if it's numeric (would need actual data to verify, but include for now)
                unified_features.append(unified_name)
        
        logger.info(f"[IRP FEATURES] Filtered to {len(unified_features)} numeric features for IRP evaluation")
        logger.info(f"[IRP FEATURES] Requirements: All numeric features harmonized + label column")
        
        return unified_features
    
    @staticmethod
    def validate_features_for_irp(feature_names: List[str], 
                                  has_label: bool = True) -> Dict[str, bool]:
        """
        Validate that required features are present for IRP calculations
        
        Args:
            feature_names: List of feature names in the harmonized dataset
            has_label: Whether label column is present
            
        Returns:
            Dict with validation results
        """
        validation = {
            'has_label_column': has_label,
            'has_numeric_features': len(feature_names) > 0,
            'min_features_present': len(feature_names) >= 1,  # At least some features
            'sufficient_for_training': len(feature_names) >= 1,
            'sufficient_for_shap': len(feature_names) >= 1,
            'sufficient_for_lime': len(feature_names) >= 1,
        }
        
        all_valid = all(validation.values())
        validation['all_requirements_met'] = all_valid
        
        if not all_valid:
            missing = [k for k, v in validation.items() if not v and k != 'all_requirements_met']
            logger.warning(f"[IRP FEATURES] Missing requirements: {missing}")
        else:
            logger.info(f"[IRP FEATURES] All requirements met: {len(feature_names)} features + label")
        
        return validation


def get_irp_features_summary() -> str:
    """Get a summary of IRP features requirements as a string"""
    req = IRPFeaturesRequirements()
    
    summary = "=" * 70 + "\n"
    summary += "IRP FEATURES REQUIREMENTS (Based on IRP_EXACT_FORMULAS.md)\n"
    summary += "=" * 70 + "\n\n"
    
    summary += "Dimension 1: Detection Performance\n"
    summary += "-" * 70 + "\n"
    for key, desc in req.get_features_for_dimension1().items():
        summary += f"  • {key}: {desc}\n"
    
    summary += "\nDimension 2: Resource Efficiency\n"
    summary += "-" * 70 + "\n"
    for key, desc in req.get_features_for_dimension2().items():
        summary += f"  • {key}: {desc}\n"
    
    summary += "\nDimension 3: Explainability\n"
    summary += "-" * 70 + "\n"
    for key, desc in req.get_features_for_dimension3().items():
        summary += f"  • {key}: {desc}\n"
    
    summary += "\nCommon Requirements (All Dimensions)\n"
    summary += "-" * 70 + "\n"
    for req_text in req.get_all_required_features()['common_requirements']:
        summary += f"  • {req_text}\n"
    
    summary += "\n" + "=" * 70 + "\n"
    summary += "CONCLUSION: All numeric features harmonized + label column\n"
    summary += "=" * 70 + "\n"
    
    return summary


if __name__ == "__main__":
    print(get_irp_features_summary())
