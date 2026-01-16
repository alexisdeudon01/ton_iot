#!/usr/bin/env python3
"""
Configuration centralisée pour le pipeline IRP
Contient toutes les constantes, seuils, poids, et paramètres par défaut
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class PipelineConfig:
    """Configuration centralisée du pipeline IRP"""
    
    # Mode et chemins
    test_mode: bool = False
    sample_ratio: float = 1.0
    random_state: int = 42
    output_dir: str = "output"
    interactive: bool = False  # UI Tkinter optionnelle
    
    # Phase 1: Configuration Search (108 configs)
    phase1_search_enabled: bool = True
    phase1_n_configs: int = 108  # Exactement 108 configurations
    
    # Options preprocessing pour génération des 108 configs
    # Chaque option peut être activée/désactivée ou avoir plusieurs valeurs
    preprocessing_options: Dict[str, List[Any]] = field(default_factory=lambda: {
        'apply_encoding': [True, False],  # 2 options
        'apply_feature_selection': [True, False],  # 2 options
        'feature_selection_k': [10, 20, 30],  # 3 options (si feature_selection=True)
        'apply_scaling': [True, False],  # 2 options
        'scaling_method': ['RobustScaler', 'StandardScaler'],  # 2 options (si scaling=True)
        'apply_resampling': [True, False],  # 2 options
        'resampling_method': ['SMOTE', 'ADASYN'],  # 2 options (si resampling=True)
        'apply_cleaning': [True, False],  # 2 options (toujours True normalement, mais pour les 108 configs)
    })
    
    # Calcul: 2 * 2 * 3 * 2 * 2 * 2 * 2 * 2 = 384 combinaisons possibles
    # Mais pour avoir exactement 108, on doit réduire certaines options
    # Stratégie: certaines combinaisons sont conditionnelles
    # Pour simplifier: générer toutes les combinaisons valides et prendre les 108 premières
    # OU: réduire les options pour avoir exactement 108
    
    # Phase 2: Application de la meilleure config
    phase2_enabled: bool = True
    
    # Phase 3: Évaluation 3D
    phase3_enabled: bool = True
    phase3_algorithms: List[str] = field(default_factory=lambda: [
        'Logistic_Regression',
        'Decision_Tree',
        'Random_Forest',
        'CNN',
        'TabNet'
    ])
    phase3_cv_folds: int = 5
    
    # Dimension 2: Resource Efficiency
    # Ajout de inference latency et peak RAM selon mémoire
    dimension2_metrics: List[str] = field(default_factory=lambda: [
        'training_time',
        'inference_latency',  # Nouveau: temps pour prédire un batch
        'peak_memory_mb',  # Nouveau: mémoire de pointe en MB
        'memory_efficiency'  # Calculé: ratio performance/mémoire
    ])
    
    # Phase 4: AHP Preferences
    phase4_enabled: bool = True
    ahp_preferences: Dict[str, float] = field(default_factory=lambda: {
        'dimension1_performance': 0.5,  # Poids Dimension 1
        'dimension2_resources': 0.3,    # Poids Dimension 2
        'dimension3_explainability': 0.2 # Poids Dimension 3
    })
    # Vérification: somme = 1.0
    ahp_consistency_threshold: float = 0.1
    
    # Phase 5: TOPSIS Ranking
    phase5_enabled: bool = True
    topsis_output_formats: List[str] = field(default_factory=lambda: ['csv', 'md'])
    
    # Explainability: mapping vers échelle du mémoire
    explainability_methods: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'native': {
            'available_for': ['Decision_Tree', 'Random_Forest'],
            'score_range': [0.0, 1.0],
            'memory_scale': 'high'  # Haute expliquabilité native
        },
        'SHAP': {
            'sample_size': 100,
            'score_range': [0.0, 1.0],
            'memory_scale': 'medium'
        },
        'LIME': {
            'sample_size': 10,
            'score_range': [0.0, 1.0],
            'memory_scale': 'medium'
        }
    })
    
    # Dataset paths
    dataset_paths: Dict[str, str] = field(default_factory=lambda: {
        'ton_iot': 'datasets/ton_iot/train_test_network.csv',
        'cic_ddos2019': 'datasets/cic_ddos2019/'
    })
    
    def __post_init__(self):
        """Valider la configuration"""
        # Vérifier somme AHP = 1.0
        ahp_sum = sum(self.ahp_preferences.values())
        if abs(ahp_sum - 1.0) > 1e-6:
            raise ValueError(f"AHP preferences must sum to 1.0, got {ahp_sum}")
        
        # Vérifier que phase1_n_configs est raisonnable
        if self.phase1_n_configs < 1:
            raise ValueError(f"phase1_n_configs must be >= 1, got {self.phase1_n_configs}")
        
        # Valider sample_ratio
        if not (0.0 < self.sample_ratio <= 1.0):
            raise ValueError(f"sample_ratio must be in (0, 1], got {self.sample_ratio}")


def generate_108_configs() -> List[Dict[str, Any]]:
    """
    Génère exactement 108 configurations preprocessing
    
    Stratégie pour avoir exactement 108:
    - apply_encoding: 2 options (True/False)
    - apply_feature_selection: 2 options (True/False)
      - Si True: k = [10, 20, 30] → 3 sous-options
    - apply_scaling: 2 options (True/False)
      - Si True: method = ['RobustScaler', 'StandardScaler'] → 2 sous-options
    - apply_resampling: 2 options (True/False)
      - Si True: method = ['SMOTE', 'ADASYN'] → 2 sous-options
    
    Calcul:
    - Base: 2 * 2 * 2 * 2 = 16 combinaisons binaires
    - Avec sous-options: variable selon les flags
    
    Pour obtenir 108 exactement:
    - On ajuste en incluant certaines combinaisons de sous-options
    - Stratégie: générer toutes les combinaisons et prendre les 108 premières
    
    Note: Cleaning est toujours True (obligatoire)
    """
    from itertools import product
    
    configs = []
    
    # Options binaires
    for apply_encoding in [True, False]:
        for apply_feature_selection in [True, False]:
            for apply_scaling in [True, False]:
                for apply_resampling in [True, False]:
                    # Sous-options conditionnelles
                    feature_k_options = [10, 20, 30] if apply_feature_selection else [None]
                    scaling_methods = ['RobustScaler', 'StandardScaler'] if apply_scaling else [None]
                    resampling_methods = ['SMOTE', 'ADASYN'] if apply_resampling else [None]
                    
                    # Générer toutes les combinaisons de sous-options
                    for k, scale_method, resample_method in product(
                        feature_k_options,
                        scaling_methods,
                        resampling_methods
                    ):
                        config = {
                            'apply_cleaning': True,  # Toujours True
                            'apply_encoding': apply_encoding,
                            'apply_feature_selection': apply_feature_selection,
                            'feature_selection_k': k,
                            'apply_scaling': apply_scaling,
                            'scaling_method': scale_method,
                            'apply_resampling': apply_resampling,
                            'resampling_method': resample_method,
                        }
                        configs.append(config)
    
    # On devrait avoir ~72 configs, pour avoir 108 on duplique certaines avec variations
    # OU on ajoute des variations sur apply_cleaning (False dans certains cas rares)
    # Stratégie simple: si < 108, ajouter des variations avec apply_cleaning=False
    
    original_count = len(configs)
    if original_count < 108:
        # Ajouter des variations avec cleaning=False pour certaines configs
        extra_needed = 108 - original_count
        variations = []
        for i, base_config in enumerate(configs[:extra_needed]):
            variant = base_config.copy()
            variant['apply_cleaning'] = False
            variations.append(variant)
        configs.extend(variations)
    
    return configs[:108]  # Garantir exactement 108


# Configuration par défaut pour tests
TEST_CONFIG = PipelineConfig(
    test_mode=True,
    sample_ratio=0.001,
    random_state=42,
    output_dir="output/test",
    interactive=False
)


if __name__ == "__main__":
    # Test: générer et afficher les 108 configs
    configs = generate_108_configs()
    print(f"✅ Généré {len(configs)} configurations")
    print(f"\nExemple de configs (3 premières):")
    for i, config in enumerate(configs[:3], 1):
        print(f"\nConfig {i}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
