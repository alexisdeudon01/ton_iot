# File Relations (imports locaux)

Relations basées sur les imports détectés entre fichiers Python locaux.


- **analyze_features.py**
  - Aucun lien local détecté.

- **datasets/cic_ddos2019/verify_files.py**
  - Aucun lien local détecté.

- **expert_pipeline.py**
  - Aucun lien local détecté.

- **main.py**
  - ↪️ src/app/cli.py (import)
  - ↪️ src/new_pipeline/main.py (import)

- **main_test.py**
  - Aucun lien local détecté.

- **req.py**
  - Aucun lien local détecté.

- **src/__init__.py**
  - Aucun lien local détecté.

- **src/ahp_topsis_framework.py**
  - Aucun lien local détecté.

- **src/app/__init__.py**
  - Aucun lien local détecté.

- **src/app/cli.py**
  - ↪️ src/config/__init__.py (import)

- **src/app/pipeline_runner.py**
  - ↪️ src/config/__init__.py (import)
  - ↪️ src/phases/phase1_config_search.py (import)
  - ↪️ src/phases/phase2_apply_best_config.py (import)
  - ↪️ src/phases/phase3_evaluation.py (import)
  - ↪️ src/phases/phase4_ahp_preferences.py (import)
  - ↪️ src/phases/phase5_topsis_ranking.py (import)

- **src/config/__init__.py**
  - Aucun lien local détecté.

- **src/config.py**
  - Aucun lien local détecté.

- **src/core/__init__.py**
  - Aucun lien local détecté.

- **src/core/data_harmonization.py**
  - ↪️ src/core/dataset_loader.py (import)
  - ↪️ src/feature_analyzer.py (import)
  - ↪️ src/irp_features_requirements.py (import)

- **src/core/dataset_loader.py**
  - ↪️ src/feature_analyzer.py (import)
  - ↪️ src/irp_features_requirements.py (import)
  - ↪️ src/system_monitor.py (import)

- **src/core/dependencies.py**
  - ↪️ src/ahp_topsis_framework.py (import)
  - ↪️ src/core/__init__.py (import)
  - ↪️ src/evaluation_3d.py (import)
  - ↪️ src/feature_analyzer.py (import)
  - ↪️ src/irp_features_requirements.py (import)
  - ↪️ src/main_pipeline.py (import)
  - ↪️ src/realtime_visualizer.py (import)
  - ↪️ src/results_visualizer.py (import)
  - ↪️ src/system_monitor.py (import)

- **src/core/dependency_graph.py**
  - Aucun lien local détecté.

- **src/core/feature_categorization.py**
  - Aucun lien local détecté.

- **src/core/feature_engineering.py**
  - Aucun lien local détecté.

- **src/core/model_utils.py**
  - Aucun lien local détecté.

- **src/core/preprocessing_pipeline.py**
  - ↪️ src/core/dataset_loader.py (import)

- **src/datastructure/base.py**
  - Aucun lien local détecté.

- **src/datastructure/flow.py**
  - Aucun lien local détecté.

- **src/evaluation/__init__.py**
  - Aucun lien local détecté.

- **src/evaluation/explainability.py**
  - Aucun lien local détecté.

- **src/evaluation/metrics.py**
  - Aucun lien local détecté.

- **src/evaluation/reporting.py**
  - Aucun lien local détecté.

- **src/evaluation/resources.py**
  - Aucun lien local détecté.

- **src/evaluation/visualizations.py**
  - Aucun lien local détecté.

- **src/evaluation_3d.py**
  - ↪️ src/core/dataset_loader.py (import)
  - ↪️ src/core/model_utils.py (import)
  - ↪️ src/core/preprocessing_pipeline.py (import)
  - ↪️ src/evaluation/visualizations.py (import)

- **src/feature_analyzer.py**
  - Aucun lien local détecté.

- **src/irp_features_requirements.py**
  - Aucun lien local détecté.

- **src/main_pipeline.py**
  - ↪️ src/ahp_topsis_framework.py (import)
  - ↪️ src/core/__init__.py (import)
  - ↪️ src/core/model_utils.py (import)
  - ↪️ src/evaluation_3d.py (import)
  - ↪️ src/models/cnn.py (import)
  - ↪️ src/models/tabnet.py (import)
  - ↪️ src/realtime_visualizer.py (import)
  - ↪️ src/system_monitor.py (import)
  - ↪️ src/ui/__init__.py (import)

- **src/models/__init__.py**
  - Aucun lien local détecté.

- **src/models/cnn.py**
  - Aucun lien local détecté.

- **src/models/registry.py**
  - Aucun lien local détecté.

- **src/models/sklearn_models.py**
  - Aucun lien local détecté.

- **src/models/tabnet.py**
  - Aucun lien local détecté.

- **src/new_pipeline/config.py**
  - Aucun lien local détecté.

- **src/new_pipeline/data_loader.py**
  - ↪️ src/system_monitor.py (import)

- **src/new_pipeline/main.py**
  - ↪️ src/core/dependency_graph.py (import)
  - ↪️ src/core/feature_categorization.py (import)
  - ↪️ src/datastructure/base.py (import)
  - ↪️ src/datastructure/flow.py (import)
  - ↪️ src/new_pipeline/config.py (import)
  - ↪️ src/new_pipeline/data_loader.py (import)
  - ↪️ src/new_pipeline/tester.py (import)
  - ↪️ src/new_pipeline/trainer.py (import)
  - ↪️ src/new_pipeline/validator.py (import)
  - ↪️ src/new_pipeline/xai_manager.py (import)
  - ↪️ src/system_monitor.py (import)

- **src/new_pipeline/tester.py**
  - Aucun lien local détecté.

- **src/new_pipeline/trainer.py**
  - Aucun lien local détecté.

- **src/new_pipeline/validator.py**
  - ↪️ src/new_pipeline/config.py (import)

- **src/new_pipeline/xai_manager.py**
  - ↪️ src/new_pipeline/config.py (import)

- **src/phases/__init__.py**
  - Aucun lien local détecté.

- **src/phases/phase1_config_search.py**
  - ↪️ src/config/__init__.py (import)
  - ↪️ src/core/__init__.py (import)

- **src/phases/phase2_apply_best_config.py**
  - ↪️ src/core/__init__.py (import)
  - ↪️ src/core/feature_engineering.py (import)

- **src/phases/phase3_evaluation.py**
  - ↪️ src/core/__init__.py (import)
  - ↪️ src/core/feature_engineering.py (import)
  - ↪️ src/core/model_utils.py (import)
  - ↪️ src/core/preprocessing_pipeline.py (import)
  - ↪️ src/evaluation_3d.py (import)
  - ↪️ src/models/__init__.py (import)

- **src/phases/phase4_ahp_preferences.py**
  - Aucun lien local détecté.

- **src/phases/phase5_topsis_ranking.py**
  - Aucun lien local détecté.

- **src/realtime_visualizer.py**
  - Aucun lien local détecté.

- **src/results_visualizer.py**
  - Aucun lien local détecté.

- **src/system_monitor.py**
  - Aucun lien local détecté.

- **src/ui/__init__.py**
  - ↪️ src/ui/features_popup.py (import)

- **src/ui/algorithm_visualizer.py**
  - Aucun lien local détecté.

- **src/ui/features_popup.py**
  - Aucun lien local détecté.

- **src/utils/__init__.py**
  - Aucun lien local détecté.

- **src/utils/optional_imports.py**
  - Aucun lien local détecté.

- **src/utils/path_helpers.py**
  - Aucun lien local détecté.

- **src/utils/viz_helpers.py**
  - Aucun lien local détecté.

- **tests/_legacy_tests/test_ahp_topsis.py**
  - ↪️ src/ahp_topsis_framework.py (import)

- **tests/_legacy_tests/test_dataset_loader_oom_fix.py**
  - ↪️ src/core/dataset_loader.py (import)

- **tests/_legacy_tests/test_dataset_source_added.py**
  - ↪️ src/core/data_harmonization.py (import)

- **tests/_legacy_tests/test_evaluation_3d_comprehensive.py**
  - ↪️ src/evaluation_3d.py (import)
  - ↪️ src/models/cnn.py (import)
  - ↪️ src/models/tabnet.py (import)

- **tests/_legacy_tests/test_explainability.py**
  - ↪️ src/evaluation/explainability.py (import)

- **tests/_legacy_tests/test_feature_engineering_common_cols.py**
  - ↪️ src/core/feature_engineering.py (import)

- **tests/_legacy_tests/test_imports_no_gui_dependency.py**
  - ↪️ src/config/__init__.py (import)
  - ↪️ src/core/__init__.py (import)
  - ↪️ src/evaluation/explainability.py (import)
  - ↪️ src/evaluation/metrics.py (import)
  - ↪️ src/evaluation/resources.py (import)
  - ↪️ src/models/registry.py (import)

- **tests/_legacy_tests/test_model_aware_profiles.py**
  - ↪️ src/config/__init__.py (import)
  - ↪️ src/phases/phase3_evaluation.py (import)

- **tests/_legacy_tests/test_model_utils.py**
  - ↪️ src/core/model_utils.py (import)
  - ↪️ src/models/cnn.py (import)
  - ↪️ src/models/tabnet.py (import)

- **tests/_legacy_tests/test_no_data_leakage.py**
  - ↪️ src/core/preprocessing_pipeline.py (import)

- **tests/_legacy_tests/test_phase1_108_configs.py**
  - ↪️ src/config/__init__.py (import)

- **tests/_legacy_tests/test_phase1_config_search.py**
  - ↪️ src/config/__init__.py (import)

- **tests/_legacy_tests/test_phase2_outputs.py**
  - ↪️ src/config/__init__.py (import)
  - ↪️ src/phases/phase2_apply_best_config.py (import)

- **tests/_legacy_tests/test_registry.py**
  - ↪️ src/config/__init__.py (import)
  - ↪️ src/models/registry.py (import)

- **tests/_legacy_tests/test_resource_metrics_non_negative.py**
  - ↪️ src/evaluation/resources.py (import)

- **tests/_legacy_tests/test_smoke_pipeline.py**
  - ↪️ src/app/pipeline_runner.py (import)
  - ↪️ src/config/__init__.py (import)
  - ↪️ src/phases/phase1_config_search.py (import)

- **tests/conftest.py**
  - ↪️ req.py (import)
  - ↪️ src/config/__init__.py (import)

- **tests/test_algo_handling.py**
  - ↪️ src/evaluation/visualizations.py (import)

- **tests/test_cnn.py**
  - ↪️ src/app/pipeline_runner.py (import)
  - ↪️ src/config/__init__.py (import)

- **tests/test_data_harmonization_small.py**
  - ↪️ src/core/data_harmonization.py (import)

- **tests/test_dataset_source_flag.py**
  - ↪️ src/phases/phase3_evaluation.py (import)

- **tests/test_model_aware_profiles.py**
  - ↪️ src/config/__init__.py (import)
  - ↪️ src/phases/phase3_evaluation.py (import)

- **tests/test_new_pipeline_components.py**
  - ↪️ src/core/feature_categorization.py (import)
  - ↪️ src/new_pipeline/data_loader.py (import)
  - ↪️ src/new_pipeline/tester.py (import)
  - ↪️ src/new_pipeline/trainer.py (import)
  - ↪️ src/new_pipeline/validator.py (import)
  - ↪️ src/new_pipeline/xai_manager.py (import)
  - ↪️ src/system_monitor.py (import)

- **tests/test_no_data_leakage.py**
  - ↪️ src/core/preprocessing_pipeline.py (import)

- **tests/test_no_global_fit_regression.py**
  - ↪️ src/core/preprocessing_pipeline.py (import)
  - ↪️ src/phases/phase3_evaluation.py (import)

- **tests/test_no_smote_leakage_phase3.py**
  - ↪️ src/core/preprocessing_pipeline.py (import)
  - ↪️ src/phases/phase3_evaluation.py (import)

- **tests/test_performance_and_ram.py**
  - ↪️ src/core/dataset_loader.py (import)
  - ↪️ src/models/sklearn_models.py (import)
  - ↪️ src/system_monitor.py (import)

- **tests/test_phase2_outputs.py**
  - ↪️ src/config/__init__.py (import)
  - ↪️ src/phases/phase2_apply_best_config.py (import)

- **tests/test_phase3_cnn_tabnet.py**
  - ↪️ src/config/__init__.py (import)
  - ↪️ src/evaluation/visualizations.py (import)
  - ↪️ src/evaluation_3d.py (import)

- **tests/test_phase3_synthetic.py**
  - ↪️ src/config/__init__.py (import)
  - ↪️ src/phases/phase3_evaluation.py (import)

- **tests/test_preprocessing_pipeline.py**
  - ↪️ src/core/preprocessing_pipeline.py (import)

- **tests/test_tabnet.py**
  - ↪️ src/app/pipeline_runner.py (import)
  - ↪️ src/config/__init__.py (import)

- **tests/test_test_transform_is_fold_fitted.py**
  - ↪️ src/core/preprocessing_pipeline.py (import)
  - ↪️ src/phases/phase3_evaluation.py (import)

- **verify_irp_compliance.py**
  - ↪️ src/core/dependencies.py (import)
