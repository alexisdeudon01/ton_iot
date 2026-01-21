#!/usr/bin/env python3
"""
Expert DDoS Detection Pipeline
AI-Powered Network Traffic Analysis with XAI Validation
Structure: Loading -> Preprocessing -> Tuning -> Training -> Evaluation -> XAI
"""

import os
import sys
import json
import time
import logging
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

# XAI
import shap
import lime
import lime.lime_tabular

# Deep Learning (Optional)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from pytorch_tabnet.tab_model import TabNetClassifier
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CONFIGURATION & LOGGING SETUP
# ==============================================================================

DEFAULT_CONFIG = {
    "data": {
        "path": "datasets/ton_iot/train_test_network.csv",
        "target": "label",
        "test_size": 0.2,
        "val_size": 0.2,
        "sample_n": 10000  # For demo speed
    },
    "models": ["LR", "DT", "RF", "XGBoost"], # Add "CNN", "TabNet" if available
    "tuning": {
        "n_iter": 5,
        "cv": 3
    },
    "xai": {
        "methods": ["SHAP", "LIME", "FI"],
        "n_samples_global": 100,
        "n_samples_local": 1
    }
}

def setup_expert_logging(log_path: Path):
    """Configure un logging ultra-explicite pour le suivi des micro-tâches."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("ExpertPipeline")

def create_directory_structure():
    """Génère la structure de dossiers pour les résultats."""
    dirs = [
        "logs",
        "plots/global",
        "plots/local",
        "models",
        "metrics",
        "rr"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Nettoyage du dossier rr (résultats récents)
    rr_path = Path("rr")
    for item in rr_path.iterdir():
        if item.is_file():
            item.unlink()
    print(f"Structure de dossiers initialisée. Dossier 'rr/' vidé.")

# ==============================================================================
# 2. DATA LOADING & PROFILING
# ==============================================================================

class ExpertDataLoader:
    """Gère le chargement et le profiling initial des données."""

    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """Charge les données réelles (CSV/Parquet)."""
        task_name = "Chargement des données"
        self.logger.info(f"START: {task_name}")
        path = Path(self.config['data']['path'])

        start_time = time.time()
        if not path.exists():
            self.logger.error(f"Fichier non trouvé: {path}. Création de données synthétiques pour la démo.")
            self.df = self._generate_synthetic_data()
        else:
            if path.suffix == '.csv':
                self.df = pd.read_csv(path, low_memory=False)
            elif path.suffix == '.parquet':
                self.df = pd.read_parquet(path)

        # Échantillonnage pour la rapidité si configuré
        if self.config['data'].get('sample_n') and len(self.df) > self.config['data']['sample_n']:
            self.df = self.df.sample(self.config['data']['sample_n'], random_state=42)

        self.logger.info(f"END: {task_name} | Shape: {self.df.shape} | Temps: {time.time()-start_time:.2f}s")
        return self.df

    def _generate_synthetic_data(self):
        """Génère un dataset de secours si le fichier est absent."""
        n = 5000
        data = pd.DataFrame(np.random.randn(n, 10), columns=[f'feat_{i}' for i in range(10)])
        data['label'] = np.random.randint(0, 2, n)
        return data

    def profile_data(self):
        """Génère un rapport de profiling (Phase 1)."""
        task_name = "Data Profiling"
        self.logger.info(f"START: {task_name}")

        target = self.config['data']['target']
        if target not in self.df.columns:
            target = self.df.columns[-1]

        counts = self.df[target].value_counts()
        total = len(self.df)

        self.logger.info(f"Rapport Phase 1: Total Rows = {total}")
        for val, count in counts.items():
            self.logger.info(f"Classe {val}: {count} ({count/total*100:.2f}%)")

        # Plot distribution
        plt.figure(figsize=(8, 5))
        sns.countplot(data=self.df, x=target)
        plt.title("Répartition des Classes")
        plt.savefig("plots/global/phase1_distribution.png")
        plt.close()

        self.logger.info(f"END: {task_name} | Graphique sauvegardé dans plots/global/")

# ==============================================================================
# 3. PREPROCESSING
# ==============================================================================

class ExpertPreprocessor:
    """Gère le nettoyage, l'encodage et le split des données."""

    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def process(self, df: pd.DataFrame) -> Tuple:
        """Exécute le pipeline de prétraitement complet."""
        task_name = "Prétraitement des données"
        self.logger.info(f"START: {task_name}")

        # 1. Séparation X, y
        target = self.config['data']['target']
        if target not in df.columns: target = df.columns[-1]
        X = df.drop(target, axis=1)
        y = df[target]

        # 2. Encodage des labels si nécessaire
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            self.logger.info("Labels encodés avec LabelEncoder.")

        # 3. Nettoyage des features (numériques uniquement pour cette démo)
        X = X.select_dtypes(include=[np.number]).fillna(0)

        # 4. Splits
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['data']['test_size'], stratify=y, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=self.config['data']['val_size'], stratify=y_train, random_state=42
        )

        # 5. Scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Conversion en DF pour garder les noms de colonnes
        X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_val = pd.DataFrame(X_val_scaled, columns=X.columns)
        X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

        self.logger.info(f"Splits finaux: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        self.logger.info(f"END: {task_name} | Features: {X.shape[1]}")
        return X_train, X_val, X_test, y_train, y_val, y_test

# ==============================================================================
# 4. MODEL FACTORY & TUNING
# ==============================================================================

class ExpertModelTrainer:
    """Gère l'entraînement et le tuning des modèles."""

    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.best_models = {}

    def get_model_template(self, name: str):
        """Retourne une instance vierge du modèle demandé."""
        templates = {
            "LR": LogisticRegression(max_iter=1000),
            "DT": DecisionTreeClassifier(),
            "RF": RandomForestClassifier(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }
        return templates.get(name)

    def tune_and_train(self, X_train, y_train, X_val, y_val):
        """Exécute le tuning d'hyperparamètres pour chaque modèle."""
        task_name = "Tuning et Entraînement"
        self.logger.info(f"START: {task_name}")

        param_grids = {
            "LR": {"C": [0.1, 1.0, 10.0]},
            "DT": {"max_depth": [5, 10, 20, None]},
            "RF": {"n_estimators": [50, 100], "max_depth": [10, 20]},
            "XGBoost": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]}
        }

        for model_name in self.config['models']:
            if model_name not in param_grids: continue

            subtask = f"Tuning {model_name}"
            self.logger.info(f"  START: {subtask}")

            search = RandomizedSearchCV(
                self.get_model_template(model_name),
                param_distributions=param_grids[model_name],
                n_iter=self.config['tuning']['n_iter'],
                cv=self.config['tuning']['cv'],
                random_state=42,
                n_jobs=-1
            )

            start_t = time.time()
            search.fit(X_train, y_train)

            self.best_models[model_name] = search.best_estimator_
            self.logger.info(f"  RESULT: Best Params for {model_name}: {search.best_params_}")
            self.logger.info(f"  END: {subtask} | Temps: {time.time()-start_t:.2f}s")

        self.logger.info(f"END: {task_name}")

# ==============================================================================
# 5. EVALUATION & XAI
# ==============================================================================

class ExpertEvaluator:
    """Gère l'évaluation des performances et les explications XAI."""

    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.results = []

    def evaluate_performance(self, models: Dict, X_test, y_test):
        """Calcule les métriques de performance finales."""
        task_name = "Évaluation finale"
        self.logger.info(f"START: {task_name}")

        for name, model in models.items():
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds

            metrics = {
                "model": name,
                "accuracy": accuracy_score(y_test, preds),
                "f1": f1_score(y_test, preds),
                "precision": precision_score(y_test, preds),
                "recall": recall_score(y_test, preds),
                "auc": roc_auc_score(y_test, probs)
            }
            self.results.append(metrics)
            self.logger.info(f"  Metrics {name}: F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}")

        # Save metrics to JSON
        with open("metrics/final_results.json", "w") as f:
            json.dump(self.results, f, indent=4)

        self._plot_performance_comparison()
        self.logger.info(f"END: {task_name}")

    def _plot_performance_comparison(self):
        df_res = pd.DataFrame(self.results)
        df_melt = df_res.melt(id_vars="model", var_name="metric", value_name="score")

        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_melt, x="model", y="score", hue="metric")
        plt.title("Comparaison des Performances par Modèle")
        plt.ylim(0, 1.1)
        plt.savefig("plots/global/performance_comparison.png")
        plt.close()

    def explain_xai(self, models: Dict, X_train, X_test):
        """Génère des explications SHAP et LIME."""
        task_name = "Génération XAI"
        self.logger.info(f"START: {task_name}")

        # On prend le meilleur modèle (ex: RF) pour les explications détaillées
        model_name = "RF" if "RF" in models else list(models.keys())[0]
        model = models[model_name]

        # 1. SHAP Global
        self.logger.info(f"  START: SHAP Global ({model_name})")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test.iloc[:self.config['xai']['n_samples_global']])

        plt.figure()
        shap.summary_plot(shap_values, X_test.iloc[:self.config['xai']['n_samples_global']], show=False)
        plt.savefig(f"plots/global/shap_summary_{model_name}.png")
        plt.close()
        self.logger.info(f"  END: SHAP Global | Plot: plots/global/shap_summary_{model_name}.png")

        # 2. LIME Local
        self.logger.info(f"  START: LIME Local ({model_name})")
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=X_train.columns.tolist(),
            class_names=['Normal', 'Attack'],
            mode='classification'
        )

        for i in range(self.config['xai']['n_samples_local']):
            exp = lime_explainer.explain_instance(X_test.iloc[i], model.predict_proba, num_features=5)
            exp.save_to_file(f"plots/local/lime_exp_{model_name}_sample_{i}.html")

        self.logger.info(f"  END: LIME Local | Plot: plots/local/lime_exp_{model_name}_sample_0.html")
        self.logger.info(f"END: {task_name}")

# ==============================================================================
# 6. MAIN ORCHESTRATOR
# ==============================================================================

def main():
    # Initialisation
    create_directory_structure()
    logger = setup_expert_logging(Path("logs/pipeline.log"))

    logger.info("!!! DÉMARRAGE DU PIPELINE EXPERT DDOS !!!")
    start_total = time.time()

    try:
        # Phase 1: Loading
        loader = ExpertDataLoader(DEFAULT_CONFIG, logger)
        df = loader.load_data()
        loader.profile_data()

        # Phase 2: Preprocessing
        preprocessor = ExpertPreprocessor(DEFAULT_CONFIG, logger)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.process(df)

        # Phase 3: Tuning & Training
        trainer = ExpertModelTrainer(DEFAULT_CONFIG, logger)
        trainer.tune_and_train(X_train, y_train, X_val, y_val)

        # Phase 4: Evaluation
        evaluator = ExpertEvaluator(DEFAULT_CONFIG, logger)
        evaluator.evaluate_performance(trainer.best_models, X_test, y_test)

        # Phase 5: XAI
        evaluator.explain_xai(trainer.best_models, X_train, X_test)

        # Final Summary
        duration = time.time() - start_total
        logger.info(f"!!! PIPELINE TERMINÉ AVEC SUCCÈS EN {duration:.2f}s !!!")
        print(f"\nRésultats disponibles dans le dossier 'plots/', 'metrics/' et 'logs/'.")

    except Exception as e:
        logger.error(f"ERREUR CRITIQUE DANS LE PIPELINE: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
