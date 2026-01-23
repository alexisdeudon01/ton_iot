import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import psutil
from typing import List, Dict, Tuple

# PyMCDM imports
import pymcdm
from pymcdm.methods import TOPSIS, WSM
from scipy.stats import spearmanr

# pymoo imports
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

class DDoSDecisionAgent:
    """
    Système d'Aide à la Décision (SAD) IA complet (MOO-MCDM-MCDA).
    Optimise le choix d'algorithmes selon Performance, Explicabilité et Ressources.
    Utilise des poids précis définis en configuration.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.hierarchy = config['mcdm_hierarchy']
        
        # Profils Utilisateurs (Poids : Performance, Explicabilité, Ressources)
        self.profiles = {
            "A": np.array([0.70, 0.15, 0.15]), # Focus Performance
            "B": np.array([0.15, 0.70, 0.15])  # Focus Explicabilité
        }

    def get_system_load_penalty(self) -> float:
        """Vérifie la charge actuelle du système local."""
        cpu_load = psutil.cpu_percent(interval=0.1)
        ram_load = psutil.virtual_memory().percent
        if cpu_load > 70 or ram_load > 70:
            return 1.5
        return 1.0

    def get_global_weights(self, profile_weights: np.ndarray = None) -> Dict[str, float]:
        """
        Calcule les poids globaux précis.
        W_global = W_pillar * W_local_criterion
        """
        global_weights = {}
        pillars = self.hierarchy['pillars']
        
        for i, pillar in enumerate(pillars):
            pillar_key = pillar['key']
            # Utilise les poids du profil si fournis, sinon ceux de la hiérarchie
            pillar_weight = profile_weights[i] if profile_weights is not None else pillar['weight']
            
            local_criteria = self.hierarchy['criteria'][pillar_key]
            for crit in local_criteria:
                global_weights[crit['key']] = pillar_weight * crit['weight']
                
        return global_weights

    # --- BLOC 1 : MOO (Génération et Filtrage) ---
    
    def run_moo_phase(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Phase MOO : Agrégation et Front de Pareto."""
        print("\n>>> DÉBUT DE LA PHASE MOO (Multi-Objective Optimization)")
        df = results_df.copy()
        penalty = self.get_system_load_penalty()
        
        for pillar in self.hierarchy['pillars']:
            key = pillar['key']
            criteria_keys = [c['key'] for c in self.hierarchy['criteria'][key]]
            
            # Normalisation Min-Max pour la moyenne
            temp_df = df[criteria_keys].copy()
            for c_key in criteria_keys:
                c_info = next(item for item in self.hierarchy['criteria'][key] if item["key"] == c_key)
                if c_info['type'] == 'benefit':
                    temp_df[c_key] = (df[c_key] - df[c_key].min()) / (df[c_key].max() - df[c_key].min() + 1e-9)
                else:
                    temp_df[c_key] = (df[c_key].max() - df[c_key]) / (df[c_key].max() - df[c_key].min() + 1e-9)
            
            df[f'dim_{key}'] = temp_df.mean(axis=1)
            if key == 'resources' and penalty > 1.0:
                df[f'dim_{key}'] *= (1.0 / penalty)

        objectives = ['dim_performance', 'dim_explainability', 'dim_resources']
        matrix = df[objectives].to_numpy()
        nds = NonDominatedSorting().do(-matrix)
        df['is_pareto_efficient'] = False
        df.loc[df.index[nds[0]], 'is_pareto_efficient'] = True
        return df

    # --- BLOC 2 : MCDM (Modélisation du Choix) ---

    def run_mcdm_phase(self, df: pd.DataFrame, profile_name: str) -> pd.DataFrame:
        """Phase MCDM : Classement via TOPSIS et WSM."""
        print(f"\n>>> DÉBUT DE LA PHASE MCDM (Profil {profile_name})")
        weights = self.profiles[profile_name]
        objectives = ['dim_performance', 'dim_explainability', 'dim_resources']
        matrix = df[objectives].to_numpy()
        types = np.array([1, 1, 1])
        
        df[f'score_topsis_{profile_name}'] = TOPSIS()(matrix, weights, types)
        df[f'score_wsm_{profile_name}'] = (matrix * weights).sum(axis=1)
        return df

    # --- BLOC 3 : MCDA (Analyse et Justification) ---

    def run_mcda_analysis(self, df: pd.DataFrame, profile_name: str) -> Dict:
        """Phase MCDA : Analyse de sensibilité et corrélation."""
        print(f"\n>>> DÉBUT DE LA PHASE MCDA (Profil {profile_name})")
        original_weights = self.profiles[profile_name].copy()
        idx_main = 0 if profile_name == "A" else 1
        original_winner = df.sort_values(by=f'score_topsis_{profile_name}', ascending=False).iloc[0]['model']
        
        test_weights = original_weights.copy()
        test_weights[idx_main] *= 1.05
        test_weights /= test_weights.sum()
        
        objectives = ['dim_performance', 'dim_explainability', 'dim_resources']
        new_scores = TOPSIS()(df[objectives].to_numpy(), test_weights, np.array([1, 1, 1]))
        new_winner = df.iloc[np.argmax(new_scores)]['model']
        
        corr, _ = spearmanr(df[f'score_topsis_{profile_name}'], df[f'score_wsm_{profile_name}'])
        return {"is_stable": original_winner == new_winner, "spearman_corr": corr, "winner": original_winner}

    def rank_models(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Wrapper pour maintenir la compatibilité avec le pipeline."""
        df = self.run_moo_phase(results_df)
        df = self.run_mcdm_phase(df, "A")
        return df

    def visualize_sad(self, results_df: pd.DataFrame, output_dir: str):
        """Génère les graphiques Heatmap, Unified, 3D et Radar."""
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('ggplot')
        
        df = self.run_moo_phase(results_df)
        df = self.run_mcdm_phase(df, "A")
        df = self.run_mcdm_phase(df, "B")
        
        all_crit_info = [c for p in self.hierarchy['criteria'].values() for c in p]
        criteria_keys = [c['key'] for c in all_crit_info]
        norm_df = self.normalize_matrix(results_df, all_crit_info)

        # 1. Heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(norm_df.set_index('model')[criteria_keys], annot=True, cmap='YlGnBu', fmt='.2f')
        plt.title('Matrice de Décision Normalisée')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'decision_matrix_heatmap.png'))
        plt.close()

        # 2. Diagramme Unifié
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        dims = ['dim_performance', 'dim_explainability', 'dim_resources']
        titles = ['Performance', 'Explicabilité', 'Ressources']
        colors = ['#3498db', '#e67e22', '#2ecc71']
        for i, dim in enumerate(dims):
            df_sorted = df.sort_values(by=dim, ascending=True)
            axes[i].barh(df_sorted['model'], df_sorted[dim], color=colors[i])
            axes[i].set_title(titles[i])
        plt.suptitle('Comparaison Unifiée des Dimensions')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'unified_dimensions_comparison.png'))
        plt.close()

        # 3. Scatter 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for i, row in df.iterrows():
            color = 'green' if row['is_pareto_efficient'] else 'red'
            ax.scatter(row['dim_performance'], row['dim_explainability'], row['dim_resources'], c=color, s=100)
            ax.text(row['dim_performance'], row['dim_explainability'], row['dim_resources'], row['model'])
        ax.set_xlabel('Performance')
        ax.set_ylabel('Explicabilité')
        ax.set_zlabel('Ressources')
        plt.savefig(os.path.join(output_dir, '3d_solution_space.png'))
        plt.close()

        # 4. Radar
        winner_a = df.sort_values(by='score_topsis_A', ascending=False).iloc[0]
        angles = np.linspace(0, 2*np.pi, len(dims), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        for i, row in df.iterrows():
            values = row[dims].tolist()
            values += values[:1]
            ax.plot(angles, values, label=row['model'], linewidth=4 if row['model'] == winner_a['model'] else 1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['Performance', 'Explicabilité', 'Ressources'])
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.savefig(os.path.join(output_dir, 'radar_comparison.png'))
        plt.close()

    def normalize_matrix(self, df: pd.DataFrame, criteria_info: List[Dict]) -> pd.DataFrame:
        norm_df = df.copy()
        for crit in criteria_info:
            key = crit['key']
            if key not in df.columns: continue
            values = df[key].to_numpy().astype(float)
            if crit['type'] == 'benefit':
                max_val = np.max(values)
                norm_df[key] = values / (max_val + 1e-9)
            else:
                min_val = np.min(values)
                norm_df[key] = min_val / (values + 1e-9)
        return norm_df

    def generate_final_report(self, results_df: pd.DataFrame) -> str:
        df = self.run_moo_phase(results_df)
        report = ["# RAPPORT DU SYSTÈME D'AIDE À LA DÉCISION (SAD)"]
        for p in ["A", "B"]:
            df = self.run_mcdm_phase(df, p)
            analysis = self.run_mcda_analysis(df, p)
            report.append(f"\n## RÉSULTAT PROFIL {p}")
            report.append(f"- **Gagnant** : **{analysis['winner']}**")
            report.append(f"- **Consensus** : {analysis['spearman_corr']:.4f}")
        return "\n".join(report)
