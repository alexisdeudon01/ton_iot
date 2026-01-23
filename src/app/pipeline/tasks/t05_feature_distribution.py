import os
import time
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.app.pipeline.registry import TaskRegistry

@TaskRegistry.register("T05_FeatureDistribution")
class T05_FeatureDistribution(Task):
    """
    Génère des graphiques de distribution comparative pour chaque feature commune.
    Affiche le nombre et le pourcentage de données pour ToN-IoT et CIC.
    """
    def run(self, context: DAGContext) -> TaskResult:
        start_ts = time.time()
        output_dir = os.path.join(context.config.paths.work_dir, "reports", "feature_distributions")
        os.makedirs(output_dir, exist_ok=True)

        # 1. Charger les données
        cic_art = context.artifact_store.load_table("cic_consolidated")
        ton_art = context.artifact_store.load_table("ton_clean")
        
        df_cic = context.table_io.read_parquet(cic_art.path).collect()
        df_ton = context.table_io.read_parquet(ton_art.path).collect()

        # 2. Identifier les features communes
        exclude = {"y", "Label", "source_file", "sample_id", "type", "ts"}
        common_features = sorted(list((set(df_cic.columns) & set(df_ton.columns)) - exclude))
        
        if not common_features:
            context.logger.warning("visualization", "Aucune feature commune trouvée pour la distribution")
            return TaskResult(task_name=self.name, status="ok", duration_s=time.time() - start_ts)

        # 3. Générer un graphique par feature
        for feat in common_features:
            plt.figure(figsize=(10, 6))
            
            # Préparation des données pour Seaborn
            data_cic = df_cic.select(feat).to_pandas()
            data_ton = df_ton.select(feat).to_pandas()
            
            n_cic = len(data_cic)
            n_ton = len(data_ton)
            total = n_cic + n_ton
            
            sns.kdeplot(data=data_cic[feat], label=f"CIC (n={n_cic}, {n_cic/total:.1%})", fill=True)
            sns.kdeplot(data=data_ton[feat], label=f"ToN-IoT (n={n_ton}, {n_ton/total:.1%})", fill=True)
            
            plt.title(f"Distribution Comparative : {feat}")
            plt.xlabel("Valeur")
            plt.ylabel("Densité")
            plt.legend()
            
            plt.savefig(os.path.join(output_dir, f"dist_{feat}.png"))
            plt.close()

        context.logger.info("visualization", f"Graphiques de distribution générés pour {len(common_features)} features")
        
        return TaskResult(task_name=self.name, status="ok", duration_s=time.time() - start_ts, outputs=[output_dir])
