import os
import time
import polars as pl
from src.core.dag.task import Task
from src.core.dag.context import DAGContext
from src.core.dag.result import TaskResult
from src.core.contracts.artifacts import TableArtifact
from src.app.pipeline.registry import TaskRegistry

@TaskRegistry.register("T01_ConsolidateCIC")
class T01_ConsolidateCIC(Task):
    def run(self, context: DAGContext) -> TaskResult:
        from src.infra.resources.monitor import ResourceMonitor
        monitor = ResourceMonitor(context.event_bus, context.run_id)
        monitor.snapshot(self.name)
        
        start_ts = time.time()
        cfg = context.config
        cic_dir = cfg.paths.cic_dir_path
        output_path = os.path.join(cfg.paths.work_dir, "data", "cic_consolidated.parquet")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        context.logger.info("loading", f"Consolidating CIC datasets from {cic_dir}")
        
        # List all CSV files
        if not os.path.exists(cic_dir):
            return TaskResult(task_name=self.name, status="failed", duration_s=time.time() - start_ts, error=f"Répertoire CIC introuvable : {cic_dir}")
            
        csv_files = []
        for root, dirs, files in os.walk(cic_dir):
            for f in files:
                if f.endswith(".csv"):
                    csv_files.append(os.path.join(root, f))
        
        if not csv_files:
            context.logger.error("loading", "ERREUR : Aucun fichier CSV trouvé pour CIC-DDoS2019. Interruption.")
            return TaskResult(task_name=self.name, status="failed", duration_s=time.time() - start_ts, error="No CIC CSV files found")

        if cfg.test_mode:
            csv_files = csv_files[:cfg.test_max_files_per_dataset]

        dfs = []
        for f in csv_files:
            # Lecture EAGER (immédiate) pour stabiliser le schéma de chaque fichier
            try:
                # Nettoyage des noms de colonnes (suppression des espaces)
                df_temp = pl.read_csv(
                    f, 
                    infer_schema_length=1000, 
                    null_values=["Infinity", "NaN", "nan", "inf", "None"],
                    ignore_errors=True,
                    n_rows=cfg.test_row_limit_per_file if cfg.test_mode else None
                )
                
                # Normalisation des noms de colonnes (trim spaces)
                df_temp.columns = [c.strip() for c in df_temp.columns]
                
                # Unification agressive des types pour éviter les conflits Float64/String
                # On définit les colonnes qui DOIVENT être des chaînes
                string_cols = {"Label", "Timestamp", "source_file", "Flow ID", "Source IP", "Destination IP"}
                
                # Toutes les autres colonnes sont forcées en Float64
                # Si une valeur n'est pas convertible, elle devient null (ex: "Infinity")
                cast_exprs = []
                for col in df_temp.columns:
                    if col in string_cols:
                        cast_exprs.append(pl.col(col).cast(pl.String))
                    else:
                        cast_exprs.append(pl.col(col).cast(pl.Float64, strict=False))
                
                df_temp = df_temp.with_columns(cast_exprs)
                df_temp = df_temp.with_columns(pl.lit(os.path.basename(f)).alias("source_file"))
                dfs.append(df_temp)
                context.logger.info("loading", f"  [OK] {os.path.basename(f)} chargé ({df_temp.height} lignes)")
            except Exception as e:
                context.logger.warning("loading", f"  [SKIP] Erreur sur {f}: {e}")

        if not dfs:
            return TaskResult(task_name=self.name, status="failed", duration_s=time.time() - start_ts, error="No valid CIC CSV files loaded")

        # Concaténation EAGER diagonale
        df_full = pl.concat(dfs, how="diagonal")

        # --- SAMPLING STRATIFIÉ (50%) ---
        sampling_ratio = cfg.sample_ratio
        context.logger.info("sampling", f"Performing stratified sampling at {sampling_ratio*100}%")
        
        # On groupe par Label pour la stratification
        df = df_full.filter(pl.col("Label").is_not_null()).group_by("Label").map_groups(
            lambda group: group.sample(fraction=sampling_ratio, seed=cfg.seed)
        )
        
        context.logger.info("sampling", f"Sampling complete: {df_full.height} -> {df.height} rows")

        # --- VALIDATION INTERACTIVE ---
        print(f"\n--- VALIDATION DES DONNÉES CIC-DDoS2019 ---")
        print(f"Fichiers trouvés : {len(csv_files)}")
        print(f"Nombre de colonnes (features) : {df.width}")
        print(f"Nombre de lignes (rows) : {df.height}")
        
        # Proposition DDoS / Pas DDoS
        if "Label" in df.columns:
            counts = df["Label"].value_counts().to_dicts()
            print("Répartition proposée (Labels) :")
            for r in counts:
                label = r.get("Label") or r.get("y")
                print(f"  - {label} : {r['count']} lignes")
        
        print(f"\nListe des features : {df.columns[:20]} ... (+{max(0, len(df.columns)-20)})")
        
        print("\nExemple de donnée au hasard :")
        print(df.sample(1).to_pandas().iloc[0].to_dict())
        
        try:
            confirm = input("\n?> Confirmez-vous l'utilisation de ces données CIC ? (o/n) : ").lower()
            if confirm != 'o':
                return TaskResult(task_name=self.name, status="failed", duration_s=time.time() - start_ts, error="Validation utilisateur refusée pour CIC")
        except EOFError:
            pass # Mode non-interactif
        
        # Add global sample_id
        df = df.with_row_index("sample_id")
        
        # Basic cleaning: rename 'Label' to 'y' if needed or create 'y'
        if "Label" in df.columns:
            df = df.with_columns([
                pl.when(pl.col("Label") == "BENIGN").then(0).otherwise(1).alias("y")
            ])
        
        y_counts = df.group_by("y").count().to_dicts()
        label_balance = {str(r["y"]): int(r["count"]) for r in y_counts}

        context.table_io.write_parquet(df, output_path)
        
        artifact = TableArtifact(
            artifact_id="cic_consolidated",
            name="CIC Consolidated",
            path=output_path,
            format="parquet",
            n_rows=df.height,
            n_cols=df.width,
            columns=df.columns,
            dtypes={col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
            version="1.0.0",
            source_step=self.name,
            fingerprint=str(hash(output_path)),
            stats={"label_balance": label_balance}
        )
        context.artifact_store.save_table(artifact)
        
        context.logger.info("writing", f"CIC Consolidation complete. Output: {output_path}", 
                            artifact=artifact.model_dump())
        
        monitor.snapshot(self.name)
        return TaskResult(task_name=self.name, status="ok", duration_s=time.time() - start_ts, outputs=["cic_consolidated"])
