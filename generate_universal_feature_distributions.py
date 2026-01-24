import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from universal_features import UniversalFeatureEngineer, UNIVERSAL_FEATURES

def analyze_distribution(data, name):
    """Analyse complète d'une distribution selon les critères demandés."""
    data = data.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(data) < 10:
        return {"type": "Insufficient data", "skew": 0, "kurt": 0, "zero_pct": 0, "transform": "None"}
    
    # Statistiques de base
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    n_unique = data.nunique()
    zero_pct = (data == 0).sum() / len(data) * 100
    
    # Test de normalité (sur échantillon si trop grand)
    sample = data.sample(min(5000, len(data)), random_state=42)
    try:
        shapiro_p = stats.shapiro(sample)[1]
    except:
        shapiro_p = 0
    
    # Classification selon la logique fournie
    if n_unique < 20:
        dist_type = "Discrète"
        transform = "OneHotEncoder ou garder tel quel"
    elif zero_pct > 50:
        dist_type = "Zero-inflated"
        transform = "Ajouter feature binaire is_zero + log1p sur non-zéros"
    elif shapiro_p > 0.05 and abs(skewness) < 0.5:
        dist_type = "Normale"
        transform = "StandardScaler ou RobustScaler"
    elif skewness > 2 and data.min() >= 0:
        dist_type = "Exponentielle"
        transform = "log1p() puis StandardScaler"
    elif skewness > 1 and data.min() >= 0:
        dist_type = "Log-normale"
        transform = "log1p() puis StandardScaler"
    elif kurtosis > 6:
        dist_type = "Heavy-tailed"
        transform = "RobustScaler + clip outliers"
    elif abs(skewness) < 0.5 and kurtosis < -0.5:
        dist_type = "Uniforme"
        transform = "MinMaxScaler"
    else:
        # Détection bimodalité simplifiée
        hist, bin_edges = np.histogram(data, bins=50)
        peaks = np.where((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))[0]
        if len(peaks) >= 2:
            dist_type = "Bimodale"
            transform = "Considérer séparation en 2 features ou clustering"
        else:
            dist_type = "Asymétrique"
            transform = "RobustScaler"
    
    return {
        "type": dist_type,
        "skew": round(float(skewness), 2),
        "kurt": round(float(kurtosis), 2),
        "zero_pct": round(float(zero_pct), 1),
        "transform": transform
    }

def main():
    print("Démarrage de la génération des graphiques de distribution universels...")
    engineer = UniversalFeatureEngineer()
    
    # Chemins des fichiers
    cic_path = './work/data/cic_consolidated.parquet'
    ton_path = './work/data/ton_clean.parquet'
    
    if not os.path.exists(ton_path):
        ton_path = './work/data/ton_cleaned.parquet'

    print(f"Chargement de CIC: {cic_path}")
    df_cic_raw = pd.read_parquet(cic_path)
    print(f"Chargement de TON: {ton_path}")
    df_ton_raw = pd.read_parquet(ton_path)

    print("Transformation via UniversalFeatureEngineer...")
    cic_universal = engineer.transform_cic(df_cic_raw)
    ton_universal = engineer.transform_ton(df_ton_raw)

    # Sampling 50% si >100k lignes
    if len(cic_universal) > 100000:
        print("Sampling CIC (50%)...")
        cic_universal = cic_universal.sample(frac=0.5, random_state=42)
    if len(ton_universal) > 100000:
        print("Sampling TON (50%)...")
        ton_universal = ton_universal.sample(frac=0.5, random_state=42)

    # Style exact demandé
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['axes.labelsize'] = 9

    analysis_results = []
    
    # --- FIGURE 1 : Features 0-9 (10 subplots, 2x5) ---
    part1_features = UNIVERSAL_FEATURES[:10]
    fig1, axes1 = plt.subplots(2, 5, figsize=(24, 10))
    axes1 = axes1.flatten()
    
    for j, feat in enumerate(part1_features):
        ax = axes1[j]
        
        # Analyse
        c_data = cic_universal[feat].replace([np.inf, -np.inf], np.nan).dropna()
        t_data = ton_universal[feat].replace([np.inf, -np.inf], np.nan).dropna()
        
        c_stats = analyze_distribution(c_data, feat)
        t_stats = analyze_distribution(t_data, feat)
        
        # Stockage pour CSV
        analysis_results.append({
            "feature": feat,
            "cic_mean": c_data.mean(), "cic_std": c_data.std(), "cic_min": c_data.min(), "cic_max": c_data.max(),
            "cic_skew": c_stats["skew"], "cic_kurt": c_stats["kurt"], "cic_zero_pct": c_stats["zero_pct"], "cic_dist_type": c_stats["type"],
            "ton_mean": t_data.mean(), "ton_std": t_data.std(), "ton_min": t_data.min(), "ton_max": t_data.max(),
            "ton_skew": t_stats["skew"], "ton_kurt": t_stats["kurt"], "ton_zero_pct": t_stats["zero_pct"], "ton_dist_type": t_stats["type"],
            "recommended_transform": c_stats["transform"]
        })
        
        # Clipping au percentile 99 pour la visualisation
        limit = max(c_data.quantile(0.99), t_data.quantile(0.99))
        v_cic = c_data[c_data <= limit]
        v_ton = t_data[t_data <= limit]
        
        # Plot
        sns.histplot(v_cic, color="#3498db", alpha=0.6, bins=50, stat="density", label="CIC", ax=ax)
        sns.histplot(v_ton, color="#e74c3c", alpha=0.6, bins=50, stat="density", label="TON", ax=ax)
        sns.kdeplot(v_cic, color="#3498db", linewidth=2, ax=ax)
        sns.kdeplot(v_ton, color="#e74c3c", linewidth=2, ax=ax)
        
        ax.set_title(f"{feat}\nType: {c_stats['type']}")
        ax.set_xlabel("Valeur")
        ax.set_ylabel("Densité")
        ax.legend(loc='upper right', fontsize='x-small')
        
        # Annotation coin inférieur droit
        annot = (f"CIC: μ={c_data.mean():.2e}, σ={c_data.std():.2e}\n"
                 f"TON: μ={t_data.mean():.2e}, σ={t_data.std():.2e}\n"
                 f"Skew: {c_stats['skew']:.2f} | Kurt: {c_stats['kurt']:.2f}\n"
                 f"Zeros: {c_stats['zero_pct']:.1f}%")
        ax.text(0.98, 0.02, annot, transform=ax.transAxes, fontsize=8,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    fig1.suptitle("Distribution des Features Universelles (1/2) - Données Post-Transformation\nComparaison CIC-DDoS2019 vs TON_IoT", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=3.0)
    plt.savefig("./work/reports/universal_features_dist_part1.png", dpi=150)
    plt.close()

    # --- FIGURE 2 : Features 10-14 (5 subplots, 1x5) ---
    part2_features = UNIVERSAL_FEATURES[10:]
    fig2, axes2 = plt.subplots(1, 5, figsize=(24, 5))
    
    for j, feat in enumerate(part2_features):
        ax = axes2[j]
        
        c_data = cic_universal[feat].replace([np.inf, -np.inf], np.nan).dropna()
        t_data = ton_universal[feat].replace([np.inf, -np.inf], np.nan).dropna()
        
        c_stats = analyze_distribution(c_data, feat)
        t_stats = analyze_distribution(t_data, feat)
        
        analysis_results.append({
            "feature": feat,
            "cic_mean": c_data.mean(), "cic_std": c_data.std(), "cic_min": c_data.min(), "cic_max": c_data.max(),
            "cic_skew": c_stats["skew"], "cic_kurt": c_stats["kurt"], "cic_zero_pct": c_stats["zero_pct"], "cic_dist_type": c_stats["type"],
            "ton_mean": t_data.mean(), "ton_std": t_data.std(), "ton_min": t_data.min(), "ton_max": t_data.max(),
            "ton_skew": t_stats["skew"], "ton_kurt": t_stats["kurt"], "ton_zero_pct": t_stats["zero_pct"], "ton_dist_type": t_stats["type"],
            "recommended_transform": c_stats["transform"]
        })
        
        limit = max(c_data.quantile(0.99), t_data.quantile(0.99))
        v_cic = c_data[c_data <= limit]
        v_ton = t_data[t_data <= limit]
        
        sns.histplot(v_cic, color="#3498db", alpha=0.6, bins=50, stat="density", label="CIC", ax=ax)
        sns.histplot(v_ton, color="#e74c3c", alpha=0.6, bins=50, stat="density", label="TON", ax=ax)
        sns.kdeplot(v_cic, color="#3498db", linewidth=2, ax=ax)
        sns.kdeplot(v_ton, color="#e74c3c", linewidth=2, ax=ax)
        
        ax.set_title(f"{feat}\nType: {c_stats['type']}")
        ax.set_xlabel("Valeur")
        ax.set_ylabel("Densité")
        ax.legend(loc='upper right', fontsize='x-small')
        
        annot = (f"CIC: μ={c_data.mean():.2e}, σ={c_data.std():.2e}\n"
                 f"TON: μ={t_data.mean():.2e}, σ={t_data.std():.2e}\n"
                 f"Skew: {c_stats['skew']:.2f} | Kurt: {c_stats['kurt']:.2f}\n"
                 f"Zeros: {c_stats['zero_pct']:.1f}%")
        ax.text(0.98, 0.02, annot, transform=ax.transAxes, fontsize=8,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    fig2.suptitle("Distribution des Features Universelles (2/2) - Données Post-Transformation\nComparaison CIC-DDoS2019 vs TON_IoT", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=3.0)
    plt.savefig("./work/reports/universal_features_dist_part2.png", dpi=150)
    plt.close()

    # Export CSV
    df_analysis = pd.DataFrame(analysis_results).drop_duplicates(subset=['feature'])
    df_analysis.to_csv("./work/reports/universal_features_analysis.csv", index=False)

    # Rapport Markdown
    with open("./work/reports/universal_features_report.md", "w") as f:
        f.write("# Analyse des Distributions - Features Universelles\n\n")
        f.write("## Résumé\n")
        f.write(f"- Nombre de features : 15\n")
        f.write(f"- Échantillons CIC : {len(cic_universal)}\n")
        f.write(f"- Échantillons TON : {len(ton_universal)}\n\n")
        
        for dtype in ["Normale", "Log-normale", "Heavy-tailed", "Zero-inflated", "Asymétrique", "Discrète", "Bimodale", "Uniforme"]:
            feats = [r["feature"] for r in analysis_results if r["cic_dist_type"] == dtype]
            if feats:
                f.write(f"### {dtype}\n")
                for feat in set(feats): f.write(f"- {feat}\n")
                f.write("\n")
        
        f.write("## Recommandations de preprocessing\n\n")
        f.write("| Feature | Type CIC | Type TON | Transformation Recommandée |\n")
        f.write("| --- | --- | --- | --- |\n")
        for r in analysis_results:
            f.write(f"| {r['feature']} | {r['cic_dist_type']} | {r['ton_dist_type']} | {r['recommended_transform']} |\n")
        
        f.write("\n## Observations clés\n")
        shifts = [r["feature"] for r in analysis_results if abs(r["cic_mean"] - r["ton_mean"]) / (abs(r["cic_mean"]) + 1e-9) > 0.5]
        if shifts:
            f.write("- Features avec forte différence de moyenne (Domain Shift potentiel) :\n")
            for s in set(shifts): f.write(f"  - {s}\n")
        f.write("- La plupart des métriques réseau présentent des distributions asymétriques ou à queue lourde.\n")

    print("Génération terminée avec succès.")

if __name__ == "__main__":
    main()
