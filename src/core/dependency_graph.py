import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

def generate_er_dependency_diagram(output_path: Path):
    """Generates an ER-like dependency diagram for the pipeline components."""
    G = nx.DiGraph()

    # Entities (Nodes)
    entities = [
        "ToN-IoT Dataset", "CICDDoS2019 Dataset",
        "RealDataLoader", "SystemMonitor",
        "FeatureCategorization", "PipelineTrainer",
        "PipelineValidator", "XAIManager", "PipelineTester",
        "Consolidated Parquet", "Results (rr/)"
    ]
    G.add_nodes_from(entities)

    # Relationships (Edges)
    relationships = [
        ("ToN-IoT Dataset", "RealDataLoader"),
        ("CICDDoS2019 Dataset", "RealDataLoader"),
        ("SystemMonitor", "RealDataLoader"),
        ("RealDataLoader", "Consolidated Parquet"),
        ("Consolidated Parquet", "FeatureCategorization"),
        ("Consolidated Parquet", "PipelineTrainer"),
        ("PipelineTrainer", "PipelineValidator"),
        ("PipelineValidator", "XAIManager"),
        ("XAIManager", "PipelineTester"),
        ("PipelineTester", "Results (rr/)"),
        ("SystemMonitor", "Results (rr/)")
    ]
    G.add_edges_from(relationships)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold')

    plt.title("Pipeline ER-like Dependency Diagram")
    plt.savefig(output_path)
    plt.close()
    print(f"RÉSULTAT: Diagramme de dépendances sauvegardé dans {output_path}")
