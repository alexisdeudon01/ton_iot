#!/usr/bin/env python3
"""
Génère un diagramme ER (Entity-Relationship) des structures de données
du projet IRP DDoS Detection.

Auteur: Système Expert IA
Date: 2025-01-21
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
from matplotlib.patches import Rectangle, Ellipse
from typing import List, Tuple, Dict
import numpy as np


class ERDiagramGenerator:
    """Générateur de diagramme Entity-Relationship pour les structures de données IRP."""
    
    def __init__(self, figsize=(16, 12)):
        """Initialise le générateur de diagramme."""
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 12)
        self.ax.axis('off')
        
        # Couleurs pour différents types d'entités
        self.colors = {
            'base': '#E8F4F8',      # Bleu clair pour classes de base
            'data': '#FFF4E6',      # Orange clair pour structures de données
            'model': '#E8F8E8',     # Vert clair pour modèles
            'pipeline': '#F8E8F8',  # Rose clair pour pipelines
            'config': '#FFF8E6',    # Jaune clair pour configuration
        }
        
        # Styles de bordures
        self.border_styles = {
            'base': {'edgecolor': '#2E86AB', 'linewidth': 2},
            'data': {'edgecolor': '#A23B72', 'linewidth': 2},
            'model': {'edgecolor': '#18A558', 'linewidth': 2},
            'pipeline': {'edgecolor': '#D67AB1', 'linewidth': 2},
            'config': {'edgecolor': '#F18F01', 'linewidth': 2},
        }
    
    def draw_entity_box(self, x: float, y: float, width: float, height: float,
                       title: str, attributes: List[str], entity_type: str = 'data'):
        """
        Dessine une boîte d'entité ER.
        
        Args:
            x, y: Position du coin inférieur gauche
            width, height: Dimensions de la boîte
            title: Titre de l'entité
            attributes: Liste des attributs
            entity_type: Type d'entité (détermine la couleur)
        """
        # Rectangle avec coins arrondis
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.1",
            facecolor=self.colors.get(entity_type, '#FFFFFF'),
            **self.border_styles.get(entity_type, {})
        )
        self.ax.add_patch(box)
        
        # Titre (en gras)
        self.ax.text(x + width/2, y + height - 0.3, title,
                    ha='center', va='top', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Ligne de séparation
        self.ax.plot([x, x + width], [y + height - 0.5, y + height - 0.5],
                    color='black', linewidth=1)
        
        # Attributs
        attr_y = y + height - 0.7
        for i, attr in enumerate(attributes):
            if attr_y < y + 0.2:
                break
            # Attribut clé en gras
            if attr.startswith('*') or attr.startswith('PK'):
                attr_display = attr.replace('*', '◆').replace('PK', '◆')
                self.ax.text(x + 0.15, attr_y, attr_display,
                           ha='left', va='top', fontsize=9, fontweight='bold')
            # Attribut foreign key
            elif attr.startswith('FK'):
                attr_display = attr.replace('FK', '◇')
                self.ax.text(x + 0.15, attr_y, attr_display,
                           ha='left', va='top', fontsize=9, style='italic', color='#666666')
            else:
                self.ax.text(x + 0.15, attr_y, attr,
                           ha='left', va='top', fontsize=9)
            attr_y -= 0.35
    
    def draw_relationship(self, x1: float, y1: float, x2: float, y2: float,
                         label: str = '', relation_type: str = 'one_to_many'):
        """
        Dessine une relation entre deux entités.
        
        Args:
            x1, y1: Position de l'entité source
            x2, y2: Position de l'entité cible
            label: Label de la relation
            relation_type: Type de relation ('one_to_one', 'one_to_many', 'many_to_many')
        """
        # Calcul du point de connexion (centre des boîtes)
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # Flèche
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->', mutation_scale=20,
            linewidth=1.5, color='#333333',
            connectionstyle='arc3,rad=0.1'
        )
        self.ax.add_patch(arrow)
        
        # Label de la relation
        if label:
            # Position du label (milieu de la flèche, légèrement décalé)
            offset_x = (x2 - x1) * 0.5
            offset_y = (y2 - y1) * 0.5
            self.ax.text(mid_x + offset_x * 0.3, mid_y + offset_y * 0.3, label,
                        ha='center', va='center', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
    
    def draw_inheritance(self, x_parent: float, y_parent: float,
                        x_child: float, y_child: float):
        """Dessine une relation d'héritage (triangle vide)."""
        # Triangle pour l'héritage
        triangle = mpatches.RegularPolygon(
            ((x_parent + x_child) / 2, (y_parent + y_child) / 2),
            numVertices=3, radius=0.3,
            orientation=np.pi/2,
            facecolor='white', edgecolor='#333333', linewidth=1.5
        )
        self.ax.add_patch(triangle)
        
        # Ligne vers le parent
        self.ax.plot([x_parent, (x_parent + x_child) / 2],
                    [y_parent, (y_parent + y_child) / 2 - 0.3],
                    'k-', linewidth=1.5)
        
        # Ligne vers l'enfant
        self.ax.plot([x_child, (x_parent + x_child) / 2],
                    [y_child, (y_parent + y_child) / 2 + 0.3],
                    'k-', linewidth=1.5)
    
    def generate_diagram(self):
        """Génère le diagramme ER complet."""
        # === ENTITÉS PRINCIPALES ===
        
        # 1. IRPBaseStructure (classe de base)
        self.draw_entity_box(
            1, 9, 1.8, 2.5,
            "IRPBaseStructure",
            [
                "*metadata: Dict[str, Any]",
                "*project_name: str",
                "add_metadata(key, value)"
            ],
            entity_type='base'
        )
        
        # 2. IRPDataFrame (hérite de IRPBaseStructure)
        self.draw_entity_box(
            1, 6, 1.8, 2.5,
            "IRPDataFrame",
            [
                "Hérite de: pd.DataFrame",
                "Hérite de: IRPBaseStructure",
                "*_constructor property",
                "+ Méthodes pandas standard",
                "+ Méthodes IRP custom"
            ],
            entity_type='data'
        )
        
        # 3. IRPDaskFrame (hérite de IRPBaseStructure)
        self.draw_entity_box(
            1, 3, 1.8, 2.5,
            "IRPDaskFrame",
            [
                "Hérite de: dd.DataFrame",
                "Hérite de: IRPBaseStructure",
                "*expr, name, meta, divisions",
                "+ Méthodes dask standard",
                "+ Méthodes IRP custom"
            ],
            entity_type='data'
        )
        
        # 4. NetworkFlow (hérite de IRPBaseStructure)
        self.draw_entity_box(
            4, 6, 2, 3,
            "NetworkFlow",
            [
                "*flow_id: str",
                "*source_ip: str",
                "*dest_ip: str",
                "*packets: List[pd.Series]",
                "*start_time: Optional[float]",
                "*end_time: Optional[float]",
                "add_packet(packet, ts_col)",
                "get_direction(packet)",
                "to_dataframe()"
            ],
            entity_type='data'
        )
        
        # 5. Packet (entité implicite)
        self.draw_entity_box(
            7, 7.5, 1.8, 2,
            "Packet",
            [
                "*timestamp (ts)",
                "*src_ip: str",
                "*dst_ip: str",
                "*protocol",
                "*size: int",
                "+ autres features",
                "(représenté comme pd.Series)"
            ],
            entity_type='data'
        )
        
        # 6. DatasetLoader
        self.draw_entity_box(
            4, 2, 2, 2.5,
            "DatasetLoader",
            [
                "*data_dir: Path",
                "*monitor: SystemMonitor",
                "*loaded_files: Set[str]",
                "*progress_callback",
                "load_cic_ddos2019()",
                "load_ton_iot()",
                "get_adaptive_chunk_size()"
            ],
            entity_type='pipeline'
        )
        
        # 7. PreprocessingPipeline
        self.draw_entity_box(
            7, 2, 2, 3,
            "PreprocessingPipeline",
            [
                "*random_state: int",
                "*n_features: int",
                "*scaler: RobustScaler",
                "*imputer: SimpleImputer",
                "*feature_selector",
                "*is_fitted: bool",
                "prepare_data(X, y)",
                "transform_test(X_test)",
                "clean_data()",
                "select_features()",
                "scale_features()"
            ],
            entity_type='pipeline'
        )
        
        # 8. PipelineConfig
        self.draw_entity_box(
            7, 5.5, 2, 2.5,
            "PipelineConfig",
            [
                "*test_mode: bool",
                "*sample_ratio: float",
                "*random_state: int",
                "*output_dir: str",
                "*phase1_configs: int",
                "*preprocessing_options: Dict",
                "*phase3_algorithms: List[str]",
                "+ autres paramètres"
            ],
            entity_type='config'
        )
        
        # 9. ModelRegistry
        self.draw_entity_box(
            4, 9, 2, 2,
            "ModelRegistry",
            [
                "*registry: Dict[str, Callable]",
                "get_model_registry(config)",
                "+ LR, DT, RF, KNN",
                "+ CNN (optionnel)",
                "+ TabNet (optionnel)"
            ],
            entity_type='model'
        )
        
        # === RELATIONS D'HÉRITAGE ===
        
        # IRPBaseStructure -> IRPDataFrame
        self.draw_inheritance(1.9, 9, 1.9, 8.5)
        
        # IRPBaseStructure -> IRPDaskFrame
        self.draw_inheritance(1.9, 9, 1.9, 5.5)
        
        # IRPBaseStructure -> NetworkFlow
        self.draw_inheritance(2.9, 9, 4, 9)
        
        # === RELATIONS DE COMPOSITION/AGGRÉGATION ===
        
        # NetworkFlow -> Packet (1-to-many)
        self.draw_relationship(6, 7.5, 7, 8.5, "contient (1:N)", "one_to_many")
        
        # NetworkFlow -> IRPDataFrame (conversion)
        self.draw_relationship(5.5, 6, 2.8, 7.5, "to_dataframe()", "one_to_one")
        
        # DatasetLoader -> IRPDaskFrame (produit)
        self.draw_relationship(5, 4.5, 2.8, 3, "charge →", "one_to_one")
        
        # PreprocessingPipeline -> IRPDataFrame/IRPDaskFrame (traite)
        self.draw_relationship(7, 5, 2.8, 7, "traite →", "one_to_many")
        
        # PipelineConfig -> PreprocessingPipeline (configure)
        self.draw_relationship(7, 6, 7, 5, "configure →", "one_to_one")
        
        # PipelineConfig -> ModelRegistry (configure)
        self.draw_relationship(7, 7.5, 6, 10, "configure →", "one_to_one")
        
        # PreprocessingPipeline -> ModelRegistry (prépare pour)
        self.draw_relationship(7, 3.5, 6, 9, "prépare →", "many_to_many")
        
        # === LÉGENDE ===
        legend_x = 0.2
        legend_y = 0.5
        
        # Titre légende
        self.ax.text(legend_x, legend_y + 1.2, "LÉGENDE", fontsize=12, fontweight='bold')
        
        # Types d'entités
        legend_items = [
            ('Classe de base', self.colors['base'], self.border_styles['base']['edgecolor']),
            ('Structure de données', self.colors['data'], self.border_styles['data']['edgecolor']),
            ('Modèle', self.colors['model'], self.border_styles['model']['edgecolor']),
            ('Pipeline', self.colors['pipeline'], self.border_styles['pipeline']['edgecolor']),
            ('Configuration', self.colors['config'], self.border_styles['config']['edgecolor']),
        ]
        
        y_pos = legend_y
        for label, color, edge_color in legend_items:
            # Boîte exemple
            box = Rectangle((legend_x, y_pos), 0.4, 0.3, facecolor=color, 
                          edgecolor=edge_color, linewidth=1.5)
            self.ax.add_patch(box)
            self.ax.text(legend_x + 0.5, y_pos + 0.15, label, 
                        fontsize=9, va='center')
            y_pos -= 0.4
        
        # Relations
        y_pos -= 0.3
        self.ax.text(legend_x, y_pos, "Relations:", fontsize=10, fontweight='bold')
        y_pos -= 0.3
        
        # Héritage
        triangle = mpatches.RegularPolygon(
            (legend_x + 0.2, y_pos), 
            numVertices=3, 
            radius=0.15,
            orientation=np.pi/2, 
            facecolor='white', 
            edgecolor='black'
        )
        self.ax.add_patch(triangle)
        self.ax.text(legend_x + 0.5, y_pos, "Héritage", fontsize=9, va='center')
        y_pos -= 0.3
        
        # Flèche
        arrow = FancyArrowPatch((legend_x, y_pos), (legend_x + 0.4, y_pos),
                               arrowstyle='->', mutation_scale=15, linewidth=1.5)
        self.ax.add_patch(arrow)
        self.ax.text(legend_x + 0.5, y_pos, "Relation (1:N, N:M, etc.)", fontsize=9, va='center')
        
        # Attributs
        y_pos -= 0.4
        self.ax.text(legend_x, y_pos, "Attributs:", fontsize=10, fontweight='bold')
        y_pos -= 0.3
        self.ax.text(legend_x, y_pos, "◆ Clé primaire", fontsize=9, fontweight='bold')
        y_pos -= 0.25
        self.ax.text(legend_x, y_pos, "◇ Clé étrangère", fontsize=9, style='italic', color='#666666')
        
        # === TITRE ===
        self.ax.text(5, 11.5, "DIAGRAMME ER - STRUCTURES DE DONNÉES IRP DDoS DETECTION",
                    ha='center', va='center', fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F4F8', edgecolor='#2E86AB', linewidth=2))
        
        # Sous-titre
        self.ax.text(5, 11, "Relations entre les entités et classes de données du projet",
                    ha='center', va='center', fontsize=11, style='italic')
        
        plt.tight_layout()
    
    def save(self, output_path: str = "output/datastructure_er_diagram.png", dpi=300):
        """Sauvegarde le diagramme."""
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"✅ Diagramme ER sauvegardé: {output_path}")
    
    def show(self):
        """Affiche le diagramme."""
        plt.show()


def main():
    """Point d'entrée principal."""
    from pathlib import Path
    
    # Créer le répertoire de sortie
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Générer le diagramme
    generator = ERDiagramGenerator(figsize=(16, 12))
    generator.generate_diagram()
    generator.save(output_dir / "datastructure_er_diagram.png")
    
    print("\n" + "="*60)
    print("DIAGRAMME ER GÉNÉRÉ AVEC SUCCÈS")
    print("="*60)
    print("\nEntités représentées:")
    print("  • IRPBaseStructure (classe de base)")
    print("  • IRPDataFrame (DataFrame Pandas personnalisé)")
    print("  • IRPDaskFrame (DataFrame Dask personnalisé)")
    print("  • NetworkFlow (flux réseau avec paquets)")
    print("  • Packet (paquet réseau)")
    print("  • DatasetLoader (chargeur de datasets)")
    print("  • PreprocessingPipeline (pipeline de prétraitement)")
    print("  • PipelineConfig (configuration)")
    print("  • ModelRegistry (registre des modèles)")
    print("\nRelations:")
    print("  • Héritage (IRPBaseStructure → IRPDataFrame, IRPDaskFrame, NetworkFlow)")
    print("  • Composition (NetworkFlow contient plusieurs Packets)")
    print("  • Utilisation (DatasetLoader produit IRPDaskFrame)")
    print("  • Configuration (PipelineConfig configure les pipelines)")


if __name__ == "__main__":
    main()
