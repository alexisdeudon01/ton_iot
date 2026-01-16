#!/usr/bin/env python3
"""
Script pour organiser les fichiers PNG par type d'algorithme
Supprime les PNG du répertoire racine et les organise dans results/
"""
import os
import shutil
from pathlib import Path

# Mapping des PNG vers leur catégorie et dossier
PNG_ORGANIZATION = {
    # Data analysis
    'correlation.png': 'results/data_analysis/',
    'correlation_matrix.png': 'results/data_analysis/',
    
    # Machine Learning
    'model_performance.png': 'results/machine_learning/',
    'Models.png': 'results/machine_learning/',
    
    # Reinforcement Learning
    'drlperf.png': 'results/reinforcement_learning/',
    
    # Documentation
    'AI_implementation.png': 'results/documentation/',
    'DLAccuracy.png': 'results/documentation/',
    'DLLoss.png': 'results/documentation/',
}

def organize_png_files(root_dir='.'):
    """Organise les fichiers PNG dans la structure de dossiers appropriée"""
    root = Path(root_dir)
    
    # Créer tous les dossiers nécessaires
    for target_dir in set(PNG_ORGANIZATION.values()):
        (root / target_dir).mkdir(parents=True, exist_ok=True)
    
    moved = []
    not_found = []
    
    for png_file, target_dir in PNG_ORGANIZATION.items():
        source = root / png_file
        target = root / target_dir / png_file
        
        if source.exists():
            # Si le fichier existe déjà dans le dossier cible, le supprimer d'abord
            if target.exists():
                target.unlink()
            
            # Déplacer le fichier
            shutil.move(str(source), str(target))
            moved.append((png_file, target_dir))
            print(f"✓ Déplacé: {png_file} -> {target_dir}")
        else:
            not_found.append(png_file)
    
    # Supprimer tous les PNG restants dans le root qui ne sont pas dans la liste
    for png_file in root.glob('*.png'):
        if png_file.name not in PNG_ORGANIZATION:
            # Fichier PNG non catalogué, le déplacer dans documentation
            target = root / 'results/documentation' / png_file.name
            if target.exists():
                target.unlink()
            shutil.move(str(png_file), str(target))
            moved.append((png_file.name, 'results/documentation/'))
            print(f"✓ Déplacé (non catalogué): {png_file.name} -> results/documentation/")
    
    print(f"\n✅ Organisation terminée!")
    print(f"   - {len(moved)} fichier(s) déplacé(s)")
    if not_found:
        print(f"   - {len(not_found)} fichier(s) non trouvé(s): {', '.join(not_found)}")
    
    return moved, not_found

if __name__ == "__main__":
    organize_png_files()
