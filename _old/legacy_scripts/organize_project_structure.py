#!/usr/bin/env python3
"""
Script to reorganize project structure by moving unused files to _old directory
"""
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Current directory
BASE_DIR = Path('.')

# Files and directories to keep (core project files)
CORE_FILES = {
    'main.py',
    'main_pipeline.py',
    'dataset_loader.py',
    'data_harmonization.py',
    'preprocessing_pipeline.py',
    'evaluation_3d.py',
    'ahp_topsis_framework.py',
    'models_cnn.py',
    'models_tabnet.py',
    'requirements.txt',
    'README.md',
    'DIMENSIONS_CALCULATION.md',
    'OUTPUT_EXPECTED.md',
    'ton_iot.code-workspace',
    'train_test_network.csv',
    '.gitignore',
    '.git',
    '__pycache__',
}

# Core directories to keep
CORE_DIRS = {
    'data',
    'output',
    '_old',  # Don't move _old itself
}

def move_to_old(source_path: Path, category: str = ''):
    """Move file/directory to _old with optional category subdirectory"""
    if not source_path.exists():
        logger.warning(f"Path does not exist: {source_path}")
        return False
    
    target_base = BASE_DIR / '_old' / category if category else BASE_DIR / '_old'
    target_base.mkdir(parents=True, exist_ok=True)
    target_path = target_base / source_path.name
    
    # If target exists, add suffix
    if target_path.exists():
        counter = 1
        while target_path.exists():
            target_path = target_base / f"{source_path.stem}_{counter}{source_path.suffix}"
            counter += 1
    
    try:
        if source_path.is_dir():
            shutil.move(str(source_path), str(target_path))
        else:
            shutil.move(str(source_path), str(target_path))
        logger.info(f"Moved: {source_path.name} -> _old/{category}/")
        return True
    except Exception as e:
        logger.error(f"Error moving {source_path}: {e}")
        return False

def main():
    """Main organization function"""
    logger.info("Starting project structure reorganization...")
    
    # Create _old directory structure
    (BASE_DIR / '_old' / 'documents').mkdir(parents=True, exist_ok=True)
    (BASE_DIR / '_old' / 'legacy_scripts').mkdir(parents=True, exist_ok=True)
    (BASE_DIR / '_old' / 'web_artifacts').mkdir(parents=True, exist_ok=True)
    (BASE_DIR / '_old' / 'documentation_old').mkdir(parents=True, exist_ok=True)
    (BASE_DIR / '_old' / 'old_results').mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    
    # Move PDFs
    logger.info("\n=== Moving PDF documents ===")
    pdfs = list(BASE_DIR.glob('*.pdf'))
    for pdf in pdfs:
        if pdf.name not in CORE_FILES:
            if move_to_old(pdf, 'documents'):
                moved_count += 1
    
    # Move legacy scripts
    logger.info("\n=== Moving legacy scripts ===")
    legacy_scripts = [
        'data_training.py',
        'RL_training.py',
        'organize_png.py',
        'repo_sanity_check.py'
    ]
    for script in legacy_scripts:
        script_path = BASE_DIR / script
        if script_path.exists():
            if move_to_old(script_path, 'legacy_scripts'):
                moved_count += 1
    
    # Move web artifact directories
    logger.info("\n=== Moving web artifacts ===")
    web_dirs = [
        'datasetCICDDos2019_files',
        'IEEE Xplore Full-Text PDF__files'
    ]
    for web_dir in web_dirs:
        web_path = BASE_DIR / web_dir
        if web_path.exists() and web_path.is_dir():
            if move_to_old(web_path, 'web_artifacts'):
                moved_count += 1
    
    # Move old documentation files (keep only core ones)
    logger.info("\n=== Moving old documentation ===")
    old_docs = [
        'RIDGE_CLASSIFIER_EXPLANATION.md',
        'PPO_EXPLANATION.md',
        'MODELS_CREATION_ANALYSIS.md',
        'ALGORITHMS_CATEGORIZATION.md',
        'AI_PARAMETERS.md'
    ]
    for doc in old_docs:
        doc_path = BASE_DIR / doc
        if doc_path.exists():
            if move_to_old(doc_path, 'documentation_old'):
                moved_count += 1
    
    # Move old results directory (we now use output/)
    logger.info("\n=== Moving old results directory ===")
    results_dir = BASE_DIR / 'results'
    if results_dir.exists() and results_dir.is_dir():
        if move_to_old(results_dir, 'old_results'):
            moved_count += 1
    
    # Move datasetCICDDos2019 directory (if exists and is empty/unused)
    logger.info("\n=== Moving unused dataset directories ===")
    unused_dataset_dirs = ['datasetCICDDos2019']
    for dir_name in unused_dataset_dirs:
        dir_path = BASE_DIR / dir_name
        if dir_path.exists() and dir_path.is_dir():
            # Check if directory is empty or contains only small/unused files
            try:
                files = list(dir_path.iterdir())
                if len(files) == 0 or all(f.is_file() and f.stat().st_size < 1024 for f in files):
                    if move_to_old(dir_path, 'old_results'):
                        moved_count += 1
            except Exception as e:
                logger.warning(f"Could not check {dir_name}: {e}")
    
    logger.info(f"\n=== Reorganization complete ===")
    logger.info(f"Total items moved: {moved_count}")
    logger.info(f"\nNew project structure:")
    logger.info("  Core files: main.py, main_pipeline.py, dataset_loader.py, etc.")
    logger.info("  Core directories: data/, output/")
    logger.info("  Archived files: _old/")
    
    # List remaining files in root
    remaining_files = [f for f in BASE_DIR.iterdir() 
                      if f.is_file() and f.name not in CORE_FILES 
                      and not f.name.startswith('.')]
    remaining_dirs = [d for d in BASE_DIR.iterdir() 
                     if d.is_dir() and d.name not in CORE_DIRS 
                     and not d.name.startswith('.')]
    
    if remaining_files or remaining_dirs:
        logger.info(f"\nFiles/directories remaining in root:")
        for item in sorted(remaining_files + remaining_dirs):
            logger.info(f"  - {item.name}")

if __name__ == "__main__":
    main()
