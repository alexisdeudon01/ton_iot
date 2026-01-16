# Example Files

This directory contains example/template CSV files that demonstrate the expected structure of CIC-DDoS2019 dataset files.

## Important

**These files are NOT loaded by the dataset loader.** They are for reference only.

The actual CIC-DDoS2019 dataset CSV files should be placed in the parent directory (`datasets/cic_ddos2019/`).

## Files

- `example_cicddos2019_structure.csv` - Example CSV showing the structure and format expected for CIC-DDoS2019 dataset files

## Usage

These files are useful for:
- Understanding the expected CSV format
- Validating that downloaded dataset files have the correct structure
- Testing data loading logic (when explicitly referenced, not via auto-discovery)

## Dataset Loader Behavior

The dataset loader automatically excludes files containing "example", "sample", "template", or "structure" in their filename to prevent accidental loading of synthetic data.
