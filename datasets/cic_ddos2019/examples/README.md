# Example Files

This directory contains example/template CSV files that demonstrate the expected structure of CIC-DDoS2019 dataset files.

## Important

**These files are NOT loaded by the dataset loader.** They are for reference only.

The actual CIC-DDoS2019 dataset CSV files should be placed in the parent directory (`datasets/cic_ddos2019/`).

## Files

- `example_cicddos2019_structure.csv` - Example CSV showing the structure and format expected for CIC-DDoS2019 dataset files

## Directory Structure

The CIC-DDoS2019 dataset is organized by days:

- **`Training-Day01/`** - Training set with 12 attack types + benign
  - Contains CSV files for all attack types used during training
  - See `Training-Day01/README.md` for details

- **`Test-Day02/`** - Test set with 7 attack types + benign  
  - Contains CSV files for a subset of attacks for evaluation
  - See `Test-Day02/README.md` for details

### Organization Options

You can organize the dataset in two ways:

**Option 1: Flat structure (current default)**
```
datasets/cic_ddos2019/
├── Benign.csv
├── UDP.csv
├── SYN.csv
├── ...
```

**Option 2: By day (recommended for large datasets)**
```
datasets/cic_ddos2019/
├── Training-Day01/
│   ├── Benign.csv
│   ├── UDP.csv
│   ├── SYN.csv
│   └── ...
└── Test-Day02/
    ├── Benign.csv
    ├── UDP.csv
    └── ...
```

The dataset loader automatically detects and loads CSV files from both structures.

## Usage

These files are useful for:
- Understanding the expected CSV format
- Validating that downloaded dataset files have the correct structure
- Testing data loading logic (when explicitly referenced, not via auto-discovery)

## Dataset Loader Behavior

The dataset loader automatically excludes files containing "example", "sample", "template", or "structure" in their filename to prevent accidental loading of synthetic data.
