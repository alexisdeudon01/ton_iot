# Example Files

This directory contains example/template CSV files that demonstrate the expected structure of CIC-DDoS2019 dataset files.

## Important

**These files are NOT loaded by the dataset loader.** They are for reference only.

The actual CIC-DDoS2019 dataset CSV files should be placed in the parent directory (`datasets/cic_ddos2019/`).

## Files

- `example_cicddos2019_structure.csv` - Example CSV showing the structure and format expected for CIC-DDoS2019 dataset files

## Directory Structure

**⚠️ IMPORTANT: No Need to Reorganize Files**

The dataset loader is designed to **automatically detect and load CSV files** from any location within `datasets/cic_ddos2019/`, including:
- Root directory (`datasets/cic_ddos2019/*.csv`)
- Subdirectories (`datasets/cic_ddos2019/*/*.csv`)
- Nested subdirectories (`datasets/cic_ddos2019/*/*/*.csv`)

**Why you don't need to reorganize:**

1. **Flexible Loading**: The dataset loader recursively searches all subdirectories, so files can remain in `examples/Training-Day01/` and `examples/Test-Day02/` as they are.

2. **Automatic Filtering**: The loader automatically excludes template/example files (files with "example", "sample", "template", or "structure" in their name) while loading actual data files.

3. **No Manual Organization Required**: Whether your CSV files are in:
   - `examples/Training-Day01/`
   - `examples/Test-Day02/`
   - Root directory
   - Any other subdirectory
   
   The loader will find and load them automatically.

4. **Prevents Data Corruption**: By filtering by filename pattern rather than directory location, we can safely store example template files alongside real data without risk of accidental loading.

**Current Organization (as-is):**
```
datasets/cic_ddos2019/
├── examples/
│   ├── Training-Day01/
│   │   ├── DrDoS_NTP.csv        ← Will be loaded
│   │   ├── UDP.csv              ← Will be loaded
│   │   ├── SYN.csv              ← Will be loaded
│   │   └── ...
│   ├── Test-Day02/
│   │   ├── DrDoS_DNS.csv        ← Will be loaded
│   │   ├── TFTP.csv             ← Will be loaded
│   │   └── ...
│   └── example_cicddos2019_structure.csv  ← Excluded (template file)
```

**Result**: The dataset loader automatically detects all valid CSV files (currently ~20 files) regardless of their location within the directory tree.

## Usage

These files are useful for:
- Understanding the expected CSV format
- Validating that downloaded dataset files have the correct structure
- Testing data loading logic (when explicitly referenced, not via auto-discovery)

## Dataset Loader Behavior

The dataset loader automatically excludes files containing "example", "sample", "template", or "structure" in their filename to prevent accidental loading of synthetic data.
