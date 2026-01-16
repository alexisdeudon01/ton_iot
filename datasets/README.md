# Datasets Directory

This directory contains the datasets used by the IRP Research Pipeline.

## Structure

```
datasets/
├── ton_iot/
│   ├── train_test_network.csv          # Main TON_IoT dataset
│   └── windows10_dataset.csv            # Alternative/processed TON_IoT dataset
│
└── cic_ddos2019/
    ├── *.csv                            # CIC-DDoS2019 CSV files (one per attack type)
    └── ...
```

## TON_IoT Dataset

**Location**: `datasets/ton_iot/`

**Files**:
- `train_test_network.csv` - Main TON_IoT dataset (required)
- `windows10_dataset.csv` - Alternative processed dataset (optional)

**Source**: [UNSW TON_IoT Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

The pipeline will automatically detect and use `train_test_network.csv` if present.

## CIC-DDoS2019 Dataset

**Location**: `datasets/cic_ddos2019/`

**Files**: Multiple CSV files (one per attack type)

**Download**: [CIC-DDoS2019 Dataset](https://www.unb.ca/cic/datasets/ddos-2019.html)

**Instructions**:
1. Download the CIC-DDoS2019 dataset from the official website
2. Extract all CSV files into `datasets/cic_ddos2019/`
3. The pipeline will automatically load and combine all CSV files

**Note**: The dataset is optional. If not available, the pipeline will use TON_IoT only.

## Legacy Locations

The pipeline also checks these legacy locations for backward compatibility:
- `train_test_network.csv` (project root)
- `data/raw/TON_IoT/`
- `data/raw/CIC-DDoS2019/`
- `Processed_datasets/Processed_Windows_dataset/windows10_dataset.csv`

However, it is recommended to use the new `datasets/` structure.
