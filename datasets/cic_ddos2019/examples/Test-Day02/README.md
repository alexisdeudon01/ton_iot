# Test Day 02 - Example Structure

This directory contains example CSV files for **Test Day 02** of the CIC-DDoS2019 dataset.

## Expected Content

Test Day 02 should contain CSV files for **7 attack types** plus benign traffic (subset of Training Day attacks):

### Attack Types (7 types):
1. **Benign.csv** - Normal network traffic
2. **NetBIOS.csv** - NetBIOS attack
3. **LDAP.csv** - LDAP amplification attack
4. **MSSQL.csv** - MSSQL attack
5. **UDP.csv** - UDP flood attack
6. **UDP-Lag.csv** - UDP-Lag attack
7. **SYN.csv** - SYN flood attack
8. **PortScan.csv** - Port scan attack (new in test day)

## Structure Example

```
Test-Day02/
├── Benign.csv
├── NetBIOS.csv
├── LDAP.csv
├── MSSQL.csv
├── UDP.csv
├── UDP-Lag.csv
├── SYN.csv
└── PortScan.csv
```

## Usage

When you download the actual CIC-DDoS2019 dataset, place all CSV files from Test Day 02 in this directory (or in the main `datasets/cic_ddos2019/` directory).

**Note**: The test day contains a subset of attacks to evaluate model generalization on unseen attack types or variations.

## Splits Strategy

- **Training Day 01**: Use for model training and validation
- **Test Day 02**: Use for final model evaluation and generalization testing

The dataset loader can load from either directory structure or a combined directory.
