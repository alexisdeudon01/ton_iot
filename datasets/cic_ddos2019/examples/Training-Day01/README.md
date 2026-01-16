# Training Day 01 - Example Structure

This directory contains example CSV files for **Training Day 01** of the CIC-DDoS2019 dataset.

## Expected Content

Training Day 01 should contain CSV files for **12 attack types** plus benign traffic:

### Attack Types (12 types):
1. **Benign.csv** - Normal network traffic
2. **PortMap.csv** - Port map attack
3. **NetBIOS.csv** - NetBIOS attack
4. **LDAP.csv** - LDAP amplification attack
5. **MSSQL.csv** - MSSQL attack
6. **UDP.csv** - UDP flood attack
7. **UDP-Lag.csv** - UDP-Lag attack
8. **SYN.csv** - SYN flood attack
9. **DNS.csv** - DNS amplification attack
10. **TFTP.csv** - TFTP attack
11. **NTP.csv** - NTP amplification attack
12. **SSDP.csv** - SSDP attack
13. **WebDDoS.csv** - Web-based DDoS attack

## Structure Example

```
Training-Day01/
├── Benign.csv
├── PortMap.csv
├── NetBIOS.csv
├── LDAP.csv
├── MSSQL.csv
├── UDP.csv
├── UDP-Lag.csv
├── SYN.csv
├── DNS.csv
├── TFTP.csv
├── NTP.csv
├── SSDP.csv
└── WebDDoS.csv
```

## Usage

When you download the actual CIC-DDoS2019 dataset, place all CSV files from Training Day 01 in this directory (or in the main `datasets/cic_ddos2019/` directory).

The dataset loader will automatically:
- Detect all CSV files in the directory
- Load and concatenate them
- Exclude example/template files (files with "example", "sample", "template" in name)
