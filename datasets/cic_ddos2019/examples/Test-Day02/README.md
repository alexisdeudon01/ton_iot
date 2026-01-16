# Test Day 02 - Example Structure

This directory contains example CSV files for **Test Day 02** of the CIC-DDoS2019 dataset.

## Expected Content

Test Day 02 should contain CSV files for **12 attack types** plus benign traffic, as documented on the [official UNB CIC-DDoS2019 website](https://www.unb.ca/cic/datasets/ddos-2019.html):

### Attack Types (12 attacks + Benign + PortScan):
According to the official documentation, the **Second Day (Test Day)** executed these attacks:
1. **Benign.csv** - Normal network traffic
2. **NTP.csv** - NTP amplification attack (10:35 - 10:45)
3. **DNS.csv** - DNS amplification attack (10:52 - 11:05)
4. **LDAP.csv** - LDAP amplification attack (11:22 - 11:32)
5. **MSSQL.csv** - MSSQL attack (11:36 - 11:45)
6. **NetBIOS.csv** - NetBIOS attack (11:50 - 12:00)
7. **SNMP.csv** - SNMP attack (12:12 - 12:23)
8. **SSDP.csv** - SSDP attack (12:27 - 12:37)
9. **UDP.csv** - UDP flood attack (12:45 - 13:09)
10. **UDP-Lag.csv** - UDP-Lag attack (13:11 - 13:15)
11. **WebDDoS.csv** - Web-based DDoS attack (13:18 - 13:29)
12. **SYN.csv** - SYN flood attack (13:29 - 13:34)
13. **TFTP.csv** - TFTP attack (13:35 - 17:15)
14. **PortScan.csv** - Port scan attack (new in test day, unknown timing)

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
