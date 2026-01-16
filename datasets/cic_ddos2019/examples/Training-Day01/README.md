# Training Day 01 - Example Structure

This directory contains example CSV files for **Training Day 01** of the CIC-DDoS2019 dataset.

## Expected Content

Training Day 01 should contain CSV files for **7 attack types** plus benign traffic, as documented on the [official UNB CIC-DDoS2019 website](https://www.unb.ca/cic/datasets/ddos-2019.html):

### Attack Types (7 attacks + Benign):
According to the official documentation, the **First Day (Training Day)** executed these attacks:
1. **Benign.csv** - Normal network traffic
2. **PortMap.csv** - Port map attack (9:43 - 9:51)
3. **NetBIOS.csv** - NetBIOS attack (10:00 - 10:09)
4. **LDAP.csv** - LDAP amplification attack (10:21 - 10:30)
5. **MSSQL.csv** - MSSQL attack (10:33 - 10:42)
6. **UDP.csv** - UDP flood attack (10:53 - 11:03)
7. **UDP-Lag.csv** - UDP-Lag attack (11:14 - 11:24)
8. **SYN.csv** - SYN flood attack (11:28 - 17:35)

**Note**: Some datasets may also include files from the second day (NTP, DNS, TFTP, SSDP, SNMP, WebDDoS) but these are from the **Test Day** according to UNB documentation.

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
