# Liste des Features Communes: TON_IoT ↔ CIC-DDoS2019

Ce document liste les features communes proposées entre TON_IoT et CIC-DDoS2019 basées sur l'analyse réelle des CSV et la documentation CICFlowMeter.

## Méthode d'Extraction

Le système utilise `FeatureAnalyzer` pour:
1. **Analyser les colonnes réelles** des CSV
2. **Détecter les unités** (bytes, seconds, packets, flags, etc.)
3. **Catégoriser les features** (temporal, packet, byte, flag, rate, etc.)
4. **Calculer la similarité** entre features des deux datasets
5. **Proposer des mappings** sémantiques intelligents

## Features Communes Proposées

### 1. Features Exactes (Exact Match)

Ces features ont exactement le même nom dans les deux datasets:

| Unified Name | TON_IoT | CIC-DDoS2019 | Category | Unit |
|--------------|---------|--------------|----------|------|
| *(À déterminer par analyse réelle)* | | | | |

### 2. Features Sémantiques (Semantic Match)

#### A. Features Temporelles (Temporal)

| Unified Name | TON_IoT Variants | CIC-DDoS2019 Variants | Category | Unit | Confidence |
|--------------|------------------|----------------------|----------|------|------------|
| `flow_duration` | duration, flow_duration, time, flow_time, dur | Flow Duration, flow duration, Duration | temporal | microseconds | High |
| `fwd_iat_mean` | fwd_iat, forward_iat, iat_mean, fwd_iat_mean | Fwd IAT Mean, fwd iat mean, Forward IAT Mean | temporal | microseconds | High |
| `bwd_iat_mean` | bwd_iat, backward_iat, bwd_iat_mean | Bwd IAT Mean, bwd iat mean, Backward IAT Mean | temporal | microseconds | High |
| `flow_iat_mean` | flow_iat, iat_mean | Flow IAT Mean, flow iat mean | temporal | microseconds | Medium |
| `active_mean` | active, active_time | Active Mean, active mean | temporal | microseconds | Medium |
| `idle_mean` | idle, idle_time | Idle Mean, idle mean | temporal | microseconds | Medium |

#### B. Features Paquets (Packet)

| Unified Name | TON_IoT Variants | CIC-DDoS2019 Variants | Category | Unit | Confidence |
|--------------|------------------|----------------------|----------|------|------------|
| `fwd_packets` | fwd_packets, forward_packets, src_packets, packets_sent, total_packets | Total Fwd Packets, total fwd packets, Fwd Packets | packet | count | High |
| `bwd_packets` | bwd_packets, backward_packets, dst_packets, packets_received | Total Backward Packets, total backward packets, Bwd Packets | packet | count | High |
| `total_packets` | total_packets, num_packets, packets | Total Packets (Fwd + Bwd) | packet | count | Medium |

#### C. Features Bytes (Byte)

| Unified Name | TON_IoT Variants | CIC-DDoS2019 Variants | Category | Unit | Confidence |
|--------------|------------------|----------------------|----------|------|------------|
| `fwd_bytes` | fwd_bytes, forward_bytes, src_bytes, bytes_sent, fwd_length, total_bytes | Total Length of Fwd Packets, total length of fwd packets, Fwd Bytes | byte | bytes | High |
| `bwd_bytes` | bwd_bytes, backward_bytes, dst_bytes, bytes_received, bwd_length | Total Length of Bwd Packets, total length of bwd packets, Bwd Bytes | byte | bytes | High |
| `fwd_packet_length_mean` | fwd_packet_length, fwd_pkt_len, forward_packet_length, packet_length | Fwd Packet Length Mean, fwd packet length mean | byte | bytes | Medium |
| `bwd_packet_length_mean` | bwd_packet_length, bwd_pkt_len, backward_packet_length | Bwd Packet Length Mean, bwd packet length mean | byte | bytes | Medium |
| `packet_length_mean` | packet_length, avg_packet_length | Packet Length Mean, packet length mean | byte | bytes | Medium |

#### D. Features Taux/Débit (Rate)

| Unified Name | TON_IoT Variants | CIC-DDoS2019 Variants | Category | Unit | Confidence |
|--------------|------------------|----------------------|----------|------|------------|
| `flow_bytes_per_sec` | flow_bytes_per_sec, bytes_per_second, throughput, rate, bps | Flow Bytes/s, flow bytes/s, Flow Bytes per second | rate | bytes_per_second | High |
| `flow_packets_per_sec` | flow_packets_per_sec, packets_per_second, pps | Flow Packets/s, flow packets/s, Flow Packets per second | rate | packets_per_second | High |
| `fwd_packets_per_sec` | fwd_packets_per_sec, fwd_pps | Fwd Packets/s, fwd packets/s | rate | packets_per_second | Medium |
| `bwd_packets_per_sec` | bwd_packets_per_sec, bwd_pps | Bwd Packets/s, bwd packets/s | rate | packets_per_second | Medium |

#### E. Features Flags TCP

| Unified Name | TON_IoT Variants | CIC-DDoS2019 Variants | Category | Unit | Confidence |
|--------------|------------------|----------------------|----------|------|------------|
| `syn_flags` | syn, syn_flag, syn_count, syn_flags | SYN Flag Count, syn flag count, SYN | flag | count | High |
| `ack_flags` | ack, ack_flag, ack_count, ack_flags | ACK Flag Count, ack flag count, ACK | flag | count | High |
| `fin_flags` | fin, fin_flag, fin_count, fin_flags | FIN Flag Count, fin flag count, FIN | flag | count | High |
| `rst_flags` | rst, rst_flag, rst_count, rst_flags | RST Flag Count, rst flag count, RST | flag | count | Medium |
| `psh_flags` | psh, psh_flag, psh_count, psh_flags | PSH Flag Count, psh flag count, PSH | flag | count | Medium |
| `urg_flags` | urg, urg_flag, urg_count, urg_flags | URG Flag Count, urg flag count, URG | flag | count | Medium |

#### F. Features Identification Réseau

| Unified Name | TON_IoT Variants | CIC-DDoS2019 Variants | Category | Unit | Confidence |
|--------------|------------------|----------------------|----------|------|------------|
| `src_ip` | src_ip, source_ip, srcip, source ip | Src IP, Source IP, src ip | ip_address | - | High |
| `dst_ip` | dst_ip, destination_ip, dstip, destination ip | Dst IP, Destination IP, dst ip | ip_address | - | High |
| `src_port` | src_port, source_port, srcport, source port | Src Port, Source Port, src port | port | - | High |
| `dst_port` | dst_port, destination_port, dstport, destination port | Dst Port, Destination Port, dst port | port | - | High |
| `protocol` | proto, protocol, protocol_name | Protocol, protocol | protocol | - | High |

#### G. Features Statistiques Flow

| Unified Name | TON_IoT Variants | CIC-DDoS2019 Variants | Category | Unit | Confidence |
|--------------|------------------|----------------------|----------|------|------------|
| `down_up_ratio` | down_up_ratio, ratio, up_down_ratio | Down/Up Ratio, down/up ratio | flow_statistic | ratio | Medium |
| `average_packet_size` | avg_packet_size, average_packet, packet_size_avg | Average Packet Size, average packet size | flow_statistic | bytes | Medium |
| `min_packet_length` | min_packet_length, min_pkt_len | Min Packet Length, min packet length | flow_statistic | bytes | Medium |
| `max_packet_length` | max_packet_length, max_pkt_len | Max Packet Length, max packet length | flow_statistic | bytes | Medium |

## Catégories de Features

### 1. **Temporal** (Features temporelles)
- Flow Duration, IAT (Inter-Arrival Time), Active Time, Idle Time
- **Unités**: microseconds, milliseconds, seconds

### 2. **Packet** (Features paquets)
- Nombre de paquets forward/backward, total
- **Unités**: count

### 3. **Byte** (Features bytes)
- Taille des paquets, longueur totale, bytes forward/backward
- **Unités**: bytes

### 4. **Rate** (Features taux/débit)
- Bytes/s, Packets/s, throughput
- **Unités**: bytes_per_second, packets_per_second

### 5. **Flag** (Features flags TCP)
- SYN, ACK, FIN, RST, PSH, URG counts
- **Unités**: count

### 6. **IP Address** (Adresses IP)
- Source IP, Destination IP
- **Unités**: - (adresse IP)

### 7. **Port** (Ports)
- Source Port, Destination Port
- **Unités**: - (numéro de port)

### 8. **Protocol** (Protocole)
- Protocol name/number
- **Unités**: - (nom ou numéro)

### 9. **Flow Statistic** (Statistiques de flux)
- Ratios, moyennes, min/max
- **Unités**: ratio, bytes, etc.

## Utilisation

### Analyse Automatique

Pour analyser les features réelles et générer un rapport:

```bash
python3 analyze_features.py
```

Cela va:
1. Charger les datasets (1% pour analyse rapide)
2. Analyser toutes les colonnes
3. Proposer des features communes
4. Générer `output/feature_mapping_report.md`

### Utilisation dans le Pipeline

Le système d'harmonisation utilise automatiquement `FeatureAnalyzer` si disponible:

```python
from src.data_harmonization import DataHarmonizer

harmonizer = DataHarmonizer()
df_cic_harm, df_ton_harm = harmonizer.harmonize_features(df_cic, df_ton)
# Utilise FeatureAnalyzer automatiquement pour trouver les features communes
```

## Notes sur les Unités

### CIC-DDoS2019 (CICFlowMeter)
- **Flow Duration**: microseconds
- **IAT (Inter-Arrival Time)**: microseconds
- **Packets**: count
- **Bytes**: bytes
- **Rates**: bytes/s ou packets/s

### TON_IoT
- **Duration**: peut être en seconds, milliseconds, ou microseconds (à vérifier)
- **Packets**: count
- **Bytes**: bytes
- **Rates**: bytes/s ou packets/s

**Important**: Vérifier les unités lors de l'harmonisation et normaliser si nécessaire.

## Features CICFlowMeter Standard (80 features)

Les features CICFlowMeter standard incluent:
- **Temporal**: Flow Duration, Flow IAT (Mean/Std/Max/Min), Fwd/Bwd IAT, Active/Idle Time
- **Packets**: Total Fwd/Bwd Packets, Fwd/Bwd Packets/s, Subflow Fwd/Bwd Packets
- **Bytes**: Total Length Fwd/Bwd, Fwd/Bwd Packet Length (Mean/Std/Max/Min), Flow Bytes/s
- **Flags**: FIN, SYN, RST, PSH, ACK, URG, CWE, ECE Flag Counts
- **Segments**: Avg Fwd/Bwd Segment Size, Fwd/Bwd Avg Bytes/Bulk, etc.
- **TCP Window**: Init_Win_bytes_forward/backward
- **Et plus...**

Voir `datasets/cic_ddos2019/FEATURES_DESCRIPTION.md` pour la liste complète.

## Références

- **CICFlowMeter**: https://github.com/ahlashkari/CICFlowMeter
- **CIC-DDoS2019**: Sharafaldin et al. (2019)
- **TON_IoT**: Moustafa et al. (2019), UNSW
