# Analyse des Distributions - Features Universelles

## Resume
- Nombre de features : 15
- Echantillons CIC : 5
- Echantillons TON : 70

## Features par type de distribution

### Normales
- Aucune

### Log-normales (necessitent log1p)
- flow_duration
- total_fwd_bytes
- total_bytes
- bytes_per_packet
- bytes_per_second
- packets_per_second
- avg_packet_size

### Heavy-tailed (necessitent RobustScaler)
- Aucune

### Zero-inflated (attention particuliere)
- Aucune

### Autres types
- Bimodale: fwd_byte_ratio
- Discrete: total_fwd_packets, total_bwd_packets, total_bwd_bytes, total_packets, fwd_packet_ratio, flow_asymmetry
- Uniforme: header_ratio

## Recommandations de preprocessing
| Feature | CIC type | TON type | Recommended transform |
| --- | --- | --- | --- |
| flow_duration | Insufficient data | Exponentielle | log1p() puis StandardScaler |
| total_fwd_packets | Insufficient data | Discrete | OneHotEncoder ou garder tel quel |
| total_bwd_packets | Insufficient data | Discrete | OneHotEncoder ou garder tel quel |
| total_fwd_bytes | Insufficient data | Exponentielle | log1p() puis StandardScaler |
| total_bwd_bytes | Insufficient data | Discrete | OneHotEncoder ou garder tel quel |
| total_packets | Insufficient data | Discrete | OneHotEncoder ou garder tel quel |
| total_bytes | Insufficient data | Exponentielle | log1p() puis StandardScaler |
| bytes_per_packet | Insufficient data | Exponentielle | log1p() puis StandardScaler |
| fwd_packet_ratio | Insufficient data | Discrete | OneHotEncoder ou garder tel quel |
| fwd_byte_ratio | Insufficient data | Bimodale | Considerer separation en 2 features ou clustering |
| bytes_per_second | Insufficient data | Exponentielle | log1p() puis StandardScaler |
| packets_per_second | Insufficient data | Exponentielle | log1p() puis StandardScaler |
| avg_packet_size | Insufficient data | Exponentielle | log1p() puis StandardScaler |
| flow_asymmetry | Insufficient data | Discrete | OneHotEncoder ou garder tel quel |
| header_ratio | Insufficient data | Uniforme | MinMaxScaler |

## Observations cles
- Types differents entre CIC et TON : flow_duration, total_fwd_packets, total_bwd_packets, total_fwd_bytes, total_bwd_bytes, total_packets, total_bytes, bytes_per_packet, fwd_packet_ratio, fwd_byte_ratio, bytes_per_second, packets_per_second, avg_packet_size, flow_asymmetry, header_ratio
- Differences de moyenne notables : flow_duration, total_fwd_packets, total_fwd_bytes, total_bwd_bytes, total_packets, total_bytes, bytes_per_packet, bytes_per_second, packets_per_second, avg_packet_size