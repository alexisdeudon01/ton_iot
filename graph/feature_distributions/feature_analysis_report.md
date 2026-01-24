# Rapport d'Analyse des Caractéristiques (CIC vs TON)

## 1. Caractéristiques Communes (15)
flow_duration, total_fwd_packets, total_bwd_packets, total_fwd_bytes, total_bwd_bytes, total_packets, total_bytes, bytes_per_packet, fwd_packet_ratio, fwd_byte_ratio, bytes_per_second, packets_per_second, avg_packet_size, flow_asymmetry, header_ratio

## 2. Features à Variance Nulle (std=0)
Aucune

## 3. Features avec Distributions Significativement Différentes (KS Test p < 0.01)
- flow_duration
- total_fwd_packets
- total_bwd_packets
- total_fwd_bytes
- total_packets
- total_bytes
- bytes_per_packet
- fwd_packet_ratio
- fwd_byte_ratio
- bytes_per_second
- packets_per_second
- avg_packet_size
- flow_asymmetry
- header_ratio