# Feature Analysis Report (CIC vs TON)

## 1. Common Features (15)
flow_duration, total_fwd_packets, total_bwd_packets, total_fwd_bytes, total_bwd_bytes, total_packets, total_bytes, bytes_per_packet, fwd_packet_ratio, fwd_byte_ratio, bytes_per_second, packets_per_second, avg_packet_size, flow_asymmetry, header_ratio

## 2. Zero-variance Features (std=0)
None

## 3. Features with Significantly Different Distributions (KS Test p < 0.01)
- flow_duration
- total_fwd_packets
- total_bwd_packets
- total_fwd_bytes
- total_bwd_bytes
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