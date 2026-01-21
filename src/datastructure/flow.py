from typing import List, Optional
import pandas as pd
from .base import IRPBaseStructure

class NetworkFlow(IRPBaseStructure):
    """
    Represents a network flow consisting of multiple packets.
    Identified by Source IP, Destination IP, and optionally Ports/Protocol.
    """
    def __init__(self, flow_id: str, source_ip: str, dest_ip: str):
        super().__init__()
        self.flow_id = flow_id
        self.source_ip = source_ip
        self.dest_ip = dest_ip
        self.packets: List[pd.Series] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def add_packet(self, packet: pd.Series, timestamp_col: str = 'ts'):
        """Adds a packet to the flow and updates time bounds."""
        self.packets.append(packet)
        ts = float(packet[timestamp_col])
        if self.start_time is None or ts < self.start_time:
            self.start_time = ts
        if self.end_time is None or ts > self.end_time:
            self.end_time = ts

    def get_direction(self, packet: pd.Series) -> str:
        """Determines if a packet is forward (src->dst) or backward (dst->src)."""
        if packet['src_ip'] == self.source_ip:
            return "FORWARD"
        return "BACKWARD"

    def to_dataframe(self) -> pd.DataFrame:
        """Converts the flow packets to a DataFrame."""
        return pd.DataFrame(self.packets)

    def __repr__(self):
        return f"Flow({self.flow_id}, Packets: {len(self.packets)}, Duration: {self.end_time - self.start_time if self.start_time else 0:.4f}s)"
