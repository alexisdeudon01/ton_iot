import polars as pl
import pytest

def test_cic_label_rule():
    # y=0 if Label == "BENIGN" else 1
    df = pl.DataFrame({
        "Label": ["BENIGN", "DDoS-ACK", "BENIGN", "DDoS-PSH"]
    })
    df = df.with_columns([
        pl.when(pl.col("Label") == "BENIGN").then(0).otherwise(1).alias("y")
    ])
    assert df["y"].to_list() == [0, 1, 0, 1]

def test_ton_label_rule():
    # y=1 if type=="ddos" else 0
    df = pl.DataFrame({
        "type": ["normal", "ddos", "normal", "ddos", "backdoor"]
    })
    # Filter first as per task
    df = df.filter(pl.col("type").is_in(["normal", "ddos"]))
    df = df.with_columns([
        pl.when(pl.col("type") == "ddos").then(1).otherwise(0).alias("y")
    ])
    assert df["y"].to_list() == [0, 1, 0, 1]
