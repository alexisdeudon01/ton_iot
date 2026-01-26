from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


EXCLUDE_COLUMNS = {"Label", "label", "y", "type", "Timestamp", "source_file"}


def _list_csv_files(root_dir: Path) -> List[Path]:
    if not root_dir.exists():
        return []
    files = []
    for path in root_dir.rglob("*.csv"):
        name = path.name.lower()
        if any(token in name for token in ("sample", "template", "example", "structure")):
            continue
        files.append(path)
    return sorted(files)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _standardize_label(df: pd.DataFrame) -> pd.DataFrame:
    label_candidates = ["y", "Label", "label", "type", "attack", "class"]
    label_col = None
    for col in label_candidates:
        if col in df.columns:
            label_col = col
            break
    if label_col is None:
        return df

    if label_col != "y":
        df = df.rename(columns={label_col: "y"})

    if df["y"].dtype == object:
        y_str = df["y"].astype(str).str.lower()
        df["y"] = y_str.str.contains("benign|normal").map({True: 0, False: 1}).astype("int64")
    else:
        df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0).astype("int64")
    return df


def _cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col in EXCLUDE_COLUMNS:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _read_csv(path: Path, nrows: Optional[int]) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        nrows=nrows,
        low_memory=False,
        on_bad_lines="skip",
        na_values=["Infinity", "NaN", "nan", "inf", "None"],
    )
    df = _normalize_columns(df)
    df = _cast_numeric(df)
    df["source_file"] = path.name
    return df


def load_cic_ddos2019(
    cic_path: str | Path,
    max_files: Optional[int] = None,
    max_rows_per_file: Optional[int] = None,
) -> pd.DataFrame:
    cic_dir = Path(cic_path)
    if cic_dir.is_file():
        files = [cic_dir]
    else:
        files = _list_csv_files(cic_dir)

    if not files:
        raise FileNotFoundError(f"No CIC-DDoS2019 CSV files found in {cic_dir}")

    if max_files is not None:
        files = files[:max_files]

    dfs: List[pd.DataFrame] = []
    for file_path in files:
        df = _read_csv(file_path, nrows=max_rows_per_file)
        dfs.append(df)

    df_full = pd.concat(dfs, ignore_index=True, sort=False)
    return _standardize_label(df_full)


def load_ton_iot(
    ton_path: str | Path,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    ton_path = Path(ton_path)
    if ton_path.is_dir():
        files = _list_csv_files(ton_path)
        if not files:
            raise FileNotFoundError(f"No TON_IoT CSV files found in {ton_path}")
        dfs = []
        for file_path in files:
            df = _read_csv(file_path, nrows=max_rows)
            dfs.append(df)
        df_full = pd.concat(dfs, ignore_index=True, sort=False)
        return _standardize_label(df_full)

    if not ton_path.exists():
        raise FileNotFoundError(f"TON_IoT CSV not found at {ton_path}")

    df = _read_csv(ton_path, nrows=max_rows)
    return _standardize_label(df)


def load_datasets(
    cic_path: str | Path,
    ton_path: str | Path,
    max_files: Optional[int] = None,
    max_rows_per_file: Optional[int] = None,
    max_rows_ton: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cic_df = load_cic_ddos2019(cic_path, max_files=max_files, max_rows_per_file=max_rows_per_file)
    ton_df = load_ton_iot(ton_path, max_rows=max_rows_ton)
    return cic_df, ton_df
