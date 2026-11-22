import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# Топ-15 тикеров MOEX по ликвидности (IMOEX constituents + проверенные на TQBR)
TICKERS = [
    ("SBER", "Сбербанк"),
    ("GAZP", "Газпром"),
    ("LKOH", "Лукойл"),
    ("GMKN", "Норникель"),
    ("YDEX", "Яндекс"),
    ("ROSN", "Роснефть"),
    ("TATN", "Татнефть"),
    ("MGNT", "Магнит"),
    ("MTSS", "МТС"),
    ("NVTK", "Новатэк"),
    ("PLZL", "Полюс"),
    ("CHMF", "Северсталь"),
    ("NLMK", "НЛМК"),
    ("AFKS", "Система"),
    ("VTBR", "ВТБ"),
]

DATE_FROM = "2023-01-01"
DATE_TILL = "2025-11-13"  # ~33 месяца, ~750 торговых дней на тикер
BOARD = "TQBR"

COLS_KEEP = [
    "TRADEDATE",
    "SECID",
    "SHORTNAME",
    "OPEN",
    "HIGH",
    "LOW",
    "CLOSE",
    "VOLUME",
    "VALUE",
    "NUMTRADES",
]


def fetch_ticker(secid: str) -> pd.DataFrame:
    """Пагинация ISS history endpoint."""
    rows = []
    start = 0
    while True:
        url = (
            f"https://iss.moex.com/iss/history/engines/stock/markets/shares/"
            f"boards/{BOARD}/securities/{secid}.json"
            f"?from={DATE_FROM}&till={DATE_TILL}&start={start}"
        )
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                payload = json.load(r)
        except Exception as e:
            print(f"[ERROR] {secid} start={start}: {e}")
            break
        block = payload.get("history", {})
        cols = block.get("columns", [])
        data = block.get("data", [])
        if not data:
            break
        rows.extend(data)
        if len(data) < 100:
            break
        start += 100
        time.sleep(0.4)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=cols)
    df = df[[c for c in COLS_KEEP if c in df.columns]]
    return df


def main() -> None:
    frames = []
    for secid, name in TICKERS:
        print(f"[fetch] {secid} ({name}) ...", end=" ", flush=True)
        df = fetch_ticker(secid)
        if df.empty:
            print("EMPTY")
            continue
        df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
        for c in ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "VALUE", "NUMTRADES"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df.to_csv(RAW / f"ohlcv_{secid}.csv", index=False)
        print(f"{len(df)} строк, {df['TRADEDATE'].min().date()}..{df['TRADEDATE'].max().date()}")
        frames.append(df)
        time.sleep(0.5)

    if not frames:
        raise SystemExit("Не удалось скачать ни одного тикера.")
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(RAW / "ohlcv_all.csv", index=False)
    print(f"\nCombined: {len(all_df)} строк, {all_df['SECID'].nunique()} тикеров")
    print(f"Saved to {RAW}/ohlcv_all.csv")


if __name__ == "__main__":
    main()
