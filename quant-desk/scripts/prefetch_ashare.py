#!/usr/bin/env python3
"""Pre-download A-share data via curl into quantkit cache.

Python's requests/urllib3 may fail to reach Eastmoney in some network
environments (TLS issue), while curl works fine. This script downloads raw
JSON from the Eastmoney API via curl, converts to OHLCV DataFrame, and
saves as parquet cache files that quantkit's fetch_ohlcv will find.
Set QUANT_DESK_PROXY (e.g. http://127.0.0.1:PORT) to route via an HTTP proxy.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROXY = os.environ.get("QUANT_DESK_PROXY", "").strip()
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# A-share symbols
SYMBOLS = [
    "600519", "000858", "601318", "600036", "000333", "600900",
    "601012", "002415", "600276", "000001", "601888", "300750",
]

# secid mapping: SH=1, SZ=0
def secid(code: str) -> str:
    """Map stock code to Eastmoney secid."""
    if code.startswith("6") or code.startswith("9"):
        return f"1.{code}"  # Shanghai
    return f"0.{code}"  # Shenzhen

def fetch_kline(code: str, start: str = "20200101", end: str = "20260808") -> pd.DataFrame:
    """Fetch daily kline from Eastmoney via curl."""
    sid = secid(code)
    url = (
        f"https://push2his.eastmoney.com/api/qt/stock/kline/get?"
        f"fields1=f1,f2,f3,f4,f5,f6"
        f"&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116"
        f"&ut=7eea3edcaed734bea9cbfc24409ed989"
        f"&klt=101&fqt=1"
        f"&secid={sid}&beg={start}&end={end}"
    )
    
    cmd = ["curl", "-4", "-s", "--max-time", "30"]
    if PROXY:
        cmd += ["-x", PROXY]
    cmd.append(url)
    result = subprocess.run(
        cmd,
        capture_output=True, text=True, timeout=45,
    )
    if result.returncode != 0:
        print(f"  curl failed for {code}: {result.stderr}", file=sys.stderr)
        return pd.DataFrame()
    
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"  JSON parse failed for {code}: {result.stdout[:100]}", file=sys.stderr)
        return pd.DataFrame()
    
    if not data or "data" not in data or data["data"] is None:
        print(f"  No data for {code}", file=sys.stderr)
        return pd.DataFrame()
    
    klines = data["data"].get("klines", [])
    if not klines:
        print(f"  Empty klines for {code}", file=sys.stderr)
        return pd.DataFrame()
    
    # Parse kline format: date,open,close,high,low,volume,amount,...
    rows = []
    for line in klines:
        parts = line.split(",")
        if len(parts) >= 6:
            rows.append({
                "date": parts[0],
                "open": float(parts[1]),
                "close": float(parts[2]),
                "high": float(parts[3]),
                "low": float(parts[4]),
                "volume": float(parts[5]),
            })
    
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


def main():
    cache_dir = DATA_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    success = 0
    failed = []
    
    for code in SYMBOLS:
        cache_key = f"akshare_cn_{code}_1d_2020-01-01_na"
        safe = cache_key.replace("/", "_").replace("\\", "_").replace(" ", "_").replace("^", "")
        cache_path = cache_dir / f"{safe}.parquet"
        
        if cache_path.exists():
            print(f"  {code}: cached ({cache_path.name})")
            success += 1
            continue
        
        print(f"  {code}: downloading via curl …")
        df = fetch_kline(code)
        
        if df.empty:
            print(f"  {code}: FAILED")
            failed.append(code)
            continue
        
        df.to_parquet(cache_path)
        print(f"  {code}: OK ({len(df)} rows → {cache_path.name})")
        success += 1
    
    print(f"\nDone: {success}/{len(SYMBOLS)} cached, {len(failed)} failed")
    if failed:
        print(f"Failed: {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
