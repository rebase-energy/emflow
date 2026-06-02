"""Build the Swedish temperature station benchmark dataset.

Downloads hourly air-temperature observations from the SMHI Open Data
Meteorological Observations (metobs) REST API for ~15 major Swedish cities and
writes a wide-format parquet file suitable for autoregressive forecasting.

Source : https://opendata.smhi.se/  (parameter 1 = air temperature, hourly, °C)
Period : ``corrected-archive`` (quality-controlled historical data)
Output : ``emflow/examples/data/swedish-temperatures.parquet`` (wide MultiIndex)
         ``emflow/examples/data/swedish-temperatures-stations.parquet`` (metadata)

Run:
    python scripts/build_swedish_temperatures.py

Dependencies: pandas, pyarrow (installed via the ``dev`` extra), and stdlib.
"""

from __future__ import annotations

import io
import json
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

# --- Configuration ----------------------------------------------------------

API = "https://opendata-download-metobs.smhi.se/api/version/1.0"
PARAMETER = 1  # Air temperature, instantaneous, once per hour (°C)
PERIOD = "corrected-archive"  # Quality-controlled history; CSV only
YEARS = 10  # Length of the window to keep, counting back from the archive end

# Curated set of ~15 geographically diverse major cities. For each city we keep
# a list of name fragments to match against the live station list; the resolver
# then picks the best active station covering the target window.
CITIES = [
    "Stockholm",
    "Göteborg",
    "Malmö",
    "Uppsala",
    "Norrköping",
    "Linköping",
    "Örebro",
    "Karlstad",
    "Sundsvall",
    "Östersund",
    "Umeå",
    "Luleå",
    "Kiruna",
    "Visby",
    "Jönköping",
]

# Stations whose actual corrected-archive coverage falls below this fraction
# within the window are flagged (the listing's from/to can't be trusted, since
# station relocations split a city's history across several station ids).
MAX_MISSING_PCT = 15.0

OUT_DIR = Path(__file__).resolve().parent.parent / "emflow" / "examples" / "data"
DATA_PATH = OUT_DIR / "swedish-temperatures.parquet"
META_PATH = OUT_DIR / "swedish-temperatures-stations.parquet"

USER_AGENT = "emflow-dataset-builder/1.0 (+https://github.com/rebase-energy/emflow)"


# --- HTTP helpers ------------------------------------------------------------

def _fetch(url: str, retries: int = 3, backoff: float = 2.0) -> bytes:
    """GET a URL, returning raw bytes, with a few polite retries."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as err:
            last_err = err
            wait = backoff * (attempt + 1)
            print(f"  ! fetch failed ({err}); retrying in {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def _normalize(text: str) -> str:
    """Lower-case and strip diacritics for tolerant name matching."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


# --- Station resolution ------------------------------------------------------

def resolve_stations() -> list[dict]:
    """Pick the best active station per city from the live parameter listing.

    For each city we prefer an active station whose data ends most recently;
    ties favour automatic stations (names ending in " A"/"Aut") which deliver
    continuous hourly data, then those with the longest history.
    """
    print(f"Fetching station list for parameter {PARAMETER} ...")
    payload = json.loads(_fetch(f"{API}/parameter/{PARAMETER}.json"))
    stations = pd.json_normalize(payload["station"])

    chosen: list[dict] = []
    for city in CITIES:
        key = _normalize(city)
        cand = stations[stations["name"].map(_normalize).str.contains(key, na=False)]
        cand = cand[cand["active"]]
        if cand.empty:
            print(f"  ! no active station found for {city!r}; skipping")
            continue

        def score(row: pd.Series) -> tuple:
            name = str(row["name"])
            automatic = name.endswith(" A") or "Aut" in name
            return (row["to"], automatic, -row["from"])

        best = max((r for _, r in cand.iterrows()), key=score)
        chosen.append(
            {
                "station_id": str(int(best["id"])),
                "station_name": str(best["name"]),
                "latitude": float(best["latitude"]),
                "longitude": float(best["longitude"]),
                "elevation": float(best.get("height", float("nan"))),
                "city": city,
                "from": int(best["from"]),
                "to": int(best["to"]),
            }
        )
        print(f"  {city:<12} -> {best['name']} (id {int(best['id'])})")
    return chosen


# --- Data download + parse ---------------------------------------------------

def fetch_station_series(station_id: str) -> pd.Series:
    """Download and parse one station's corrected-archive hourly temperature."""
    url = f"{API}/parameter/{PARAMETER}/station/{station_id}/period/{PERIOD}/data.csv"
    text = _fetch(url).decode("utf-8")

    # The CSV has several ';'-separated metadata blocks before the data block.
    # The data block is introduced by a header line starting with "Datum".
    lines = text.splitlines()
    header_idx = next(
        i for i, ln in enumerate(lines) if ln.startswith("Datum;Tid (UTC);Lufttemperatur")
    )
    block = "\n".join(lines[header_idx:])
    df = pd.read_csv(
        io.StringIO(block),
        sep=";",
        usecols=[0, 1, 2, 3],
        names=["date", "time", "temperature", "quality"],
        header=0,
        dtype={"date": str, "time": str, "quality": str},
    )

    df = df[df["quality"].isin(["G", "Y"])]  # keep approved + coarse-but-usable
    ts = pd.to_datetime(df["date"] + " " + df["time"], utc=True, errors="coerce")
    temp = pd.to_numeric(df["temperature"], errors="coerce")
    series = pd.Series(temp.values, index=ts, name=station_id)
    series = series[series.index.notna()].dropna()
    # Collapse any rare duplicate timestamps (keep last reported value).
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series


# --- Assembly ----------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stations = resolve_stations()
    if not stations:
        raise SystemExit("No stations resolved; aborting.")

    # The archive end is the most recent observation across chosen stations
    # (the parameter listing's ``to`` field, epoch-ms UTC). Snap to the hour and
    # keep the last YEARS of data.
    window_end = pd.to_datetime(max(st["to"] for st in stations), unit="ms", utc=True).floor("h")
    window_start = window_end - pd.DateOffset(years=YEARS)
    print(f"\nWindow: {window_start} -> {window_end}")

    print(f"\nDownloading {len(stations)} stations ...")
    series_map: dict[tuple[str, str], pd.Series] = {}
    for st in stations:
        sid, name = st["station_id"], st["station_name"]
        print(f"  downloading {name} (id {sid}) ...")
        s = fetch_station_series(sid)
        s = s[(s.index >= window_start) & (s.index <= window_end)]
        series_map[(sid, name)] = s
        time.sleep(0.5)  # be polite to the API

    # Regular hourly grid over the whole window so gaps become explicit NaN.
    grid = pd.date_range(window_start, window_end, freq="h", tz="UTC", name="timestamp")
    df = pd.DataFrame(index=grid)
    for (sid, name), s in series_map.items():
        df[(sid, name)] = s.reindex(grid)

    df.columns = pd.MultiIndex.from_tuples(
        df.columns, names=["station_id", "station_name"]
    )
    df = df.sort_index(axis=1).round(1)

    # --- Station metadata sidecar -------------------------------------------
    meta_rows = []
    for st in stations:
        col = df[(st["station_id"], st["station_name"])]
        meta_rows.append(
            {
                "station_id": st["station_id"],
                "station_name": st["station_name"],
                "city": st["city"],
                "latitude": st["latitude"],
                "longitude": st["longitude"],
                "elevation": st["elevation"],
                "first_obs": col.first_valid_index(),
                "last_obs": col.last_valid_index(),
                "n_obs": int(col.notna().sum()),
                "pct_missing": round(100 * col.isna().mean(), 2),
            }
        )
    meta_df = pd.DataFrame(meta_rows).set_index("station_id")

    # --- Write --------------------------------------------------------------
    df.to_parquet(DATA_PATH, engine="pyarrow", compression="snappy")
    meta_df.to_parquet(META_PATH, engine="pyarrow", compression="snappy")

    # --- Summary ------------------------------------------------------------
    print("\n=== Summary ===")
    print(f"Data   : {DATA_PATH}  ({DATA_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"Meta   : {META_PATH}  ({META_PATH.stat().st_size / 1e3:.1f} KB)")
    print(f"Shape  : {df.shape[0]:,} hours x {df.shape[1]} stations")
    print(f"Range  : {df.index.min()} -> {df.index.max()}")
    print("\nMissing per station:")
    print(meta_df[["station_name", "pct_missing", "n_obs"]].to_string())

    poor = meta_df[meta_df["pct_missing"] > MAX_MISSING_PCT]
    if not poor.empty:
        print(
            f"\n! WARNING: {len(poor)} station(s) exceed {MAX_MISSING_PCT}% missing "
            f"(likely incomplete corrected-archive coverage): "
            + ", ".join(poor["station_name"])
        )


if __name__ == "__main__":
    main()
