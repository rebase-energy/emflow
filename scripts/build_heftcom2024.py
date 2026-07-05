"""Build the HEFTCom2024 benchmark dataset and upload it to Hugging Face.

Downloads the openly archived competition data (Zenodo record 13950764 — the
official static copy of the IEEE DataPort competition data), cleans it into
canonical parquet, and uploads two Rebase-compatible ``rb://`` dataset repos:

* ``<owner>/heftcom2024``          (public)  — NWP features for the full span,
  energy actuals + prices strictly *before* the competition evaluation start,
  the official final leaderboard, and the ``rebase.yaml`` manifest.
* ``<owner>/heftcom2024-private``  (private) — energy actuals + prices for the
  competition evaluation period (2024-02-20 → 2024-05-20). Only the verifier's
  token can read these; agents benchmark against ``validation`` and physically
  cannot score themselves on the holdout.

Canonical schema
----------------
``hornsea_nwp`` / ``pes10_nwp``: (issue_time, valid_time)-indexed parquet of
spatially averaged DWD ICON-EU runs (grid mean over the Hornsea-1 box / the
PES-10 points), horizons ≤ 54 h.
``energy`` / ``prices``: half-hourly UTC actuals. Generation is in MWh credits
per settlement period as scored by the competition
(``wind = 0.5*Wind_MW - boa_MWh``, ``solar = 0.5*Solar_MW``).

Usage:
    python scripts/build_heftcom2024.py                 # download + build only
    python scripts/build_heftcom2024.py --upload        # ... and push to HF
    # HF token: --token, else HUGGINGFACE_TOKEN / HF_TOKEN env var
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ZENODO = "https://zenodo.org/records/13950764/files/{}?download=1"
RAW_FILES = [
    "dwd_icon_eu_hornsea_1_20200920_20231027.nc",
    "dwd_icon_eu_hornsea_1_20231027_20240108.nc",
    "dwd_icon_eu_hornsea_1_20240129_20240519.nc",
    "dwd_icon_eu_pes10_20200920_20231027.nc",
    "dwd_icon_eu_pes10_20231027_20240108.nc",
    "dwd_icon_eu_pes10_20240129_20240519.nc",
    "Energy_Data_20200920_20240118.csv",
    "Energy_Data_20240119_20240519.csv",
    "overall_leaderboard.csv",
]

CACHE = Path.home() / ".cache" / "emflow" / "heftcom2024"

#: Start of the official competition evaluation period — the public/private cut.
#: First scored settlement period: delivery day 2024-02-20 in the CET market
#: calendar, i.e. 23:00 UTC on 2024-02-19 (verified against the trades archive).
COMPETITION_START = pd.Timestamp("2024-02-19 23:00", tz="UTC")
DATA_END = pd.Timestamp("2024-05-20 00:00", tz="UTC")
MAX_HORIZON_H = 54

WIND_VARS = {"WindSpeed": "wind_speed", "WindSpeed:100": "wind_speed_100",
             "WindDirection:100": "wind_direction_100", "Temperature": "temperature",
             "RelativeHumidity": "relative_humidity"}
SOLAR_VARS = {"SolarDownwardRadiation": "solar_down_radiation",
              "CloudCover": "cloud_cover", "Temperature": "temperature"}


def download(raw_dir: Path) -> None:
    import urllib.request

    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES:
        dest = raw_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(ZENODO.format(name.replace(" ", "%20")), dest)


def load_nwp(path: Path, var_map: dict, spatial_dims) -> pd.DataFrame:
    """One netCDF run file -> tidy (issue_time, valid_time) frame.

    Handles both archive namings: historic files use ``ref_datetime`` +
    ``valid_datetime`` (hours offset); 2024 files use ``reference_time`` +
    ``valid_time``.
    """
    import xarray as xr

    ds = xr.open_dataset(path)
    ref_dim = "ref_datetime" if "ref_datetime" in ds.dims else "reference_time"
    valid_dim = "valid_datetime" if "valid_datetime" in ds.dims else "valid_time"

    ds = ds[list(var_map)].mean(dim=spatial_dims)
    df = ds.to_dataframe().reset_index().rename(columns={
        ref_dim: "issue_time", valid_dim: "horizon_h", **var_map})

    df["issue_time"] = pd.to_datetime(df["issue_time"], utc=True)
    df = df[df["horizon_h"] <= MAX_HORIZON_H]
    df["valid_time"] = df["issue_time"] + pd.to_timedelta(df["horizon_h"], unit="h")
    return df.drop(columns="horizon_h").set_index(["issue_time", "valid_time"]).sort_index()


def build_nwp(raw_dir: Path, prefix: str, var_map: dict, spatial_dims) -> pd.DataFrame:
    parts = [load_nwp(p, var_map, spatial_dims)
             for p in sorted(raw_dir.glob(f"dwd_icon_eu_{prefix}_*.nc"))]
    df = pd.concat(parts)
    return df[~df.index.duplicated(keep="last")].sort_index()


def build_energy(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [pd.read_csv(raw_dir / f) for f in
              ("Energy_Data_20200920_20240118.csv", "Energy_Data_20240119_20240519.csv")]
    df = pd.concat(frames)
    df["dtm"] = pd.to_datetime(df["dtm"], utc=True)
    df = df.drop_duplicates("dtm").set_index("dtm").sort_index()
    df.index.name = "timestamp"

    energy = pd.DataFrame({
        "wind_mwh_credit": 0.5 * df["Wind_MW"] - df["boa_MWh"],
        "solar_mwh_credit": 0.5 * df["Solar_MW"],
        "solar_capacity_mwp": df["Solar_capacity_mwp"],
        "solar_installed_capacity_mwp": df["Solar_installedcapacity_mwp"],
    })
    energy["total_generation_mwh"] = energy["wind_mwh_credit"] + energy["solar_mwh_credit"]

    prices = df[["DA_Price", "SS_Price", "MIP"]].rename(columns={
        "DA_Price": "da_price", "SS_Price": "ss_price", "MIP": "mip"})
    return energy, prices


PUBLIC_MANIFEST = {
    "name": "heftcom2024",
    "description": (
        "Hybrid Energy Forecasting and Trading Competition 2024 (HEFTCom24): "
        "forecast and trade the combined output of the Hornsea-1 offshore wind "
        "farm and East-England (PES-10) solar. Public split: NWP features for "
        "the full span; energy actuals and prices strictly before the "
        "competition evaluation period (2024-02-20). Holdout labels live in "
        "the companion -private repo."
    ),
    "citation": "Browell et al., 'The Hybrid Renewable Energy Forecasting and "
                "Trading Competition 2024', arXiv:2507.01579. Data: "
                "doi.org/10.5281/zenodo.13950764 (CC-BY-4.0).",
    "fields": {
        "hornsea_nwp": {"path": "data/hornsea_nwp.parquet", "kind": "forecast",
                        "availability_lag": "4h",
                        "description": "DWD ICON-EU grid-mean over Hornsea-1, horizons ≤ 54h"},
        "pes10_nwp": {"path": "data/pes10_nwp.parquet", "kind": "forecast",
                      "availability_lag": "4h",
                      "description": "DWD ICON-EU point-mean over PES-10 solar sites"},
        "energy": {"path": "data/energy.parquet", "kind": "actual",
                   "availability_lag": "1D",
                   "description": "Half-hourly MWh credits (wind/solar/total) + solar capacity"},
        "prices": {"path": "data/prices.parquet", "kind": "actual",
                   "availability_lag": "1D",
                   "description": "Half-hourly day-ahead price, single system price, MIP"},
    },
}

PRIVATE_MANIFEST = {
    "name": "heftcom2024-private",
    "description": "HEFTCom24 holdout labels: energy actuals + prices for the "
                   "official evaluation period (2024-02-20 → 2024-05-20). "
                   "Verifier-only — do not grant agents read access.",
    "fields": {
        "energy": {"path": "data/energy.parquet", "kind": "actual",
                   "availability_lag": "1D"},
        "prices": {"path": "data/prices.parquet", "kind": "actual",
                   "availability_lag": "1D"},
    },
}


def build(raw_dir: Path, out_dir: Path) -> None:
    pub, priv = out_dir / "public", out_dir / "private"
    (pub / "data").mkdir(parents=True, exist_ok=True)
    (priv / "data").mkdir(parents=True, exist_ok=True)

    print("building NWP parquet ...")
    build_nwp(raw_dir, "hornsea_1", WIND_VARS,
              ["latitude", "longitude"]).to_parquet(pub / "data" / "hornsea_nwp.parquet")
    build_nwp(raw_dir, "pes10", SOLAR_VARS,
              ["point"]).to_parquet(pub / "data" / "pes10_nwp.parquet")

    print("building energy/prices parquet ...")
    energy, prices = build_energy(raw_dir)
    energy.loc[:COMPETITION_START - pd.Timedelta("30min")].to_parquet(pub / "data" / "energy.parquet")
    prices.loc[:COMPETITION_START - pd.Timedelta("30min")].to_parquet(pub / "data" / "prices.parquet")
    energy.loc[COMPETITION_START:DATA_END].to_parquet(priv / "data" / "energy.parquet")
    prices.loc[COMPETITION_START:DATA_END].to_parquet(priv / "data" / "prices.parquet")

    (pub / "rebase.yaml").write_text(yaml.safe_dump(PUBLIC_MANIFEST, sort_keys=False))
    (priv / "rebase.yaml").write_text(yaml.safe_dump(PRIVATE_MANIFEST, sort_keys=False))

    # Official final leaderboard: into the public repo AND the benchmark package
    # (problem.py embeds reference scores from the packaged copy — no network).
    lb = pd.read_csv(raw_dir / "overall_leaderboard.csv")
    lb.to_csv(pub / "overall_leaderboard.csv", index=False)
    pkg_copy = Path(__file__).resolve().parents[1] / "emflow" / "benchmarks" / "heftcom2024" / "leaderboard.csv"
    pkg_copy.parent.mkdir(parents=True, exist_ok=True)
    lb.to_csv(pkg_copy, index=False)

    for repo in (pub, priv):
        for f in sorted(repo.rglob("*")):
            if f.is_file():
                print(f"  {f.relative_to(out_dir)}  {f.stat().st_size/1e6:8.2f} MB")


def upload(out_dir: Path, owner: str, token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    for sub, private in (("public", False), ("private", True)):
        repo_id = f"{owner}/heftcom2024{'-private' if private else ''}"
        print(f"uploading {out_dir / sub} -> {repo_id} (private={private}) ...")
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        api.upload_folder(folder_path=str(out_dir / sub), repo_id=repo_id,
                          repo_type="dataset",
                          commit_message="Build HEFTCom2024 benchmark dataset")
        print(f"  -> rb://dataset/{repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=CACHE / "raw")
    ap.add_argument("--out-dir", type=Path, default=CACHE / "build")
    ap.add_argument("--owner", default="rebase-energy")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    download(args.raw_dir)
    build(args.raw_dir, args.out_dir)

    if args.upload:
        token = args.token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("--upload needs an HF token (--token or HUGGINGFACE_TOKEN)")
        upload(args.out_dir, args.owner, token)


if __name__ == "__main__":
    main()
