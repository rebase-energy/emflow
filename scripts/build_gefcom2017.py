"""Build the GEFCom2017 benchmark dataset and upload to Hugging Face.

Source: github.com/camroach87/gefcom2017data (ISO New England public data,
packaged for the competition) — hourly demand, dry-bulb and dew-point
temperature for the 8 ISO-NE zones plus the MASS and TOTAL aggregates,
2003-03 → 2017-04.

Public/private split at the holdout start (2017-03-01): the qualifying match's
rounds 4-6 (targets March + April 2017) are the holdout; rounds 1-3
(January + February) are validation.

Usage:
    python scripts/build_gefcom2017.py [--upload] [--owner rebase-energy]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import yaml

CACHE = Path.home() / ".cache" / "emflow" / "gefcom2017"
RDA_URL = "https://github.com/camroach87/gefcom2017data/raw/master/data/gefcom.rda"

HOLDOUT_START = pd.Timestamp("2017-03-01")

MANIFEST = {
    "name": "gefcom2017",
    "description": "GEFCom2017 qualifying match — hierarchical probabilistic "
                   "load forecasting for ISO New England (8 zones + MASS + "
                   "TOTAL), 9 quantiles, pinball loss. Public split: history "
                   "before 2017-03-01; rounds 4-6 actuals are private holdout.",
    "citation": "Hong, Xie & Black, 'Global energy forecasting competition "
                "2017: Hierarchical probabilistic load forecasting', IJF 35(4), "
                "2019. Data: github.com/camroach87/gefcom2017data (ISO-NE).",
    "fields": {
        "load": {"path": "data/load.parquet", "kind": "actual",
                 "description": "Hourly demand (MW) per zone incl. MASS/TOTAL aggregates"},
        "temperature": {"path": "data/temperature.parquet", "kind": "actual",
                        "description": "Zonal dry-bulb + dew-point (°F); no future "
                                       "weather — the match was ex-ante"},
    },
}


def download(raw: Path) -> Path:
    import urllib.request

    raw.mkdir(parents=True, exist_ok=True)
    rda = raw / "gefcom.rda"
    if not rda.exists():
        print("downloading gefcom.rda ...")
        urllib.request.urlretrieve(RDA_URL, rda)
    return rda


def build(raw: Path, out_dir: Path) -> None:
    import pyreadr

    df = pyreadr.read_r(download(raw))["gefcom"]
    df["ts"] = pd.to_datetime(df["ts"])

    load = df.pivot(index="ts", columns="zone", values="demand").sort_index()
    drybulb = df.pivot(index="ts", columns="zone", values="drybulb")
    dewpnt = df.pivot(index="ts", columns="zone", values="dewpnt")
    temperature = pd.concat(
        {"drybulb": drybulb, "dewpnt": dewpnt}, axis=1)
    temperature.columns = [f"{zone}_{var}" for var, zone in temperature.columns]
    temperature = temperature.sort_index()

    pub, priv = out_dir / "public", out_dir / "private"
    (pub / "data").mkdir(parents=True, exist_ok=True)
    (priv / "data").mkdir(parents=True, exist_ok=True)

    for name, frame in (("load", load), ("temperature", temperature)):
        frame.loc[:HOLDOUT_START - pd.Timedelta("1h")].to_parquet(pub / "data" / f"{name}.parquet")
        frame.loc[HOLDOUT_START:].to_parquet(priv / "data" / f"{name}.parquet")

    (pub / "rebase.yaml").write_text(yaml.safe_dump(MANIFEST, sort_keys=False))
    (priv / "rebase.yaml").write_text(yaml.safe_dump(
        {**MANIFEST, "name": "gefcom2017-private",
         "description": "GEFCom2017 holdout actuals (2017-03 →) — verifier-only."},
        sort_keys=False))

    for repo in (pub, priv):
        for f in sorted(repo.rglob("*")):
            if f.is_file():
                print(f"  {f.relative_to(out_dir)}  {f.stat().st_size / 1e6:8.2f} MB")


def upload(out_dir: Path, owner: str, token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    for sub, private in (("public", False), ("private", True)):
        repo_id = f"{owner}/gefcom2017{'-private' if private else ''}"
        print(f"uploading {out_dir / sub} -> {repo_id} (private={private}) ...")
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        api.upload_folder(folder_path=str(out_dir / sub), repo_id=repo_id,
                          repo_type="dataset",
                          commit_message="Build GEFCom2017 benchmark dataset")
        print(f"  -> rb://dataset/{repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=CACHE / "build")
    ap.add_argument("--owner", default="rebase-energy")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    build(CACHE / "raw", args.out_dir)

    if args.upload:
        token = args.token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("--upload needs an HF token")
        upload(args.out_dir, args.owner, token)


if __name__ == "__main__":
    main()
