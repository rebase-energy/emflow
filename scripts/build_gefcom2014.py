"""Build the GEFCom2014 benchmark dataset (all four tracks) and upload to HF.

Source: the official archive from Dr. Tao Hong's release (Dropbox link on
blog.drhongtao.com), containing per-task data for the Load, Price, Wind and
Solar tracks plus the official provisional leaderboard.

The competition's *incremental release* structure maps directly onto emflow's
bitemporal availability:

* each task k reveals the previous period's actuals → target rows carry
  ``knowledge_time = asof_{k+1}`` (the moment they became public);
* each task provides explanatory variables for the target period → predictor
  rows are a forecast field with ``issue_time = asof_k``.

The DataFeed then reproduces, at every origin, exactly the information set
participants had.

Public/private split: everything knowable at task 4's origin is public
(tasks 1–3 were trial tasks — the validation split); later releases are
holdout labels in the private repo (tasks 4–15, the officially scored ones).

Outputs per track: ``<track>_targets.parquet`` (bitemporal actuals),
``<track>_nwp.parquet`` / ``price_load_forecasts.parquet`` (forecast fields),
``load_temperature.parquet`` (bitemporal — the load track deliberately has NO
future temperature: modelling temperature uncertainty was the challenge).
Plus ``tasks.csv`` (the task calendar) and ``leaderboard.csv`` (mean pinball
per team over the scored evaluation weeks), both also packaged into
``emflow/benchmarks/gefcom2014/``.

Usage:
    python scripts/build_gefcom2014.py [--upload] [--owner rebase-energy]
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CACHE = Path.home() / ".cache" / "emflow" / "gefcom2014"
ROOT = CACHE / "raw" / "GEFCom2014 Data"
PKG = Path(__file__).resolve().parents[1] / "emflow" / "benchmarks" / "gefcom2014"

N_TASKS = 15
HOLDOUT_FIRST_TASK = 4  # tasks 1-3 were trial: validation; 4-15 officially scored
QUANTS = [q / 100 for q in range(1, 100)]

#: task-1 target period start per monthly track (hour-ending: first stamp +1h)
FIRST_TARGET = {"load": "2010-10-01", "wind": "2012-10-01", "solar": "2013-04-01"}


def month_task_calendar(track: str) -> pd.DataFrame:
    """Monthly tracks: task k targets the k-th month from FIRST_TARGET."""
    rows = []
    start = pd.Timestamp(FIRST_TARGET[track])
    for k in range(1, N_TASKS + 1):
        t0 = start + pd.DateOffset(months=k - 1)
        t1 = start + pd.DateOffset(months=k)
        idx = pd.date_range(t0 + pd.Timedelta("1h"), t1, freq="1h")
        rows.append({"track": track, "task": k, "asof": t0 + pd.Timedelta("30min"),
                     "target_start": idx[0], "target_end": idx[-1]})
    return pd.DataFrame(rows)


def _knowledge_and_issue(timestamps: pd.Series, tasks: pd.DataFrame,
                         final_release: pd.Timestamp):
    """Map each timestamp to (knowledge_time, issue_time) per the release rules."""
    starts = tasks["target_start"].to_numpy()
    asofs = tasks["asof"].to_numpy()
    idx = np.searchsorted(starts, timestamps.to_numpy(), side="right") - 1
    knowledge = np.where(idx < 0, asofs[0],
                         np.append(asofs[1:], np.datetime64(final_release))[np.clip(idx, 0, None)])
    issue = np.where(idx < 0, asofs[0], asofs[np.clip(idx, 0, None)])
    return pd.Series(knowledge, index=timestamps.index), pd.Series(issue, index=timestamps.index)


# -- solar ----------------------------------------------------------------------------


def parse_ymd(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%Y%m%d %H:%M")


def build_solar(tasks: pd.DataFrame):
    trains, preds = [], []
    for k in range(1, N_TASKS + 1):
        d = ROOT / "Solar" / f"Task {k}"
        trains.append(pd.read_csv(d / f"train{k}.csv"))
        preds.append(pd.read_csv(d / f"predictors{k}.csv"))
    sol_dir = next((ROOT / "Solar").glob("Solution*"))
    solution = pd.read_csv(next(sol_dir.glob("*.csv")))

    power = pd.concat(trains + [solution.rename(columns=str.upper)], ignore_index=True)
    power["TIMESTAMP"] = parse_ymd(power["TIMESTAMP"])
    power = power.drop_duplicates(["ZONEID", "TIMESTAMP"])
    targets = power.pivot(index="TIMESTAMP", columns="ZONEID", values="POWER")
    targets.columns = [f"z{int(z)}" for z in targets.columns]

    nwp = pd.concat(preds, ignore_index=True)
    nwp["TIMESTAMP"] = parse_ymd(nwp["TIMESTAMP"])
    nwp = nwp.drop_duplicates(["ZONEID", "TIMESTAMP"])
    var_cols = [c for c in nwp.columns if c.startswith("VAR")]
    wide = nwp.pivot(index="TIMESTAMP", columns="ZONEID", values=var_cols)
    wide.columns = [f"z{int(z)}_{v.lower()}" for v, z in wide.columns]
    return targets.sort_index(), wide.sort_index()


# -- wind -----------------------------------------------------------------------------


def build_wind(tasks: pd.DataFrame):
    trains, exps = [], []
    for k in range(1, N_TASKS + 1):
        d = ROOT / "Wind" / f"Task {k}"
        for stem, sink in ((f"Task{k}_W_Zone1_10", trains),
                           (f"TaskExpVars{k}_W_Zone1_10", exps)):
            out = CACHE / "extract" / stem
            if not out.exists():
                with zipfile.ZipFile(d / f"{stem}.zip") as z:
                    z.extractall(out)
            for csv in sorted(out.rglob("*.csv")):
                sink.append(pd.read_csv(csv))
    sol_dir = next((ROOT / "Wind").glob("Solution*"), None)
    if sol_dir:
        for csv in sorted(sol_dir.rglob("*.csv")):
            trains.append(pd.read_csv(csv))

    hist = pd.concat(trains, ignore_index=True)
    hist["TIMESTAMP"] = parse_ymd(hist["TIMESTAMP"])
    hist = hist.drop_duplicates(["ZONEID", "TIMESTAMP"])
    targets = hist.pivot(index="TIMESTAMP", columns="ZONEID", values="TARGETVAR")
    targets.columns = [f"z{int(z)}" for z in targets.columns]

    uv_cols = ["U10", "V10", "U100", "V100"]
    future = pd.concat(exps, ignore_index=True)
    future["TIMESTAMP"] = parse_ymd(future["TIMESTAMP"])
    weather = pd.concat([hist[["ZONEID", "TIMESTAMP", *uv_cols]], future], ignore_index=True)
    weather = weather.drop_duplicates(["ZONEID", "TIMESTAMP"])
    wide = weather.pivot(index="TIMESTAMP", columns="ZONEID", values=uv_cols)
    wide.columns = [f"z{int(z)}_{v.lower()}" for v, z in wide.columns]
    return targets.sort_index(), wide.sort_index()


# -- load -----------------------------------------------------------------------------


def build_load(tasks: pd.DataFrame):
    """Task files are hourly-contiguous; timestamps are anchored, not parsed
    (the source's m/d/Y stamps are ambiguous without zero-padding)."""
    frames = []
    cursor = pd.Timestamp("2001-01-01 01:00")
    for k in range(1, N_TASKS + 1):
        df = pd.read_csv(ROOT / "Load" / f"Task {k}" / f"L{k}-train.csv")
        df.index = pd.date_range(cursor, periods=len(df), freq="1h")
        cursor = df.index[-1] + pd.Timedelta("1h")
        frames.append(df.drop(columns=["ZONEID", "TIMESTAMP"]))
        expected_end = tasks.loc[tasks.task == k, "target_start"].iloc[0]
        assert frames[-1].index[-1] == expected_end - pd.Timedelta("1h"), \
            f"load task {k} ends {frames[-1].index[-1]}, expected {expected_end}"
    sol_dir = next((ROOT / "Load").glob("Solution*"))
    sol = pd.read_csv(sol_dir / "solution15_L.csv")
    sol_temp = pd.read_csv(sol_dir / "solution15_L_temperature.csv")
    sol_idx = pd.date_range(cursor, periods=len(sol), freq="1h")
    tail = pd.DataFrame(index=sol_idx)
    tail["LOAD"] = sol.filter(like="LOAD").iloc[:, 0].to_numpy() if sol.filter(like="LOAD").shape[1] \
        else sol.iloc[:, -1].to_numpy()
    for c in [c for c in sol_temp.columns if c.startswith("w")]:
        tail[c] = sol_temp[c].to_numpy()[: len(tail)]
    full = pd.concat(frames + [tail])

    targets = full[["LOAD"]].rename(columns={"LOAD": "load"})
    temperature = full[[c for c in full.columns if c.startswith("w")]]
    return targets, temperature


# -- price ----------------------------------------------------------------------------


def build_price():
    """Price task periods are irregular — derive them from each file's trailing
    NaN block. Returns (targets, load_forecasts, tasks_frame)."""
    files, calendars = {}, []
    for k in range(1, N_TASKS + 1):
        df = pd.read_csv(ROOT / "Price" / f"Task {k}" / f"Task{k}_P.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m%d%Y %H:%M")
        df = df.set_index("timestamp").sort_index()
        files[k] = df
        na = df["Zonal Price"].isna()
        target = df.index[na & (df.index > df.index[na.argmax()] - pd.Timedelta("1h"))]
        # trailing contiguous NaN block:
        rev = na.to_numpy()[::-1]
        n_tail = int(np.argmin(rev)) if not rev.all() else len(rev)
        target_idx = df.index[-n_tail:]
        calendars.append({"track": "price", "task": k,
                          "asof": target_idx[0] - pd.Timedelta("30min"),
                          "target_start": target_idx[0], "target_end": target_idx[-1]})
    tasks = pd.DataFrame(calendars)

    sol_dir = next((ROOT / "Price").glob("Solution*"))
    sol = pd.read_csv(next(sol_dir.glob("*.csv")))
    tcol = next(c for c in sol.columns if "time" in c.lower())
    sol[tcol] = pd.to_datetime(sol[tcol], format="%m%d%Y %H:%M")
    pcol = next(c for c in sol.columns if "price" in c.lower())
    sol = sol.set_index(tcol)[[pcol]].rename(columns={pcol: "price"})

    last = files[N_TASKS]
    price = last[["Zonal Price"]].rename(columns={"Zonal Price": "price"})
    price = pd.concat([price.dropna(), sol]).sort_index()
    price = price[~price.index.duplicated(keep="last")]

    loads = last[["Forecasted Total Load", "Forecasted Zonal Load"]].rename(
        columns={"Forecasted Total Load": "total_load_fc",
                 "Forecasted Zonal Load": "zonal_load_fc"})
    return price, loads, tasks


# -- leaderboard ------------------------------------------------------------------------


def build_leaderboard() -> pd.DataFrame:
    """Mean pinball per team over the scored evaluation weeks, per track."""
    xl = pd.ExcelFile(ROOT / "Provisional_Leaderboard_V2.xlsx")
    rows = []
    for prefix, track in (("L", "load"), ("P", "price"), ("W", "wind"), ("S", "solar")):
        df = xl.parse(f"{prefix}-score", header=0)
        df.columns = [str(c).strip().lower() for c in df.columns]
        name_col = next(c for c in df.columns if "name" in c)
        score_col = next(c for c in df.columns if "score" in c)
        week_col = next(c for c in df.columns if "week" in c)
        df = df[[name_col, score_col, week_col]].dropna()
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
        pivot = df.pivot_table(index=name_col, columns=week_col, values=score_col)
        complete = pivot.dropna()  # rank only teams scored in every week
        mean = complete.mean(axis=1).sort_values()
        for rank, (team, score) in enumerate(mean.items(), start=1):
            rows.append({"track": track, "rank": rank, "team": str(team).strip(),
                         "pinball": float(score), "n_weeks": pivot.shape[1]})
    return pd.DataFrame(rows)


# -- assembly ---------------------------------------------------------------------------


def bitemporal(frame: pd.DataFrame, tasks: pd.DataFrame, final_release) -> pd.DataFrame:
    ts = frame.index.to_series()
    knowledge, _ = _knowledge_and_issue(ts, tasks, final_release)
    out = frame.copy()
    out["knowledge_time"] = knowledge.to_numpy()
    return out


def forecast_frame(frame: pd.DataFrame, tasks: pd.DataFrame) -> pd.DataFrame:
    ts = frame.index.to_series()
    _, issue = _knowledge_and_issue(ts, tasks, tasks["asof"].iloc[-1])
    out = frame.copy()
    out["issue_time"] = issue.to_numpy()
    out = out.reset_index().rename(columns={out.index.name or "index": "valid_time",
                                            "TIMESTAMP": "valid_time",
                                            "timestamp": "valid_time"})
    return out.set_index(["issue_time", "valid_time"]).sort_index()


MANIFEST_BASE = {
    "name": "gefcom2014",
    "description": "Global Energy Forecasting Competition 2014 — probabilistic "
                   "load, price, wind and solar tracks (99 quantiles, pinball "
                   "loss, 15 incremental monthly tasks per track).",
    "citation": "Hong, Pinson, Fan, Zareipour, Troccoli & Hyndman, "
                "'Probabilistic energy forecasting: GEFCom2014 and beyond', "
                "IJF 32(3), 2016.",
}


def field_spec(path, kind="actual", knowledge=False, description=None):
    spec = {"path": path, "kind": kind}
    if knowledge:
        spec["knowledge_col"] = "knowledge_time"
    if description:
        spec["description"] = description
    return spec


def build(out_dir: Path) -> None:
    pub, priv = out_dir / "public", out_dir / "private"
    (pub / "data").mkdir(parents=True, exist_ok=True)
    (priv / "data").mkdir(parents=True, exist_ok=True)

    all_tasks = []
    fields_pub, fields_priv = {}, {}

    def emit(name, frame, kind, description=None):
        """Split one field's rows by whether they were knowable at task 4."""
        cut = cutovers[name]
        col = "knowledge_time" if kind == "actual" else None
        if kind == "actual":
            mask = frame["knowledge_time"] <= cut
            pub_frame, priv_frame = frame[mask], frame[~mask]
        else:
            issues = frame.index.get_level_values("issue_time")
            pub_frame, priv_frame = frame[issues <= cut], frame[issues > cut]
        pub_frame.to_parquet(pub / "data" / f"{name}.parquet")
        fields_pub[name] = field_spec(f"data/{name}.parquet", kind,
                                      knowledge=kind == "actual", description=description)
        if len(priv_frame):
            priv_frame.to_parquet(priv / "data" / f"{name}.parquet")
            fields_priv[name] = field_spec(f"data/{name}.parquet", kind,
                                           knowledge=kind == "actual", description=description)

    # -- track calendars ------------------------------------------------------
    tasks_by_track = {t: month_task_calendar(t) for t in ("load", "wind", "solar")}
    price_targets, price_loads, price_tasks = build_price()
    tasks_by_track["price"] = price_tasks
    cutover_by_track = {t: df.loc[df.task == HOLDOUT_FIRST_TASK, "asof"].iloc[0]
                        for t, df in tasks_by_track.items()}

    def final_release(track):
        df = tasks_by_track[track]
        return df["target_end"].iloc[-1] + pd.Timedelta("31D")

    cutovers = {}

    # solar
    print("building solar ...")
    s_targets, s_nwp = build_solar(tasks_by_track["solar"])
    cutovers.update({"solar_targets": cutover_by_track["solar"],
                     "solar_nwp": cutover_by_track["solar"]})
    emit("solar_targets", bitemporal(s_targets, tasks_by_track["solar"], final_release("solar")),
         "actual", "Normalized solar power, 3 Australian plants (z1-z3)")
    emit("solar_nwp", forecast_frame(s_nwp, tasks_by_track["solar"]),
         "forecast", "12 ECMWF variables per plant, issued per task")

    # wind
    print("building wind ...")
    w_targets, w_nwp = build_wind(tasks_by_track["wind"])
    cutovers.update({"wind_targets": cutover_by_track["wind"],
                     "wind_nwp": cutover_by_track["wind"]})
    emit("wind_targets", bitemporal(w_targets, tasks_by_track["wind"], final_release("wind")),
         "actual", "Normalized wind power, 10 Australian farms (z1-z10)")
    emit("wind_nwp", forecast_frame(w_nwp, tasks_by_track["wind"]),
         "forecast", "ECMWF U/V at 10m and 100m per farm, issued per task")

    # load
    print("building load ...")
    l_targets, l_temp = build_load(tasks_by_track["load"])
    cutovers.update({"load_targets": cutover_by_track["load"],
                     "load_temperature": cutover_by_track["load"]})
    emit("load_targets", bitemporal(l_targets, tasks_by_track["load"], final_release("load")),
         "actual", "Hourly utility load")
    emit("load_temperature", bitemporal(l_temp, tasks_by_track["load"], final_release("load")),
         "actual", "25 weather stations (w1-w25); NO future temperature — "
                   "modelling temperature uncertainty is the challenge")

    # price
    print("building price ...")
    cutovers.update({"price_targets": cutover_by_track["price"],
                     "price_load_forecasts": cutover_by_track["price"]})
    emit("price_targets", bitemporal(price_targets, price_tasks, final_release("price")),
         "actual", "Hourly zonal electricity price")
    emit("price_load_forecasts", forecast_frame(price_loads, price_tasks),
         "forecast", "Given zonal + system load forecasts, issued per task")

    # calendars + leaderboard (packaged AND in the public repo)
    tasks_frame = pd.concat(tasks_by_track.values(), ignore_index=True)
    leaderboard = build_leaderboard()
    PKG.mkdir(parents=True, exist_ok=True)
    for name, frame in (("tasks.csv", tasks_frame), ("leaderboard.csv", leaderboard)):
        frame.to_csv(pub / name, index=False)
        frame.to_csv(PKG / name, index=False)

    (pub / "rebase.yaml").write_text(yaml.safe_dump(
        {**MANIFEST_BASE, "fields": fields_pub}, sort_keys=False))
    (priv / "rebase.yaml").write_text(yaml.safe_dump(
        {**MANIFEST_BASE, "name": "gefcom2014-private",
         "description": "GEFCom2014 holdout releases (tasks 4-15) — verifier-only.",
         "fields": fields_priv}, sort_keys=False))

    for repo in (pub, priv):
        for f in sorted(repo.rglob("*")):
            if f.is_file():
                print(f"  {f.relative_to(out_dir)}  {f.stat().st_size / 1e6:8.2f} MB")


def upload(out_dir: Path, owner: str, token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    for sub, private in (("public", False), ("private", True)):
        repo_id = f"{owner}/gefcom2014{'-private' if private else ''}"
        print(f"uploading {out_dir / sub} -> {repo_id} (private={private}) ...")
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        api.upload_folder(folder_path=str(out_dir / sub), repo_id=repo_id,
                          repo_type="dataset",
                          commit_message="Build GEFCom2014 benchmark dataset")
        print(f"  -> rb://dataset/{repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=CACHE / "build")
    ap.add_argument("--owner", default="rebase-energy")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    if not ROOT.exists():
        raise SystemExit(
            f"{ROOT} not found — download https://www.dropbox.com/s/pqenrr2mcvl0hk9/"
            f"GEFCom2014.zip and extract under {CACHE / 'raw'}"
        )
    build(args.out_dir)

    if args.upload:
        token = args.token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("--upload needs an HF token")
        upload(args.out_dir, args.owner, token)


if __name__ == "__main__":
    main()
