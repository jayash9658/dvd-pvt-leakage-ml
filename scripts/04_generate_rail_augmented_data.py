"""
Part B follow-up: augment each transistor case with extra samples
concentrated near Vd/Vs rail extremes.

Why: gate-level validation showed the Part A models are much less accurate
specifically when a transistor's actual operating point has one terminal
near 0V and the other near VDD -- exactly what a real gate produces for its
"off" device, but a region under-represented by the original uniform(0,VDD)
Vd/Vs sampling. This draws additional samples with Vd/Vs fractions pulled
from a U-shaped Beta(0.3, 0.3) distribution (concentrates mass near 0 and 1,
still covers the full range) instead of uniform, merges them with the
existing 10,000-row CSVs, and leaves everything else (VDD/W/temp/process
param ranges, case definitions, netlist format) identical to
01_generate_transistor_data.py.

Usage: python scripts/04_generate_rail_augmented_data.py [--n 5000]
"""
import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config                      # noqa: E402
import spice_utils as su           # noqa: E402
from importlib import import_module
gen = import_module("01_generate_transistor_data")

# Point spice_utils at the ngspice installed for this project (different
# machine than the one config.py's hardcoded candidate paths assume)
os.environ["NGSPICE"] = r"C:\ngspice\Spice64\bin\ngspice_con.exe"

SEED_AUGMENT = {"NMOS_ON": 2101, "NMOS_OFF": 2202, "PMOS_ON": 2303, "PMOS_OFF": 2404}
WORK_ROOT = config.DATA_DIR / "transistor" / "_augment_work"
PARAM_LIB_NAME = f"{config.TECH}_param.lib"
BATCH_SIZE = 250


def draw_rail_samples(case: str, n: int, nominal: dict) -> pd.DataFrame:
    spec = config.CASES[case]
    dev = spec["type"]
    rng = np.random.default_rng(SEED_AUGMENT[case])

    vdd = rng.uniform(config.VDD_NOM * (1 - config.VDD_TOL),
                      config.VDD_NOM * (1 + config.VDD_TOL), n)
    # U-shaped instead of uniform: concentrates near 0 and 1, still full range.
    # Applied to Vd/Vs (round 1 finding: off-device at rail-to-rail Vds is
    # hardest) AND temperature (round 2 finding: the worst gate-level corners
    # were consistently near -40C/125C, the ends of the temp range -- same
    # under-sampled-extreme problem, different parameter).
    vd = rng.beta(0.3, 0.3, n) * vdd
    vs = rng.beta(0.3, 0.3, n) * vdd
    w = rng.uniform(config.W_MIN_MULT, config.W_MAX_MULT, n) * config.L_MIN
    t_lo, t_hi = config.TEMP_RANGE
    temp = t_lo + rng.beta(0.3, 0.3, n) * (t_hi - t_lo)
    proc = {p: nominal[dev][p] * rng.uniform(1 - config.PROC_TOL, 1 + config.PROC_TOL, n)
            for p in config.PROC_PARAMS}
    vg = vdd if spec["vg"] == "VDD" else np.zeros(n)
    vb = vdd if spec["vb"] == "VDD" else np.zeros(n)

    return pd.DataFrame({"vdd": vdd, "temp": temp, "w": w, "vd": vd, "vs": vs,
                         "vg": vg, "vb": vb, **proc})


def run_case_no_example(case: str, n: int, exe: str, nominal: dict, workers: int) -> pd.DataFrame:
    """Same as 01_generate_transistor_data.run_case, but never touches the
    netlists/transistor/<case>_example.cir deliverable (that stays as his
    original sample 0, not ours)."""
    df = draw_rail_samples(case, n, nominal)

    work = WORK_ROOT / case
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    su.write_param_model(work / PARAM_LIB_NAME)

    names = []
    for i, row in enumerate(df.itertuples(index=False)):
        name = f"s{i:05d}.cir"
        (work / name).write_text(gen.sample_netlist(case, i, row, nominal, PARAM_LIB_NAME))
        names.append(name)

    batches = [names[i:i + BATCH_SIZE] for i in range(0, len(names), BATCH_SIZE)]
    vecs = ["i(vd)", "i(vg)", "i(vs)", "i(vb)"]
    results = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(su.run_batch, exe, work, b, vecs, f"driver_{k:03d}.cir")
                for k, b in enumerate(batches)]
        for f in futs:
            results.update(f.result())
    dt = time.time() - t0

    cur = np.full((n, 4), np.nan)
    for i, name in enumerate(names):
        r = results.get(name)
        if r:
            cur[i] = [-r["i(vd)"], -r["i(vg)"], -r["i(vs)"], -r["i(vb)"]]
    df["id"], df["ig"], df["is"], df["ib"] = cur.T

    n_fail = int(np.isnan(cur).any(axis=1).sum())
    print(f"  {case}: {n - n_fail}/{n} samples simulated in {dt:.1f} s"
          + (f"  ({n_fail} FAILED - dropped)" if n_fail else ""))
    df = df.dropna().reset_index(drop=True)
    shutil.rmtree(work)
    return df[config.CSV_COLUMNS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=list(config.CASES))
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    exe = su.find_ngspice()
    nominal = su.read_nominal_params()
    print(f"ngspice: {exe}")

    for case in args.cases:
        existing = pd.read_csv(config.DATA_DIR / "transistor" / f"{case}.csv")
        print(f"\n{case}: existing {len(existing)} rows, generating {args.n} rail-biased rows...")
        new = run_case_no_example(case, args.n, exe, nominal, args.workers)
        merged = pd.concat([existing, new], ignore_index=True)[config.CSV_COLUMNS]
        out = config.DATA_DIR / "transistor" / f"{case}.csv"
        merged.to_csv(out, index=False, float_format="%.10e")
        print(f"  -> {out}  ({len(merged)} rows total, {len(new)} new)")


if __name__ == "__main__":
    main()
