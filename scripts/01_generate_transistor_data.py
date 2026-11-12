"""
Part A, step 1: generate the single-transistor datasets.

For each case (NMOS_ON, NMOS_OFF, PMOS_ON, PMOS_OFF) this script
  1. draws N_SAMPLES random samples (VDD first, then Vd, Vs, W, T and the five
     process parameters), all independent, uniform as in Section 4 of the spec
     except for a mild enrichment of the extremes of Vd, Vs and T (see
     config.EDGE_*: ~15 % of samples near each variable's ends instead of
     10-12 %; the rest stays uniform),
  2. writes one single-FET netlist per sample (each terminal has its own
     voltage source; required .options line included),
  3. simulates them with ngspice (op analysis), many circuits per ngspice
     process and several processes in parallel,
  4. saves data/transistor/<CASE>.csv with columns
     vdd,temp,w,vd,vs,vg,vb,vth0,toxe,ndep,u0,xj,id,ig,is,ib
  5. saves data/transistor/components/<CASE>_components.csv: BSIM4's internal
     current components for the same samples (channel current, junction
     currents, gate tunnelling, GIDL, impact ionisation), used to train the
     physics-based models. Same row order as <CASE>.csv.

Units: volts, degC, metres (w, toxe, xj), cm^-3 (ndep), m^2/Vs (u0), amperes.
Current sign: current flowing INTO the device terminal (so id+ig+is+ib = 0).
ngspice reports I(Vx) as current from the + node through the source, so the
terminal current is -I(Vx).

Usage:  python scripts/01_generate_transistor_data.py [--cases NMOS_ON ...] [--n 10000]
"""
import argparse
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

OUT_DIR = config.DATA_DIR / "transistor"
NETLIST_OUT = config.NETLIST_DIR / "transistor"   # keeps the 4 example netlists only
WORK_ROOT = OUT_DIR / "_work"                     # temporary per-sample netlists (deleted)
PARAM_LIB_NAME = f"{config.TECH}_param.lib"
BATCH_SIZE = 250
COMP_VECS = [f"@m1[{c}]" for c in config.COMPONENTS]


def edge_uniform(rng, n, lo, hi, band, p_edge):
    """
    Uniform on [lo, hi], except that with probability p_edge a value is drawn
    uniformly inside [lo, lo+band] or [hi-band, hi] (either end, 50/50).
    All random numbers are drawn for every sample, so the stream is the same
    whatever the outcome.
    """
    u = rng.uniform(lo, hi, n)
    use_edge = rng.uniform(0, 1, n) < p_edge
    top = rng.uniform(0, 1, n) < 0.5
    e = rng.uniform(0, band, n)
    edge_val = np.where(top, hi - e, lo + e)
    return np.where(use_edge, edge_val, u)


def draw_samples(case: str, n: int, nominal: dict) -> pd.DataFrame:
    spec = config.CASES[case]
    dev = spec["type"]
    rng = np.random.default_rng(config.SEED_TRANSISTOR_DATA[case])

    vdd = rng.uniform(config.VDD_NOM * (1 - config.VDD_TOL),
                      config.VDD_NOM * (1 + config.VDD_TOL), n)     # VDD first
    vd = edge_uniform(rng, n, 0.0, 1.0, config.EDGE_V_BAND, config.EDGE_P_V) * vdd
    vs = edge_uniform(rng, n, 0.0, 1.0, config.EDGE_V_BAND, config.EDGE_P_V) * vdd
    w = rng.uniform(config.W_MIN_MULT, config.W_MAX_MULT, n) * config.L_MIN
    temp = edge_uniform(rng, n, *config.TEMP_RANGE, config.EDGE_TEMP_BAND, config.EDGE_P_TEMP)
    proc = {p: nominal[dev][p] * rng.uniform(1 - config.PROC_TOL, 1 + config.PROC_TOL, n)
            for p in config.PROC_PARAMS}
    vg = vdd if spec["vg"] == "VDD" else np.zeros(n)
    vb = vdd if spec["vb"] == "VDD" else np.zeros(n)

    df = pd.DataFrame({"vdd": vdd, "temp": temp, "w": w, "vd": vd, "vs": vs,
                       "vg": vg, "vb": vb, **proc})
    return df


def sample_netlist(case: str, idx: int, row, nominal: dict, lib_ref: str,
                   with_control: bool = False) -> str:
    dev = config.CASES[case]["type"]
    nm = dict(nominal["nmos"])
    pm = dict(nominal["pmos"])
    target = nm if dev == "nmos" else pm
    for p in config.PROC_PARAMS:
        target[p] = getattr(row, p)
    txt = (f"* {config.TECH} single {dev.upper()} - case {case} - sample {idx}\n"
           f"* VDD={row.vdd:.6f} V, T={row.temp:.4f} C, W={row.w:.6e} m\n"
           + su.param_lines(nm, pm)
           + f'.include "{lib_ref}"\n'
           + config.SPICE_OPTIONS + "\n"
           + f".temp {row.temp:.6f}\n\n"
           + f"M1 drain gate source body {dev} W={row.w:.6e} L={config.L_MIN:.3e}\n"
           + f"Vd drain 0 dc {row.vd:.9f}\n"
           + f"Vg gate 0 dc {row.vg:.9f}\n"
           + f"Vs source 0 dc {row.vs:.9f}\n"
           + f"Vb body 0 dc {row.vb:.9f}\n")
    if with_control:
        txt += ("\n.control\nset numdgt=12\nop\n"
                "* terminal currents INTO the device = -I(source)\n"
                "let id = -i(vd)\nlet ig = -i(vg)\nlet is = -i(vs)\nlet ib = -i(vb)\n"
                "print id ig is ib\n"
                "echo Kirchhoff check (id+ig+is+ib):\nprint id+ig+is+ib\n.endc\n")
    return txt + ".end\n"


def run_case(case: str, n: int, exe: str, nominal: dict, workers: int):
    df = draw_samples(case, n, nominal)

    # per-sample netlists are written to a temporary folder, simulated, then deleted;
    # only the example netlist (sample 0) is kept in netlists/transistor/
    work = WORK_ROOT / case
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    su.write_param_model(work / PARAM_LIB_NAME)

    names = []
    for i, row in enumerate(df.itertuples(index=False)):
        name = f"s{i:05d}.cir"
        (work / name).write_text(sample_netlist(case, i, row, nominal, PARAM_LIB_NAME))
        names.append(name)

    # example netlist deliverable (sample 0, standalone, runnable with `source`)
    NETLIST_OUT.mkdir(parents=True, exist_ok=True)
    (NETLIST_OUT / f"{case}_example.cir").write_text(
        sample_netlist(case, 0, next(df.itertuples(index=False)), nominal,
                       PARAM_LIB_NAME, with_control=True))

    batches = [names[i:i + BATCH_SIZE] for i in range(0, len(names), BATCH_SIZE)]
    vecs = ["i(vd)", "i(vg)", "i(vs)", "i(vb)"] + COMP_VECS
    results = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(su.run_batch, exe, work, b, vecs, f"driver_{k:03d}.cir")
                for k, b in enumerate(batches)]
        for f in futs:
            results.update(f.result())
    dt = time.time() - t0

    # current INTO the device terminal = -I(source)
    cur = np.full((n, 4), np.nan)
    comp = np.full((n, len(COMP_VECS)), np.nan)
    for i, name in enumerate(names):
        r = results.get(name)
        if r:
            cur[i] = [-r["i(vd)"], -r["i(vg)"], -r["i(vs)"], -r["i(vb)"]]
            comp[i] = [r[v] for v in COMP_VECS]
    df["id"], df["ig"], df["is"], df["ib"] = cur.T
    comp = pd.DataFrame(comp, columns=config.COMPONENTS)

    n_fail = int(np.isnan(cur).any(axis=1).sum())
    print(f"  {case}: {n - n_fail}/{n} samples simulated in {dt:.1f} s"
          + (f"  ({n_fail} FAILED - dropped)" if n_fail else ""))
    ok = ~(np.isnan(cur).any(axis=1) | comp.isna().any(axis=1).to_numpy())
    df, comp = df[ok].reset_index(drop=True), comp[ok].reset_index(drop=True)
    shutil.rmtree(work)
    return df[config.CSV_COLUMNS], comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=list(config.CASES))
    ap.add_argument("--n", type=int, default=config.N_SAMPLES)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write CSVs, example netlists and temp files here instead of the "
                         "project folders (e.g. to verify reproducibility)")
    args = ap.parse_args()

    global OUT_DIR, NETLIST_OUT, WORK_ROOT
    if args.out_dir is not None:
        OUT_DIR = args.out_dir / "data"
        NETLIST_OUT = args.out_dir / "netlists"
        WORK_ROOT = OUT_DIR / "_work"

    exe = su.find_ngspice()
    nominal = su.read_nominal_params()
    for dev in ("nmos", "pmos"):
        for p in config.PROC_PARAMS:
            assert np.isclose(nominal[dev][p], config.NOMINAL[dev][p]), (dev, p)
    print(f"ngspice: {exe} ({su.ngspice_version(exe)})")
    print(f"model  : {config.MODEL_FILE.name}, VDD_nom = {config.VDD_NOM} V, "
          f"L = {config.L_MIN*1e9:.0f} nm, N = {args.n} per case")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    su.write_param_model(NETLIST_OUT / PARAM_LIB_NAME)   # used by example netlists
    for case in args.cases:
        df, comp = run_case(case, args.n, exe, nominal, args.workers)
        out = OUT_DIR / f"{case}.csv"
        df.to_csv(out, index=False, float_format="%.10e")
        (OUT_DIR / "components").mkdir(exist_ok=True)
        comp.to_csv(OUT_DIR / "components" / f"{case}_components.csv", index=False,
                    float_format="%.10e")
        shown = out.relative_to(config.ROOT) if out.is_relative_to(config.ROOT) else out
        print(f"  -> {shown}  ({len(df)} rows)")
    if WORK_ROOT.exists() and not any(WORK_ROOT.iterdir()):
        WORK_ROOT.rmdir()


if __name__ == "__main__":
    main()
