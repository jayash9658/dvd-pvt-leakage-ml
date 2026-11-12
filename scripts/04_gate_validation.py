"""
Part B: gate-level validation (steps 3-5 of the spec).

  3. SPICE reference: INV, NAND2, NOR2, AND2, OR2 as transistor netlists, every
     input combination (18 cases) at 200 random corners.
  4. ML prediction: the node voltages recorded from SPICE give each
     transistor's Vd, Vg, Vs, Vb; the four Part A models predict its terminal
     currents; the gate leakage follows from them (see gate_lib.py).
  5. Compare: gate results table and parity plot.

Also recorded: the ML estimate of the internal node voltages (Kirchhoff's law
solved with the transistor models) and the leakage that results from them.

Outputs
  data/gates/corners.csv                 the 200 corners (reused in Part C)
  data/gates/gate_simulations.csv        every case x corner: voltages, SPICE and ML leakage
  netlists/gates/<GATE>_example.cir      one netlist per gate (corner 0)
  results/tables/gate_results_table.csv  18 rows + one row per gate + overall
  results/tables/gate_results.txt        the same, formatted for the report
  results/plots/gate_parity_plot.png

Usage:  python scripts/04_gate_validation.py [--workers 12]
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
import gate_lib as gl              # noqa: E402

WORK = gl.GATE_DATA / "_work"
NETLIST_OUT = config.NETLIST_DIR / "gates"
BATCH = 200


def _valid(r, gate, vdd) -> bool:
    """A usable solution: converged, and every node voltage inside [0, VDD] (1 uV margin)."""
    if r is None:
        return False
    return all(-1e-6 <= r[f"v({n})"] <= vdd + 1e-6 for n in gl.GATES[gate]["internal"] + ["out"])


def run_spice(corners, exe, workers) -> pd.DataFrame:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    su.write_param_model(WORK / gl.PARAM_LIB_NAME)
    todo = [(gate, combo, ci) for gate in gl.GATES for combo in gl.input_combos(gate) for ci in range(len(corners))]
    good, n_retry = {}, 0
    for mode in (0, 1, 2):            # retry unusable solutions with another starting guess
        jobs = []
        for gate in gl.GATES:
            names = []
            for g, combo, ci in todo:
                if g == gate:
                    name = f"{gate}_{combo}_{ci:03d}_m{mode}.cir"
                    (WORK / name).write_text(gl.gate_netlist(gate, combo, corners.iloc[ci], guess_mode=mode))
                    names.append(name)
            for k in range(0, len(names), BATCH):
                jobs.append((names[k:k + BATCH], gl.gate_print_vectors(gate), f"drv_{gate}_{mode}_{k:05d}.cir"))
        res = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for f in [ex.submit(su.run_batch, exe, WORK, *j) for j in jobs]:
                res.update(f.result())
        left = []
        for gate, combo, ci in todo:
            r = res.get(f"{gate}_{combo}_{ci:03d}_m{mode}.cir")
            if _valid(r, gate, corners.iloc[ci]["vdd"]):
                good[(gate, combo, ci)] = r
            else:
                left.append((gate, combo, ci))
        todo = left
        if mode == 0:
            n_retry = len(todo)
        if not todo:
            break
    print(f"SPICE: {n_retry} simulations needed a retry, {len(todo)} still unusable")

    out = []
    for gate in gl.GATES:
        g = gl.GATES[gate]
        for combo in gl.input_combos(gate):
            for ci in range(len(corners)):
                c, r = corners.iloc[ci], good.get((gate, combo, ci))
                rec = {"gate": gate, "input": combo, "corner": ci, "vdd": c["vdd"], "temp": c["temp"],
                       "spice_leak": np.nan}
                if r is not None:
                    leak = -r["i(vdd)"] - sum(r[f"i(v{n})"] for n, b in zip(g["inputs"], combo) if b == "1")
                    rec.update({"v_out_spice": r["v(out)"], "spice_leak": leak})
                    for n in g["internal"]:
                        rec[f"v_{n}_spice"] = r[f"v({n})"]
                out.append(rec)
    shutil.rmtree(WORK)
    return pd.DataFrame(out)


def run_ml(sim, corners) -> pd.DataFrame:
    sim = sim.copy()
    for col in ("ml_leak", "ml_leak_mlnodes"):
        sim[col] = np.nan
    for gate, grp in sim.groupby("gate", sort=False):
        g = gl.GATES[gate]
        cr = corners.iloc[grp["corner"].to_numpy()].reset_index(drop=True)
        combos = grp["input"].tolist()
        nodes = {n: grp[f"v_{n}_spice"].to_numpy() for n in g["internal"]}
        nodes["out"] = grp["v_out_spice"].to_numpy()
        sim.loc[grp.index, "ml_leak"] = gl.gate_leakage_ml(gate, combos, cr, nodes)
        if g["internal"]:
            ml_nodes = gl.solve_internal_nodes(gate, combos, cr, nodes["out"])
            for n, val in ml_nodes.items():
                sim.loc[grp.index, f"v_{n}_ml"] = val
            sim.loc[grp.index, "ml_leak_mlnodes"] = gl.gate_leakage_ml(
                gate, combos, cr, dict(ml_nodes, out=nodes["out"]))
        else:
            sim.loc[grp.index, "ml_leak_mlnodes"] = sim.loc[grp.index, "ml_leak"]
    sim["err_pct"] = 100 * (sim["ml_leak"] - sim["spice_leak"]).abs() / sim["spice_leak"].abs()
    return sim


def results_table(sim) -> pd.DataFrame:
    rows = []
    for (gate, combo) in gl.CASES18:
        g = sim[(sim["gate"] == gate) & (sim["input"] == combo)]
        m = gl.leak_metrics(g["spice_leak"], g["ml_leak"])
        m2 = gl.leak_metrics(g["spice_leak"], g["ml_leak_mlnodes"])
        rows.append({"gate": gate, "input": combo, **m,
                     "median_err_pct_mlnodes": m2["median_err_pct"], "p95_err_pct_mlnodes": m2["p95_err_pct"]})
    for gate in list(gl.GATES) + ["ALL"]:
        g = sim if gate == "ALL" else sim[sim["gate"] == gate]
        m = gl.leak_metrics(g["spice_leak"], g["ml_leak"])
        m2 = gl.leak_metrics(g["spice_leak"], g["ml_leak_mlnodes"])
        rows.append({"gate": gate, "input": "all", **m,
                     "median_err_pct_mlnodes": m2["median_err_pct"], "p95_err_pct_mlnodes": m2["p95_err_pct"]})
    return pd.DataFrame(rows)


def write_text(table, sim, path, n_fail):
    L = ["PART B: GATE-LEVEL VALIDATION (ML vs SPICE)",
         f"{config.TECH}, VDD = {config.VDD_NOM} V +/- 10 %, {config.N_CORNERS} corners (seed {config.SEED_CORNERS}), "
         f"18 gate/input cases, {len(sim)} simulations" + (f" ({n_fail} failed)" if n_fail else ""),
         "",
         "Leakage = current drawn from all sources at VDD (supply + high inputs) = static power / VDD.",
         "ML leakage: transistor terminal voltages from the SPICE node voltages -> Part A models -> gate leakage.",
         "Error = |ML - SPICE| / SPICE. Bias = median of (ML - SPICE) / SPICE. Input order is AB",
         "(NAND2: A drives the NMOS next to the output; NOR2: A drives the PMOS next to VDD).",
         "'ML nodes' columns: the same, but with internal node voltages solved by the ML models themselves.",
         "",
         f"{'Gate':6s} {'In':>3s} {'SPICE med (A)':>14s} {'ML med (A)':>12s} {'Bias %':>8s} {'Med err %':>10s} "
         f"{'p95 err %':>10s} {'Mean err %':>10s} | {'ML nodes: med %':>15s} {'p95 %':>8s}",
         "-" * 108]
    for r in table.itertuples(index=False):
        if r.input == "all" and r.gate == "INV":
            L.append("-" * 108)
        L.append(f"{r.gate:6s} {r.input:>3s} {r.spice_median_A:14.4e} {r.ml_median_A:12.4e} {r.bias_pct:8.2f} "
                 f"{r.median_err_pct:10.2f} {r.p95_err_pct:10.2f} {r.mean_err_pct:10.2f} | "
                 f"{r.median_err_pct_mlnodes:15.2f} {r.p95_err_pct_mlnodes:8.2f}")
    allrow = table.iloc[-1]
    L += ["", f"Overall mean accuracy = {100 - allrow['mean_err_pct']:.2f} %  "
              f"(mean error {allrow['mean_err_pct']:.2f} %, median {allrow['median_err_pct']:.2f} %)", ""]
    worst = sim.sort_values("err_pct", ascending=False).head(10)
    L.append("Ten largest errors:")
    for r in worst.itertuples(index=False):
        L.append(f"  {r.gate:6s} {r.input:>3s} corner {r.corner:3d}  VDD {r.vdd:.3f} V  T {r.temp:6.1f} C  "
                 f"SPICE {r.spice_leak:.3e} A  ML {r.ml_leak:.3e} A  error {r.err_pct:.2f} %")
    path.write_text("\n".join(L) + "\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    exe = su.find_ngspice()
    gl.GATE_DATA.mkdir(parents=True, exist_ok=True)
    NETLIST_OUT.mkdir(parents=True, exist_ok=True)
    tables, plots = config.RESULTS_DIR / "tables", config.RESULTS_DIR / "plots"

    corners = gl.draw_corners()
    corners.to_csv(gl.GATE_DATA / "corners.csv", index=False, float_format="%.10e")

    # example netlists: one per gate (corner 0, all inputs high except INV with A=0)
    su.write_param_model(NETLIST_OUT / gl.PARAM_LIB_NAME)
    for gate in gl.GATES:
        combo = gl.input_combos(gate)[0]
        (NETLIST_OUT / f"{gate}_example.cir").write_text(
            gl.gate_netlist(gate, combo, corners.iloc[0], with_control=True))

    t0 = time.time()
    sim = run_spice(corners, exe, args.workers)
    n_fail = int(sim["spice_leak"].isna().sum())
    print(f"SPICE: {len(sim) - n_fail}/{len(sim)} gate simulations in {time.time() - t0:.0f} s")
    sim = sim.dropna(subset=["spice_leak"]).reset_index(drop=True)

    t0 = time.time()
    sim = run_ml(sim, corners)
    print(f"ML: done in {time.time() - t0:.0f} s")
    cols = ["gate", "input", "corner", "vdd", "temp", "v_out_spice", "v_x_spice", "v_y_spice", "v_n_spice",
            "v_x_ml", "v_y_ml", "v_n_ml", "spice_leak", "ml_leak", "ml_leak_mlnodes", "err_pct"]
    sim["input"] = sim["input"].astype(str)
    sim[cols].to_csv(gl.GATE_DATA / "gate_simulations.csv", index=False, float_format="%.10e")

    table = results_table(sim)
    table.to_csv(tables / "gate_results_table.csv", index=False, float_format="%.6g")
    print(write_text(table, sim, tables / "gate_results.txt", n_fail))
    gl.parity_plot(plots / "gate_parity_plot.png",
                   [(g, s["spice_leak"].to_numpy(), s["ml_leak"].to_numpy()) for g, s in sim.groupby("gate", sort=False)],
                   f"Gate leakage: ML vs SPICE ({len(sim)} simulations, 18 cases x {config.N_CORNERS} corners)")


if __name__ == "__main__":
    main()
