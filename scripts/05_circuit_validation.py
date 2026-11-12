"""
Part C: test-circuit validation on the 3-bit carry-lookahead adder (steps 6-8).

The adder is built only from the Part B gates (NAND2, AND2, OR2; an XOR is four
NAND2):
    g_i = a_i AND b_i            p_i = a_i XOR b_i            s_i = p_i XOR c_i
    c1 = g0 + p0 c0
    c2 = g1 + p1 g0 + p1 p0 c0
    c3 = g2 + p2 g1 + p2 p1 g0 + p2 p1 p0 c0
42 gates (24 NAND2, 12 AND2, 6 OR2), 204 transistors, 7 inputs -> 128 input vectors.

  6. SPICE reference: the complete transistor netlist, all 128 input vectors at
     the same 200 corners as Part B (25,600 operating points).
  7. ML prediction: for each input vector the logic value at every gate input
     is found by logic simulation, the gate model of Part B is called for each
     gate, and the 42 gate leakages are summed.
  8. Compare.

Leakage = current drawn from all sources at VDD (supply + high primary inputs),
the same definition as for the gates.

Outputs
  netlists/circuit/CLA3.cir                 the test-circuit netlist (corner 0)
  data/circuit/cla3_simulations.csv         every corner x input vector
  results/tables/circuit_results_table.csv  overall row + one row per input vector
  results/tables/circuit_results.txt        summary for the report
  results/plots/circuit_parity_plot.png

Usage:  python scripts/05_circuit_validation.py [--workers 12]      (run 04 first)
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

DATA = config.DATA_DIR / "circuit"
WORK = DATA / "_work"
NETLIST_OUT = config.NETLIST_DIR / "circuit"
NB = config.CLA_BITS
INPUTS = [f"a{i}" for i in range(NB)] + [f"b{i}" for i in range(NB)] + ["c0"]
OUTPUTS = [f"s{i}" for i in range(NB)] + [f"c{NB}"]


# ----------------------------------------------------------------------------
# Circuit: list of gate instances (name, type, input a, input b, output)
# ----------------------------------------------------------------------------
def build_cla():
    inst = []

    def add(typ, a, b, out):
        inst.append((f"x{len(inst) + 1:02d}", typ, a, b, out))
        return out

    def xor(a, b, out):
        n1 = add("NAND2", a, b, f"{out}_n1")
        n2 = add("NAND2", a, n1, f"{out}_n2")
        n3 = add("NAND2", b, n1, f"{out}_n3")
        return add("NAND2", n2, n3, out)

    for i in range(NB):
        add("AND2", f"a{i}", f"b{i}", f"g{i}")
        xor(f"a{i}", f"b{i}", f"p{i}")
    # carry lookahead: c_{i+1} = g_i + p_i g_{i-1} + ... + p_i..p_0 c0
    for i in range(NB):
        terms, prod = [f"g{i}"], f"p{i}"
        for j in range(i - 1, -2, -1):
            src = f"g{j}" if j >= 0 else "c0"
            terms.append(add("AND2", prod, src, f"t{i}_{j + 1}"))
            if j >= 0:
                prod = add("AND2", prod, f"p{j}", f"pp{i}_{j}")
        acc = terms[-1]
        for k in range(len(terms) - 2, -1, -1):
            acc = add("OR2", terms[k], acc, f"c{i + 1}" if k == 0 else f"o{i}_{k}")
    for i in range(NB):
        xor(f"p{i}", f"c{i}", f"s{i}")
    return inst


CLA = build_cla()


def logic_sim(bits: dict) -> dict:
    """Logic value of every net for one input vector (bits: {input name: 0/1})."""
    val, pending = dict(bits), list(CLA)
    while pending:
        rest = []
        for g in pending:
            _, typ, a, b, out = g
            if a in val and b in val:
                val[out] = gl.LOGIC[typ](val[a], val[b])
            else:
                rest.append(g)
        assert len(rest) < len(pending), "combinational loop"
        pending = rest
    return val


def vectors():
    """All input vectors as dicts, in order of the integer a2a1a0 b2b1b0 c0."""
    out = []
    for k in range(2 ** len(INPUTS)):
        s = format(k, f"0{len(INPUTS)}b")
        a, b, c = s[:NB], s[NB:2 * NB], s[-1]
        bits = {f"a{i}": int(a[NB - 1 - i]) for i in range(NB)}
        bits.update({f"b{i}": int(b[NB - 1 - i]) for i in range(NB)})
        bits["c0"] = int(c)
        out.append((f"A={a} B={b} Cin={c}", bits))
    return out


# ----------------------------------------------------------------------------
# SPICE
# ----------------------------------------------------------------------------
def subckt(gate) -> str:
    g = gl.GATES[gate]
    txt = f".subckt {gate} {' '.join(g['inputs'])} out vdd\n"
    for dev in g["devices"]:
        txt += gl.device_line(*dev)
    return txt + f".ends {gate}\n"


def cla_netlist(c, bits=None, with_control=False) -> str:
    bits = bits or {k: 0 for k in INPUTS}
    used = sorted({g[1] for g in CLA})
    txt = (f"* {config.TECH} {NB}-bit carry-lookahead adder, corner {int(c['corner'])}\n"
           f"* VDD={c['vdd']:.6f} V, T={c['temp']:.4f} C; {len(CLA)} gates "
           f"({', '.join(f'{sum(g[1] == u for g in CLA)} {u}' for u in used)})\n"
           + gl.corner_header(c) + "\n" + "\n".join(subckt(u) for u in used) + "\n")
    for name, typ, a, b, out in CLA:
        txt += f"{name} {a} {b} {out} vdd {typ}\n"
    txt += f"\nVdd vdd 0 dc {c['vdd']:.9f}\n"
    for k in INPUTS:
        txt += f"V{k} {k} 0 dc {c['vdd'] * bits[k]:.9f}\n"
    if with_control:
        hi = "".join(f" - i(v{k})" for k in INPUTS if bits[k])
        txt += ("\n.control\nset numdgt=12\nop\n"
                "* leakage = current drawn from every source at VDD (supply + high inputs)\n"
                f"let ileak = -i(vdd){hi}\nprint {' '.join(f'v({o})' for o in OUTPUTS)} ileak\n.endc\n")
    return txt + ".end\n"


PRINT = ["i(vdd)"] + [f"i(v{k})" for k in INPUTS] + [f"v({o})" for o in OUTPUTS]
BATCH = 128


def run_spice(corners, vecs, exe, workers) -> pd.DataFrame:
    """One netlist per corner and input vector, each solved from scratch (op analysis)."""
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    su.write_param_model(WORK / gl.PARAM_LIB_NAME)
    names = []
    for ci in range(len(corners)):
        for vi, (_, bits) in enumerate(vecs):
            name = f"c{ci:03d}_v{vi:03d}.cir"
            (WORK / name).write_text(cla_netlist(corners.iloc[ci], bits))
            names.append(name)
    res = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(su.run_batch, exe, WORK, names[k:k + BATCH], PRINT, f"drv_{k:06d}.cir")
                for k in range(0, len(names), BATCH)]
        for f in futs:
            res.update(f.result())
    rows = []
    for ci in range(len(corners)):
        c = corners.iloc[ci]
        for vi, (label, bits) in enumerate(vecs):
            r = res.get(f"c{ci:03d}_v{vi:03d}.cir")
            rec = {"corner": ci, "vector": label, "vdd": c["vdd"], "temp": c["temp"], "spice_leak": np.nan}
            if r:
                rec["spice_leak"] = -r["i(vdd)"] - sum(r[f"i(v{k})"] for k in INPUTS if bits[k])
                logic = logic_sim(bits)
                rec["logic_ok"] = all((r[f"v({o})"] > 0.5 * c["vdd"]) == bool(logic[o]) for o in OUTPUTS)
            rows.append(rec)
    shutil.rmtree(WORK)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# ML: circuit = sum of gate models
# ----------------------------------------------------------------------------
def circuit_ml(corners, vecs):
    """(ML leakage, sum of standalone-gate SPICE leakages), each n_corners x n_vectors."""
    ml_tab, spice_tab = gl.gate_model_table(corners)
    ml = np.zeros((len(corners), len(vecs)))
    sp = np.zeros_like(ml)
    for vi, (_, bits) in enumerate(vecs):
        val = logic_sim(bits)
        for _, typ, a, b, _ in CLA:
            key = (typ, f"{val[a]}{val[b]}")
            ml[:, vi] += ml_tab[key]
            sp[:, vi] += spice_tab[key]
    return ml, sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    exe = su.find_ngspice()
    DATA.mkdir(parents=True, exist_ok=True)
    NETLIST_OUT.mkdir(parents=True, exist_ok=True)
    tables, plots = config.RESULTS_DIR / "tables", config.RESULTS_DIR / "plots"

    corners = pd.read_csv(gl.GATE_DATA / "corners.csv")
    assert np.allclose(corners["vdd"], gl.draw_corners()["vdd"]), "corners.csv is not the Part B corner set"
    vecs = vectors()

    su.write_param_model(NETLIST_OUT / gl.PARAM_LIB_NAME)
    ex_bits = vecs[0b1010111][1]          # A=101 B=011 Cin=1
    (NETLIST_OUT / "CLA3.cir").write_text(cla_netlist(corners.iloc[0], ex_bits, with_control=True))

    t0 = time.time()
    sim = run_spice(corners, vecs, exe, args.workers)
    n_fail = int(sim["spice_leak"].isna().sum())
    print(f"SPICE: {len(sim) - n_fail}/{len(sim)} operating points in {time.time() - t0:.0f} s, "
          f"logic outputs correct in {int(sim['logic_ok'].fillna(False).sum())}")
    ml, gate_spice = circuit_ml(corners, vecs)
    sim["ml_leak"] = ml.ravel()
    sim["gate_spice_sum"] = gate_spice.ravel()
    sim = sim.dropna(subset=["spice_leak", "ml_leak"]).reset_index(drop=True)
    sim["err_pct"] = 100 * (sim["ml_leak"] - sim["spice_leak"]).abs() / sim["spice_leak"]
    sim.drop(columns="logic_ok").to_csv(DATA / "cla3_simulations.csv", index=False, float_format="%.10e")

    overall = gl.leak_metrics(sim["spice_leak"], sim["ml_leak"])
    rows = [{"vector": "ALL", **overall}]
    for label, _ in vecs:
        g = sim[sim["vector"] == label]
        rows.append({"vector": label, **gl.leak_metrics(g["spice_leak"], g["ml_leak"])})
    table = pd.DataFrame(rows)
    table.to_csv(tables / "circuit_results_table.csv", index=False, float_format="%.6g")

    comp = gl.leak_metrics(sim["spice_leak"], sim["gate_spice_sum"])     # composition error alone
    gate_all = pd.read_csv(tables / "gate_results_table.csv").iloc[-1]
    per = table.iloc[1:].sort_values("median_err_pct")
    fmt = lambda r: (f"{r['vector']:24s} {r['spice_median_A']:12.4e} {r['ml_median_A']:12.4e} {r['bias_pct']:8.2f} "
                     f"{r['median_err_pct']:10.2f} {r['p95_err_pct']:10.2f} {r['mean_err_pct']:10.2f}")
    head = (f"{'Input vector':24s} {'SPICE med A':>12s} {'ML med A':>12s} {'Bias %':>8s} {'Med err %':>10s} "
            f"{'p95 err %':>10s} {'Mean err %':>10s}")
    L = [f"PART C: TEST-CIRCUIT VALIDATION, {NB}-BIT CARRY-LOOKAHEAD ADDER (ML vs SPICE)",
         f"{config.TECH}, {len(CLA)} gates / {sum(len(gl.GATES[g[1]]['devices']) for g in CLA)} transistors, "
         f"{len(vecs)} input vectors x {len(corners)} corners = {len(sim)} operating points"
         + (f" ({n_fail} SPICE failures dropped)" if n_fail else ""),
         "ML leakage = sum of the 42 Part B gate models (which call the Part A transistor models).",
         "", head, "-" * 92, fmt(table.iloc[0]), "",
         f"Overall mean accuracy = {100 - overall['mean_err_pct']:.2f} %",
         f"Gate level (Part B, all 18 cases): median {gate_all['median_err_pct']:.2f} %, "
         f"p95 {gate_all['p95_err_pct']:.2f} %, mean {gate_all['mean_err_pct']:.2f} %",
         f"Composition alone (sum of standalone-gate SPICE leakages vs circuit SPICE): bias {comp['bias_pct']:.3f} %, "
         f"median {comp['median_err_pct']:.3f} %, p95 {comp['p95_err_pct']:.3f} %",
         "", "Five most accurate input vectors:", head] + [fmt(r) for _, r in per.head(5).iterrows()] + [
         "", "Five least accurate input vectors:", head] + [fmt(r) for _, r in per.tail(5).iloc[::-1].iterrows()] + [
         "", "Ten largest single errors:"]
    for r in sim.sort_values("err_pct", ascending=False).head(10).itertuples(index=False):
        L.append(f"  {r.vector:24s} corner {r.corner:3d}  VDD {r.vdd:.3f} V  T {r.temp:6.1f} C  "
                 f"SPICE {r.spice_leak:.3e} A  ML {r.ml_leak:.3e} A  error {r.err_pct:.2f} %")
    L.append("\nAll 128 per-vector rows are in circuit_results_table.csv.")
    (tables / "circuit_results.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    gl.parity_plot(plots / "circuit_parity_plot.png",
                   [(f"{NB}-bit CLA", sim["spice_leak"].to_numpy(), sim["ml_leak"].to_numpy())],
                   f"{NB}-bit CLA leakage: ML vs SPICE ({len(sim)} points, 128 vectors x {len(corners)} corners)")


if __name__ == "__main__":
    main()
