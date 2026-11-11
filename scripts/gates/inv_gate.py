"""
Part B, gate 1 of 5: INV. Simplest gate -- 2 transistors, no internal node.

  Mp: drain=y gate=a source=vdd body=vdd   (PMOS pull-up)
  Mn: drain=y gate=a source=0   body=0     (NMOS pull-down)

For a given input A (exactly 0 or VDD) and PVT corner, SPICE's DC operating
point gives the gate's total static leakage directly as the current the VDD
source delivers. The ML side predicts each transistor's own terminal
currents from that SAME operating point's node voltages (we do not
re-derive them -- this is what the spec calls for) and sums what leaves VDD.
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import gate_common as gc

NETLIST = """INV gate - corner test
{model_text}

Mp y a vdd vdd pmos w={wp} l=32n
Mn y a 0   0   nmos w={wn} l=32n

Vdd vdd 0 DC {vdd}
Va  a   0 DC {a_val}

.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9
.nodeset v(y)={y_guess}

.control
op
wrdata __OUT__ v(y) i(vdd) i(va)
.endc
.end
"""


def run_one(corner, a_val, workdir):
    vdd = corner["vdd"]
    y_guess = vdd if a_val == 0.0 else 0.0   # INV: y = NOT(a)
    text = NETLIST.format(
        model_text=gc.patched_model_text(corner["nmos"], corner["pmos"]),
        wp=corner["w_p"], wn=corner["w_n"], vdd=vdd, a_val=a_val, y_guess=y_guess,
    )
    df = gc.run_netlist(text, workdir, f"inv_a{a_val}")
    # wrdata columns: idx, v(y), idx, i(vdd), idx, i(va)  (ngspice repeats the sweep index)
    v_y, i_vdd, i_va = df.iloc[0, 1], df.iloc[0, 3], df.iloc[0, 5]

    vdd = corner["vdd"]
    # Mp: drain=y gate=a source=vdd body=vdd
    pmos_pred = gc.predict_currents("PMOS_ON" if a_val == 0 else "PMOS_OFF",
                                     vd=v_y, vg=a_val, vs=vdd, vb=vdd,
                                     vdd=vdd, temp=corner["temp"], w=corner["w_p"],
                                     proc=corner["pmos"])
    # Mn: drain=y gate=a source=0 body=0
    nmos_pred = gc.predict_currents("NMOS_ON" if a_val == vdd else "NMOS_OFF",
                                     vd=v_y, vg=a_val, vs=0.0, vb=0.0,
                                     vdd=vdd, temp=corner["temp"], w=corner["w_n"],
                                     proc=corner["nmos"])

    # Canonical ML gate-leakage metric, used for every gate: current drawn
    # from VDD = sum of predicted currents into whichever terminals are
    # actually tied to the VDD rail (here, Mp's source and body). This is
    # directly comparable to SPICE's own -i(Vdd). We deliberately do NOT
    # also try to reconstruct it from the NMOS side and reconcile the two:
    # the two per-transistor models were trained independently and are not
    # forced to agree on a shared node's current the way a real SPICE
    # solve is -- checking that disagreement is itself a finding (see
    # notes), not something to paper over by picking whichever is closer.
    ml_leak = pmos_pred["is"] + pmos_pred["ib"]
    spice_leak = -i_vdd

    return dict(v_y=v_y, i_vdd=i_vdd, i_va=i_va, spice_leak=spice_leak,
                pmos_pred=pmos_pred, nmos_pred=nmos_pred, ml_leak=ml_leak)


if __name__ == "__main__":
    import sys
    n_corners = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    rows = []
    rng = np.random.default_rng(0)
    workdir = Path(tempfile.mkdtemp(prefix="inv_"))
    n_failed = 0
    try:
        for i in range(n_corners):
            corner = gc.sample_corner(rng)
            for a_label, a_val in (("0", 0.0), ("1", corner["vdd"])):
                try:
                    r = run_one(corner, a_val, workdir)
                except Exception as e:
                    n_failed += 1
                    print(f"  [skip] corner {i} A={a_label}: {type(e).__name__}: {e}", file=sys.stderr)
                    continue
                err_pct = 100 * abs(r["ml_leak"] - r["spice_leak"]) / max(abs(r["spice_leak"]), 1e-14)
                rows.append(dict(corner=i, A=a_label, vdd=corner["vdd"], temp=corner["temp"],
                                  spice_leak=r["spice_leak"], ml_leak=r["ml_leak"], err_pct=err_pct))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if n_failed:
        print(f"\n{n_failed} corner/input combos failed to converge and were skipped", file=sys.stderr)

    df = pd.DataFrame(rows)
    df.insert(0, "gate", "INV")
    df.to_csv("inv_200_raw.csv", index=False)
    pd.set_option("display.float_format", lambda x: f"{x:.4e}")
    print(df.to_string(index=False))
    print(f"\nmedian err%={df['err_pct'].median():.2f}  p95 err%={df['err_pct'].quantile(0.95):.2f}")
