"""
Part B, gate 4 of 5: AND2 = NAND2 + INV, simulated as ONE complete 6-transistor
circuit (per spec -- never stitched from separate NAND2 and INV runs).
Internal nodes: x (inside the NAND2 stack) and nq (the NAND2 output, which
is the inverter's gate input -- an analog node, not a true rail, but it
settles very close to 0 or VDD just like any other gate output).

  Mn1: drain=x  gate=a  source=0   body=0      ) NAND2
  Mn2: drain=nq gate=b  source=x   body=0      )
  Mp1: drain=nq gate=a  source=vdd body=vdd    )
  Mp2: drain=nq gate=b  source=vdd body=vdd    )
  Mp3: drain=y  gate=nq source=vdd body=vdd    ) inverter stage
  Mn3: drain=y  gate=nq source=0   body=0      )

VDD-tied sources: Mp1, Mp2, Mp3 -- the ML leakage metric sums is+ib over
all three.
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import gate_common as gc

NETLIST = """AND2 gate - corner test
{model_text}

Mn1 x  a  0   0   nmos w={wn} l=32n
Mn2 nq b  x   0   nmos w={wn} l=32n
Mp1 nq a  vdd vdd pmos w={wp} l=32n
Mp2 nq b  vdd vdd pmos w={wp} l=32n
Mp3 y  nq vdd vdd pmos w={wp} l=32n
Mn3 y  nq 0   0   nmos w={wn} l=32n

Vdd vdd 0 DC {vdd}
Va  a   0 DC {a_val}
Vb  b   0 DC {b_val}

.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9
.nodeset v(x)=0 v(nq)={nq_guess} v(y)={y_guess}

.control
op
wrdata __OUT__ v(x) v(nq) v(y) i(vdd)
.endc
.end
"""


def run_one(corner, a_val, b_val, workdir):
    vdd = corner["vdd"]
    both_high = (a_val == vdd and b_val == vdd)
    nq_guess = 0.0 if both_high else vdd   # NAND2 stage output
    y_guess = vdd if both_high else 0.0    # AND2: y = a AND b
    text = NETLIST.format(
        model_text=gc.patched_model_text(corner["nmos"], corner["pmos"]),
        wp=corner["w_p"], wn=corner["w_n"], vdd=vdd,
        a_val=a_val, b_val=b_val, nq_guess=nq_guess, y_guess=y_guess,
    )
    df = gc.run_netlist(text, workdir, f"and2_{a_val}_{b_val}")
    v_x, v_nq, v_y, i_vdd = df.iloc[0, 1], df.iloc[0, 3], df.iloc[0, 5], df.iloc[0, 7]

    vdd, temp = corner["vdd"], corner["temp"]

    def nmos_case(vg):
        return "NMOS_ON" if vg > vdd / 2 else "NMOS_OFF"

    def pmos_case(vg):
        return "PMOS_ON" if vg < vdd / 2 else "PMOS_OFF"

    gc.predict_currents(nmos_case(a_val), vd=v_x, vg=a_val, vs=0.0, vb=0.0,
                         vdd=vdd, temp=temp, w=corner["w_n"], proc=corner["nmos"])
    gc.predict_currents(nmos_case(b_val), vd=v_nq, vg=b_val, vs=v_x, vb=0.0,
                         vdd=vdd, temp=temp, w=corner["w_n"], proc=corner["nmos"])
    mp1 = gc.predict_currents(pmos_case(a_val), vd=v_nq, vg=a_val, vs=vdd, vb=vdd,
                               vdd=vdd, temp=temp, w=corner["w_p"], proc=corner["pmos"])
    mp2 = gc.predict_currents(pmos_case(b_val), vd=v_nq, vg=b_val, vs=vdd, vb=vdd,
                               vdd=vdd, temp=temp, w=corner["w_p"], proc=corner["pmos"])
    mp3 = gc.predict_currents(pmos_case(v_nq), vd=v_y, vg=v_nq, vs=vdd, vb=vdd,
                               vdd=vdd, temp=temp, w=corner["w_p"], proc=corner["pmos"])
    gc.predict_currents(nmos_case(v_nq), vd=v_y, vg=v_nq, vs=0.0, vb=0.0,
                         vdd=vdd, temp=temp, w=corner["w_n"], proc=corner["nmos"])

    ml_leak = sum(p["is"] + p["ib"] for p in (mp1, mp2, mp3))
    spice_leak = -i_vdd
    return dict(v_x=v_x, v_nq=v_nq, v_y=v_y, spice_leak=spice_leak, ml_leak=ml_leak)


if __name__ == "__main__":
    import sys
    n_corners = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    rows = []
    rng = np.random.default_rng(0)
    workdir = Path(tempfile.mkdtemp(prefix="and2_"))
    n_failed = 0
    try:
        for i in range(n_corners):
            corner = gc.sample_corner(rng)
            vdd = corner["vdd"]
            for a_label, b_label, a_val, b_val in (
                ("0", "0", 0.0, 0.0), ("0", "1", 0.0, vdd),
                ("1", "0", vdd, 0.0), ("1", "1", vdd, vdd),
            ):
                try:
                    r = run_one(corner, a_val, b_val, workdir)
                except Exception as e:
                    n_failed += 1
                    print(f"  [skip] corner {i} AB={a_label}{b_label}: {type(e).__name__}: {e}", file=sys.stderr)
                    continue
                err_pct = 100 * abs(r["ml_leak"] - r["spice_leak"]) / max(abs(r["spice_leak"]), 1e-14)
                rows.append(dict(corner=i, AB=a_label + b_label, vdd=vdd, temp=corner["temp"],
                                  spice_leak=r["spice_leak"], ml_leak=r["ml_leak"], err_pct=err_pct))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if n_failed:
        print(f"{n_failed} corner/input combos failed to converge and were skipped", file=sys.stderr)

    df = pd.DataFrame(rows)
    df.insert(0, "gate", "AND2")
    df.to_csv("and2_200_raw.csv", index=False)
    pd.set_option("display.float_format", lambda x: f"{x:.4e}")
    print(df.to_string(index=False))
    print(f"\nmedian err%={df['err_pct'].median():.2f}  p95 err%={df['err_pct'].quantile(0.95):.2f}")
