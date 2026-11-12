"""
Part B, gate 3 of 5: NOR2. 4 transistors, one internal node y (between the
series PMOS).

  Mp1 (top, at vdd):    drain=y  gate=a source=vdd body=vdd
  Mp2 (bottom, at out):  drain=out gate=b source=y   body=vdd
  Mn1:                   drain=out gate=a source=0   body=0
  Mn2:                   drain=out gate=b source=0   body=0

Only Mp1's source is the true VDD rail (Mp2's source is the internal node),
so the ML leakage metric here is just Mp1's is+ib.
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import gate_common as gc

NETLIST = """NOR2 gate - corner test
{model_text}

Mp1 y   a vdd vdd pmos w={wp} l=32n
Mp2 out b y   vdd pmos w={wp} l=32n
Mn1 out a 0   0   nmos w={wn} l=32n
Mn2 out b 0   0   nmos w={wn} l=32n

Vdd vdd 0 DC {vdd}
Va  a   0 DC {a_val}
Vb  b   0 DC {b_val}

.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9
.nodeset v(y)={vdd} v(out)={out_guess}

.control
op
wrdata __OUT__ v(y) v(out) i(vdd)
.endc
.end
"""


def run_one(corner, a_val, b_val, workdir):
    vdd = corner["vdd"]
    out_guess = 0.0 if (a_val == vdd or b_val == vdd) else vdd   # NOR2: out = NOT(a OR b)
    text = NETLIST.format(
        model_text=gc.patched_model_text(corner["nmos"], corner["pmos"]),
        wp=corner["w_p"], wn=corner["w_n"], vdd=vdd,
        a_val=a_val, b_val=b_val, out_guess=out_guess,
    )
    df = gc.run_netlist(text, workdir, f"nor2_{a_val}_{b_val}")
    v_y, v_out, i_vdd = df.iloc[0, 1], df.iloc[0, 3], df.iloc[0, 5]

    vdd, temp = corner["vdd"], corner["temp"]

    def nmos_case(vg):
        return "NMOS_ON" if vg > vdd / 2 else "NMOS_OFF"

    def pmos_case(vg):
        return "PMOS_ON" if vg < vdd / 2 else "PMOS_OFF"

    mp1 = gc.predict_currents(pmos_case(a_val), vd=v_y, vg=a_val, vs=vdd, vb=vdd,
                               vdd=vdd, temp=temp, w=corner["w_p"], proc=corner["pmos"])
    # Mp2, Mn1, Mn2 predicted for completeness (internal-node bookkeeping),
    # not part of the VDD-current sum
    gc.predict_currents(pmos_case(b_val), vd=v_out, vg=b_val, vs=v_y, vb=vdd,
                         vdd=vdd, temp=temp, w=corner["w_p"], proc=corner["pmos"])
    gc.predict_currents(nmos_case(a_val), vd=v_out, vg=a_val, vs=0.0, vb=0.0,
                         vdd=vdd, temp=temp, w=corner["w_n"], proc=corner["nmos"])
    gc.predict_currents(nmos_case(b_val), vd=v_out, vg=b_val, vs=0.0, vb=0.0,
                         vdd=vdd, temp=temp, w=corner["w_n"], proc=corner["nmos"])

    ml_leak = mp1["is"] + mp1["ib"]
    spice_leak = -i_vdd
    return dict(v_y=v_y, v_out=v_out, spice_leak=spice_leak, ml_leak=ml_leak)


if __name__ == "__main__":
    import sys
    n_corners = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    rows = []
    rng = np.random.default_rng(0)
    workdir = Path(tempfile.mkdtemp(prefix="nor2_"))
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
    df.insert(0, "gate", "NOR2")
    df.to_csv("nor2_200_raw.csv", index=False)
    pd.set_option("display.float_format", lambda x: f"{x:.4e}")
    print(df.to_string(index=False))
    print(f"\nmedian err%={df['err_pct'].median():.2f}  p95 err%={df['err_pct'].quantile(0.95):.2f}")
