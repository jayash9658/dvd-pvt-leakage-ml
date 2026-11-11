"""
Shared infrastructure for Part B (gate-level validation): PVT-corner sampling,
per-corner model-card patching, running ngspice, and the SPICE-vs-ML leakage
comparison. One netlist builder per gate (inv_gate.py, nand2_gate.py, ...)
lives in its own small script; everything generic lives here.

Methodology (matches the spec):
  - A "corner" = one random draw of VDD, temperature, and process parameters,
    using the Section 4 distributions. All NMOS devices in the gate share one
    process draw, all PMOS devices share another (separate from NMOS's).
  - Ground truth = the DC operating-point current drawn from VDD (the gate's
    total static leakage), read straight from ngspice.
  - ML estimate = for every transistor in the gate, feed its actual terminal
    voltages (from this same SPICE run, per the spec -- we don't re-derive
    them analytically) into the matching trained case model (common.py's
    TransistorModel, loaded via common.load_model), and sum the predicted
    drain currents of the transistors tied to VDD (equivalent to the total
    current the supply delivers).
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          # scripts/gates
SCRIPTS = HERE.parent                            # scripts
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "ml"))
import config                      # noqa: E402
import common as jayash_common     # noqa: E402

os.environ.setdefault("NGSPICE", r"C:\ngspice\Spice64\bin\ngspice_con.exe")
import spice_utils as su           # noqa: E402
NGSPICE = su.find_ngspice()

L_MIN = config.L_MIN
_BASE_MODEL_TEXT = config.MODEL_FILE.read_text()

NOMINAL = config.NOMINAL

_model_cache = {}


def _patch_block(block_text, params):
    for key, val in params.items():
        block_text, n = re.subn(rf"\b{key}(\s*=\s*)[-+0-9.eE]+", rf"{key}\g<1>{val:.6e}",
                                 block_text, count=1)
        assert n == 1, f"could not find/patch {key} in model block"
    return block_text


def patched_model_text(nmos_params: dict, pmos_params: dict) -> str:
    """Return the full .model text with vth0/toxe/ndep/u0/xj overridden per corner."""
    nmos_start = _BASE_MODEL_TEXT.index(".model  nmos")
    pmos_start = _BASE_MODEL_TEXT.index(".model  pmos")
    head = _BASE_MODEL_TEXT[:nmos_start]
    nmos_block = _BASE_MODEL_TEXT[nmos_start:pmos_start]
    pmos_block = _BASE_MODEL_TEXT[pmos_start:]
    return head + _patch_block(nmos_block, nmos_params) + _patch_block(pmos_block, pmos_params)


def sample_corner(rng: np.random.Generator, vdd_nominal: float = None) -> dict:
    """One PVT corner per the spec's Section 4 distributions."""
    vdd_nominal = config.VDD_NOM if vdd_nominal is None else vdd_nominal
    vdd = rng.uniform(0.9, 1.1) * vdd_nominal
    temp = rng.uniform(-40, 125)

    def proc(dev):
        nom = NOMINAL[dev]
        return {k: rng.uniform(0.9, 1.1) * v for k, v in nom.items()}

    return dict(
        vdd=vdd, temp=temp,
        w_n=rng.uniform(1, 8) * L_MIN, w_p=rng.uniform(1, 8) * L_MIN,
        nmos=proc("nmos"), pmos=proc("pmos"),
    )


def run_netlist(netlist_text: str, workdir: Path, name: str) -> pd.DataFrame:
    """Writes, runs (ngspice_con -b), and parses one netlist's wrdata output."""
    cir = workdir / f"{name}.cir"
    out = workdir / f"{name}.txt"
    cir.write_text(netlist_text.replace("__OUT__", out.name))
    result = subprocess.run([NGSPICE, "-b", cir.name], cwd=workdir,
                             capture_output=True, text=True, timeout=30)
    if not out.exists():
        raise RuntimeError(f"ngspice produced no output for {name}:\n{result.stdout}\n{result.stderr}")
    df = pd.read_csv(out, sep=r"\s+", header=None)
    return df


def predict_currents(case: str, vd, vg, vs, vb, vdd, temp, w, proc: dict):
    """One transistor's (id, ig, is, ib) from the matching trained case model."""
    model = jayash_common.load_model(case)
    row = pd.DataFrame([{
        "vdd": vdd, "temp": temp, "w": w, "vd": vd, "vs": vs, "vg": vg, "vb": vb,
        **proc,
    }])
    pred = model.predict(row)
    return {k: float(pred[k].iloc[0]) for k in ("id", "ig", "is", "ib")}
