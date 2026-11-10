"""
Shared configuration for the DVD ML-driven PVT-aware leakage estimation project.

Team assignment:  32nm LP model file, nominal VDD = 1.0 V, test circuit = 3-bit CLA.
Every script imports its paths, ranges and seeds from here so the whole flow
can be regenerated from the model file alone.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "models" / "32nm_LP.pm"

NETLIST_DIR = ROOT / "netlists"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
ML_MODEL_DIR = ROOT / "ml_models"
TOOLS_DIR = ROOT / "tools"

# ----------------------------------------------------------------------------
# Assignment
# ----------------------------------------------------------------------------
TECH = "32nm_LP"
VDD_NOM = 1.0             # V
L_MIN = 32e-9             # m, channel length = feature size
W_MIN_MULT, W_MAX_MULT = 1, 8   # W = 1..8 x Lmin

# ----------------------------------------------------------------------------
# Sampling ranges (Section 4 of the project spec)
# ----------------------------------------------------------------------------
VDD_TOL = 0.10            # VDD = VDD_NOM +/- 10 %
TEMP_RANGE = (-40.0, 125.0)   # degC
PROC_TOL = 0.10           # process params = nominal +/- 10 %
PROC_PARAMS = ["vth0", "toxe", "ndep", "u0", "xj"]

# Nominal process parameters read from models/32nm_LP.pm
# (also re-parsed at run time from the model file; these are a cross-check)
NOMINAL = {
    "nmos": {"vth0": 0.63,    "toxe": 1.6e-9,  "ndep": 4.12e18, "u0": 0.042, "xj": 5e-8},
    "pmos": {"vth0": -0.5808, "toxe": 1.62e-9, "ndep": 3.07e18, "u0": 0.016, "xj": 1e-8},
}

# ----------------------------------------------------------------------------
# Simulator options required in every netlist
# ----------------------------------------------------------------------------
SPICE_OPTIONS = ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9"

# ----------------------------------------------------------------------------
# Part A cases: gate / body voltage fixed per case ("VDD" means the sampled VDD)
# ----------------------------------------------------------------------------
CASES = {
    "NMOS_ON":  {"type": "nmos", "vg": "VDD", "vb": "0"},
    "NMOS_OFF": {"type": "nmos", "vg": "0",   "vb": "0"},
    "PMOS_ON":  {"type": "pmos", "vg": "0",   "vb": "VDD"},
    "PMOS_OFF": {"type": "pmos", "vg": "VDD", "vb": "VDD"},
}
N_SAMPLES = 10_000

# ----------------------------------------------------------------------------
# Random seeds (fixed for reproducibility)
# ----------------------------------------------------------------------------
SEED_TRANSISTOR_DATA = {"NMOS_ON": 101, "NMOS_OFF": 202, "PMOS_ON": 303, "PMOS_OFF": 404}
SEED_SPLIT = 42
SEED_CORNERS = 2025

# CSV column order required by the deliverables
CSV_COLUMNS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb",
               "vth0", "toxe", "ndep", "u0", "xj", "id", "ig", "is", "ib"]
