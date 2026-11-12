"""
Gate library for Parts B and C: gate definitions, SPICE netlists, corners and
the ML gate model (which calls the four transistor models of Part A).

Gates (static CMOS). Node names: vdd, 0, inputs a/b, output out, internal
nodes x (between the series NMOS of a NAND2), y (between the series PMOS of a
NOR2) and n (the NAND2/NOR2 output inside AND2/OR2).

    INV    mp1: out a vdd vdd      mn1: out a 0 0
    NAND2  mp1: out a vdd vdd      mp2: out b vdd vdd
           mn1: out a x 0 (top)    mn2: x b 0 0 (bottom)
    NOR2   mp1: y a vdd vdd (top)  mp2: out b y vdd (bottom)
           mn1: out a 0 0          mn2: out b 0 0
    AND2 = NAND2 (output n) + INV,  OR2 = NOR2 (output n) + INV

Leakage of a gate = total current drawn from every source at VDD (the supply
and the inputs that are high) = static power / VDD. With ideal sources and
Kirchhoff's law this equals

    sum over transistors of (Vd*Id + Vg*Ig + Vs*Is + Vb*Ib) / VDD

(currents into the device terminals), which is what the ML gate model
evaluates with the predicted terminal currents. Counting the high inputs makes
gate leakages add up correctly in a larger circuit, where a gate's input
current is supplied by the gate driving it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "ml"))
import config                      # noqa: E402
import spice_utils as su           # noqa: E402
import common                      # noqa: E402

PARAM_LIB_NAME = f"{config.TECH}_param.lib"
GATE_DATA = config.DATA_DIR / "gates"


# ----------------------------------------------------------------------------
# Gate definitions
# ----------------------------------------------------------------------------
def _inv(a, out, tag=""):
    wn, wp = config.GATE_W["INV"]
    return [(f"mp{tag}1", "pmos", out, a, "vdd", "vdd", wp),
            (f"mn{tag}1", "nmos", out, a, "0", "0", wn)]


def _nand2(out):
    wn, wp = config.GATE_W["NAND2"]
    return [("mp1", "pmos", out, "a", "vdd", "vdd", wp),
            ("mp2", "pmos", out, "b", "vdd", "vdd", wp),
            ("mn1", "nmos", out, "a", "x", "0", wn),
            ("mn2", "nmos", "x", "b", "0", "0", wn)]


def _nor2(out):
    wn, wp = config.GATE_W["NOR2"]
    return [("mp1", "pmos", "y", "a", "vdd", "vdd", wp),
            ("mp2", "pmos", out, "b", "y", "vdd", wp),
            ("mn1", "nmos", out, "a", "0", "0", wn),
            ("mn2", "nmos", out, "b", "0", "0", wn)]


# devices: (name, type, drain, gate, source, body, width in Lmin)
GATES = {
    "INV":   {"inputs": ["a"],      "internal": [],         "devices": _inv("a", "out")},
    "NAND2": {"inputs": ["a", "b"], "internal": ["x"],      "devices": _nand2("out")},
    "NOR2":  {"inputs": ["a", "b"], "internal": ["y"],      "devices": _nor2("out")},
    "AND2":  {"inputs": ["a", "b"], "internal": ["x", "n"], "devices": _nand2("n") + _inv("n", "out", "i")},
    "OR2":   {"inputs": ["a", "b"], "internal": ["y", "n"], "devices": _nor2("n") + _inv("n", "out", "i")},
}
LOGIC = {
    "INV": lambda a: 1 - a, "NAND2": lambda a, b: 1 - (a & b), "NOR2": lambda a, b: 1 - (a | b),
    "AND2": lambda a, b: a & b, "OR2": lambda a, b: a | b,
}


def input_combos(gate):
    """All input combinations as strings: '0','1' or '00','01','10','11' (order AB)."""
    n = len(GATES[gate]["inputs"])
    return [format(i, f"0{n}b") for i in range(2 ** n)]


CASES18 = [(g, c) for g in GATES for c in input_combos(g)]


# ----------------------------------------------------------------------------
# Corners
# ----------------------------------------------------------------------------
PROC_COLS = [f"{d}_{p}" for d in ("n", "p") for p in config.PROC_PARAMS]


def draw_corners(n=config.N_CORNERS, seed=config.SEED_CORNERS) -> pd.DataFrame:
    """One corner = VDD, temperature, 5 NMOS and 5 PMOS process parameters (Section 4)."""
    rng = np.random.default_rng(seed)
    nominal = su.read_nominal_params()
    df = pd.DataFrame({
        "corner": np.arange(n),
        "vdd": rng.uniform(config.VDD_NOM * (1 - config.VDD_TOL), config.VDD_NOM * (1 + config.VDD_TOL), n),
        "temp": rng.uniform(*config.TEMP_RANGE, n)})
    for d, dev in (("n", "nmos"), ("p", "pmos")):
        for p in config.PROC_PARAMS:
            df[f"{d}_{p}"] = nominal[dev][p] * rng.uniform(1 - config.PROC_TOL, 1 + config.PROC_TOL, n)
    return df


def corner_header(c) -> str:
    """.param / .include / .options / .temp lines of one corner (c: row of the corners table)."""
    nm = {p: c[f"n_{p}"] for p in config.PROC_PARAMS}
    pm = {p: c[f"p_{p}"] for p in config.PROC_PARAMS}
    return (su.param_lines(nm, pm) + f'.include "{PARAM_LIB_NAME}"\n'
            + config.SPICE_OPTIONS + "\n" + f".temp {c['temp']:.6f}\n")


def device_line(name, typ, d, g, s, b, wmult) -> str:
    return f"{name} {d} {g} {s} {b} {typ} W={wmult * config.L_MIN:.4e} L={config.L_MIN:.3e}\n"


def gate_netlist(gate, combo, c, with_control=False, guess_mode=0) -> str:
    """guess_mode picks the solver's starting guess: 0 = logic levels on driven nodes,
    1 = none, 2 = logic levels plus stack nodes near their rail (used for retries)."""
    g = GATES[gate]
    txt = (f"* {config.TECH} {gate}, inputs {''.join(g['inputs']).upper()}={combo}, corner {int(c['corner'])}\n"
           f"* VDD={c['vdd']:.6f} V, T={c['temp']:.4f} C\n" + corner_header(c) + "\n")
    for dev in g["devices"]:
        txt += device_line(*dev)
    txt += f"Vdd vdd 0 dc {c['vdd']:.9f}\n"
    for name, bit in zip(g["inputs"], combo):
        txt += f"V{name} {name} 0 dc {c['vdd'] * int(bit):.9f}\n"
    # starting guess for the solver: ideal logic levels on the driven nodes
    bits = [int(ch) for ch in combo]
    guess = {"out": LOGIC[gate](*bits)}
    if "n" in g["internal"]:
        guess["n"] = 1 - guess["out"]
    if guess_mode == 2:
        guess.update({k: v for k, v in (("x", 0.05), ("y", 0.95)) if k in g["internal"]})
    if guess_mode != 1:
        txt += ".nodeset " + " ".join(f"v({k})={c['vdd'] * v:.6f}" for k, v in guess.items()) + "\n"
    if with_control:
        hi = "".join(f" - i(v{n})" for n, bit in zip(g["inputs"], combo) if bit == "1")
        nodes = " ".join(f"v({n})" for n in g["internal"] + ["out"])
        txt += ("\n.control\nset numdgt=12\nop\n"
                "* leakage = current drawn from every source at VDD (supply + high inputs)\n"
                f"let ileak = -i(vdd){hi}\nprint {nodes} ileak\n.endc\n")
    return txt + ".end\n"


def gate_print_vectors(gate):
    g = GATES[gate]
    return [f"v({n})" for n in g["internal"] + ["out"]] + ["i(vdd)"] + [f"i(v{n})" for n in g["inputs"]]


# ----------------------------------------------------------------------------
# ML: transistor models -> gate model
# ----------------------------------------------------------------------------
_models = {}


def _model(case):
    if case not in _models:
        _models[case] = common.load_model(case)
    return _models[case]


def predict_devices(typ, wmult, vd, vg, vs, vb, corners: pd.DataFrame) -> np.ndarray:
    """
    Terminal currents (n x 4: id, ig, is, ib, into the device) of one transistor
    at n operating points. The ON or OFF model is chosen from the gate voltage.
    """
    d = "n" if typ == "nmos" else "p"
    df = pd.DataFrame({"vdd": corners["vdd"].to_numpy(), "temp": corners["temp"].to_numpy(),
                       "w": wmult * config.L_MIN, "vd": vd, "vs": vs, "vg": vg, "vb": vb,
                       **{p: corners[f"{d}_{p}"].to_numpy() for p in config.PROC_PARAMS}})
    high = df["vg"].to_numpy() > 0.5 * df["vdd"].to_numpy()
    on = high if typ == "nmos" else ~high
    out = np.zeros((len(df), 4))
    for mask, state in ((on, "ON"), (~on, "OFF")):
        if mask.any():
            out[mask] = _model(f"{typ.upper()}_{state}").predict(df[mask])[["id", "ig", "is", "ib"]].to_numpy()
    return out


def _node_voltages(gate, combos, corners, nodes):
    """{node name: array} for every node of the gate. `nodes` holds internal nodes and out."""
    vdd = corners["vdd"].to_numpy()
    v = {"vdd": vdd, "0": np.zeros(len(vdd))}
    for k, name in enumerate(GATES[gate]["inputs"]):
        v[name] = vdd * np.array([int(c[k]) for c in combos])
    v.update({k: np.asarray(val, dtype=float) for k, val in nodes.items()})
    return v


def gate_leakage_ml(gate, combos, corners, nodes) -> np.ndarray:
    """
    ML gate model: leakage of `gate` for each row (input combination `combos[i]`
    at corner row `corners.iloc[i]`), given the node voltages in `nodes`
    (internal nodes and 'out').
    """
    v = _node_voltages(gate, combos, corners, nodes)
    power = np.zeros(len(corners))
    for _, typ, d, g, s, b, wm in GATES[gate]["devices"]:
        i = predict_devices(typ, wm, v[d], v[g], v[s], v[b], corners)
        power += v[d] * i[:, 0] + v[g] * i[:, 1] + v[s] * i[:, 2] + v[b] * i[:, 3]
    return power / corners["vdd"].to_numpy()


def solve_internal_nodes(gate, combos, corners, v_out, sweeps=2, iters=45) -> dict:
    """
    ML estimate of the internal node voltages: for each internal node, find by
    bisection the voltage at which the predicted currents into the transistors
    connected to it sum to zero (Kirchhoff's current law). The output node is
    kept at `v_out`. Nodes are solved one after another, `sweeps` times.
    """
    g = GATES[gate]
    vdd = corners["vdd"].to_numpy()
    bits = np.array([[int(ch) for ch in c] for c in combos])
    start = {"x": np.zeros(len(vdd)), "y": vdd.copy()}
    if "n" in g["internal"]:
        nand = gate == "AND2"
        hi = (1 - (bits[:, 0] & bits[:, 1])) if nand else (1 - (bits[:, 0] | bits[:, 1]))
        start["n"] = vdd * hi
    nodes = {k: start[k] for k in g["internal"]}
    nodes["out"] = np.asarray(v_out, dtype=float)

    def residual(node, val):
        trial = dict(nodes, **{node: val})
        v = _node_voltages(gate, combos, corners, trial)
        r = np.zeros(len(vdd))
        for _, typ, d, gt, s, b, wm in g["devices"]:
            terms = (d, gt, s, b)
            if node in terms:
                i = predict_devices(typ, wm, v[d], v[gt], v[s], v[b], corners)
                for k, t in enumerate(terms):
                    if t == node:
                        r += i[:, k]
        return r        # current leaving the node into the devices; rises with the node voltage

    for _ in range(sweeps):
        for node in g["internal"]:
            lo, hi = np.zeros(len(vdd)), vdd.copy()
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                pos = residual(node, mid) > 0
                hi, lo = np.where(pos, mid, hi), np.where(pos, lo, mid)
            nodes[node] = 0.5 * (lo + hi)
    return {k: nodes[k] for k in g["internal"]}


def gate_model_table(corners: pd.DataFrame) -> dict:
    """
    The gate models evaluated at every corner: {(gate, input combination): array of
    ML leakage per corner}. Uses the node voltages recorded in Part B
    (data/gates/gate_simulations.csv); corners whose gate simulation is missing give NaN.
    Also returns the SPICE gate leakages in the same layout.
    """
    sim = pd.read_csv(GATE_DATA / "gate_simulations.csv", dtype={"input": str})
    ml, spice = {}, {}
    for (gate, combo), grp in sim.groupby(["gate", "input"], sort=False):
        cr = corners.iloc[grp["corner"].to_numpy()].reset_index(drop=True)
        nodes = {n: grp[f"v_{n}_spice"].to_numpy() for n in GATES[gate]["internal"]}
        nodes["out"] = grp["v_out_spice"].to_numpy()
        for store, vals in ((ml, gate_leakage_ml(gate, [combo] * len(grp), cr, nodes)),
                            (spice, grp["spice_leak"].to_numpy())):
            full = np.full(len(corners), np.nan)
            full[grp["corner"].to_numpy()] = vals
            store[(gate, combo)] = full
    return ml, spice


# ----------------------------------------------------------------------------
# Result metrics (deliverables 6 and 8)
# ----------------------------------------------------------------------------
def leak_metrics(spice, ml) -> dict:
    spice, ml = np.asarray(spice), np.asarray(ml)
    rel = (ml - spice) / np.abs(spice)
    return {"n": len(spice), "spice_median_A": np.median(spice), "ml_median_A": np.median(ml),
            "bias_pct": 100 * np.median(rel), "median_err_pct": 100 * np.median(np.abs(rel)),
            "p95_err_pct": 100 * np.percentile(np.abs(rel), 95),
            "mean_err_pct": 100 * np.mean(np.abs(rel))}


def parity_plot(path, groups, title):
    """groups: list of (label, spice array, ml array)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 6))
    allv = np.concatenate([np.r_[s, m] for _, s, m in groups])
    lo, hi = allv[allv > 0].min() * 0.7, allv.max() * 1.4
    for label, s, m in groups:
        ax.scatter(s, m, s=5, alpha=0.45, label=label, linewidths=0)
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="ML = SPICE")
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi),
           xlabel="SPICE leakage (A)", ylabel="ML leakage (A)")
    ax.set_title(title, fontsize=9)
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
