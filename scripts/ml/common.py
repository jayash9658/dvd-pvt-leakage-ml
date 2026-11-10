"""
Shared code for the Part A transistor models (step 2: train models).

Everything a model type needs except the regressor itself lives here:
  * loading a case CSV and the fixed 70/15/15 split (seed config.SEED_SPLIT)
  * source/drain orientation for the channel features: the channel is
    symmetric, so the physics features are computed with the "high" terminal
    (higher polarity-corrected potential) acting as drain. The targets keep
    the real drain/source labels, because the junction currents of this model
    card are NOT symmetric (the drain carries a ~1e-14 A term the source does not).
  * feature engineering (polarity-corrected terminal voltages, thermal voltage,
    EKV-style channel-current estimate, junction bias terms, ...)
  * a small physics baseline for the ON-state channel current (ChannelBaseline)
  * target transform y = asinh(x / x0): like sign(x)*ln|x| for |x| >> x0 but
    smooth through 0
  * SPICE noise-floor detection (noise_mask)
  * error metrics (R2, median and 95th-percentile relative error)
  * TransistorModel: one object per case. It is what gets saved to ml_models/
    and what Parts B and C call:  model.predict(df) -> DataFrame(id, ig, is, ib)

Each model type is a separate file in this folder (model_*.py) that defines
NAME, LABEL, HYPERPARAMS and make_regressor().
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

INPUTS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj"]
TARGETS = ["id", "ig", "is", "ib"]
K_OVER_Q = 8.617333262e-5          # Boltzmann constant / electron charge, V/K
ULP_1 = 2.0 ** -52                 # spacing of doubles near 1.0 (= 2.22e-16)
NOISE_ABS = 1e-14                  # only values below this can be noise
UNRESOLVED_FRACTION = 0.5          # a current is "unresolved" if > half its samples are noise
I_FLOOR = 1e-15                    # A, floor in the relative-error denominator (see rel_error)

# How Id and Is are learned, per case (see TransistorModel):
#   "kcl"    : learn g = (i_hi - i_lo)/|vds| (as a correction to a physics baseline),
#              ig and ib; rebuild id and is from Kirchhoff's law
#   "direct" : learn i_hi, i_lo, ig, ib separately
TARGET_MODE = {"NMOS_ON": "kcl", "PMOS_ON": "kcl", "NMOS_OFF": "direct", "PMOS_OFF": "direct"}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_case(case: str) -> pd.DataFrame:
    return pd.read_csv(config.DATA_DIR / "transistor" / f"{case}.csv")


def split_indices(n: int, seed: int = config.SEED_SPLIT, frac=(0.70, 0.15, 0.15)):
    """Fixed random 70/15/15 split -> (train_idx, val_idx, test_idx)."""
    perm = np.random.default_rng(seed).permutation(n)
    n_tr = int(round(frac[0] * n))
    n_va = int(round(frac[1] * n))
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]


def noise_mask(i: np.ndarray) -> np.ndarray:
    """
    True where a SPICE current is numerical noise rather than a physical value.

    Some gate currents (NMOS_ON with small Vgs, all of PMOS_OFF) come out of
    ngspice as exactly 0 or as small integer multiples of 2^-52 A (2.2e-16,
    4.4e-16, ...): the result of a cancellation in double precision, not a
    gate-tunnelling current. Those samples are not used for training or for
    the error metrics. Genuinely tiny but smooth currents (NMOS_OFF Ig ~1e-22 A)
    are not multiples of 2^-52 and are kept.
    """
    i = np.asarray(i, dtype=float)
    k = np.round(np.abs(i) / ULP_1)
    quantised = (np.abs(i) < NOISE_ABS) & (k >= 1) & (np.abs(np.abs(i) / ULP_1 - k) < 1e-3)
    return (i == 0.0) | quantised


# ----------------------------------------------------------------------------
# Device variables (oriented, polarity-corrected, normalised)
# ----------------------------------------------------------------------------
def device_vars(df: pd.DataFrame, dev: str) -> dict:
    """
    Polarity-corrected voltages (x(+1) NMOS, x(-1) PMOS, so "more positive Vgs"
    means "more ON" for both) with source/drain oriented so that vds >= 0:
    'hi' is the terminal at the higher corrected potential, 'lo' the other one.
    `swapped` is True where hi is the CSV *source* terminal.
    """
    pol = 1.0 if dev == "nmos" else -1.0
    nom = config.NOMINAL[dev]
    vd, vs, vg, vb = (pol * df[c].to_numpy(dtype=float) for c in ("vd", "vs", "vg", "vb"))
    swapped = vs > vd
    hi, lo = np.where(swapped, vs, vd), np.where(swapped, vd, vs)
    tk = df["temp"].to_numpy(dtype=float) + 273.15
    return {
        "pol": pol, "swapped": swapped, "hi": hi, "lo": lo, "vd": vd, "vs": vs,
        "vds_raw": (df["vd"] - df["vs"]).to_numpy(dtype=float),
        "vgs": vg - lo, "vgd": vg - hi, "vds": hi - lo, "vbs": vb - lo, "vbd": vb - hi,
        "vdd": df["vdd"].to_numpy(dtype=float), "tk": tk, "vt": K_OVER_Q * tk,
        "wl": df["w"].to_numpy(dtype=float) / config.L_MIN,
        # process parameters normalised to the model-card value (1.0 = nominal)
        "vth0": np.abs(df["vth0"].to_numpy(dtype=float)) / abs(nom["vth0"]),
        "toxe": df["toxe"].to_numpy(dtype=float) / nom["toxe"],
        "ndep": df["ndep"].to_numpy(dtype=float) / nom["ndep"],
        "u0": df["u0"].to_numpy(dtype=float) / nom["u0"],
        "xj": df["xj"].to_numpy(dtype=float) / nom["xj"],
        "vth0_nom": abs(nom["vth0"]),
    }


def _ln_ekv_conductance(vgs, vgd, vds, vth, n, vt):
    """ln[(F(vgs-vth) - F(vgd-vth)) / vds], F(x) = ln^2(1+exp(x/2nVt)), with the vds->0 limit."""
    a = 2 * n * vt
    us, ud = (vgs - vth) / a, (vgd - vth) / a
    fs, fd = np.logaddexp(0.0, us) ** 2, np.logaddexp(0.0, ud) ** 2
    dfdv = 2 * np.logaddexp(0.0, us) / (1 + np.exp(-us)) / a          # dF/dv at vgs
    small = vds < 1e-6
    g = np.where(small, dfdv, (fs - fd) / np.where(small, 1.0, vds))
    return np.log(np.maximum(g, 1e-300))


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
def features(v: dict) -> np.ndarray:
    """Model inputs (26 columns) from device_vars(); same for every model type and case."""
    vth_s = (v["vth0"] * v["vth0_nom"]
             + 0.3 * (np.sqrt(np.maximum(0.9 - v["vbs"], 0.05)) - np.sqrt(0.9)))
    ln_g = _ln_ekv_conductance(v["vgs"], v["vgd"], v["vds"], vth_s, 1.4, v["vt"])
    vt = v["vt"]
    cols = [
        v["vdd"], v["tk"] - 273.15, np.log(v["wl"]), v["hi"], v["lo"],
        v["vgs"], v["vgd"], v["vds"], v["vbs"], v["vbd"],
        v["vth0"], v["toxe"], v["ndep"], v["u0"], v["xj"],
        vt * 1e2, 1e3 / v["tk"],
        (v["vgs"] - vth_s) / vt, (v["vgd"] - vth_s) / vt, v["vds"] / vt,
        ln_g + np.log(v["wl"]), np.exp(np.minimum(v["vbs"], 0) / vt),
        np.exp(np.minimum(v["vbd"], 0) / vt), np.exp(-v["vds"] / vt),
        v["vgs"] / v["toxe"], v["vgd"] / v["toxe"],
        v["vd"], v["vs"], v["swapped"].astype(float),
    ]
    return np.column_stack(cols).astype(float)


FEATURE_NAMES = ["vdd", "temp", "ln_W/L", "v_hi", "v_lo", "vgs", "vgd", "vds", "vbs", "vbd",
                 "vth0/nom", "toxe/nom", "ndep/nom", "u0/nom", "xj/nom", "vt_x100", "1000/T",
                 "(vgs-vth)/vt", "(vgd-vth)/vt", "vds/vt", "ln_g_ekv", "exp(vbs/vt)",
                 "exp(vbd/vt)", "exp(-vds/vt)", "vgs/toxe", "vgd/toxe",
                 "vd_pol", "vs_pol", "source_is_hi"]


# ----------------------------------------------------------------------------
# Physics baseline for the ON-state channel conductance
# ----------------------------------------------------------------------------
class ChannelBaseline:
    """
    EKV-style compact expression for g = (i_hi - i_lo)/vds with 13 coefficients
    fitted by robust least squares on the TRAINING set:

      ln g = c0 + ln(W/L) + cu*ln(u0) + cT*ln(T/300) - ln(1 + theta*max(vgs-Vth, 0))
             + ln[(F(vgs-Vth) - F(vgd-Vth)) / vds],    F(x) = ln^2(1 + exp(x / 2nVt))
      Vth  = a*vth0 + kt1*(T/300-1) + gamma*(sqrt(0.9-vbs) - sqrt(0.9)) - eta*vds
             + ct*(toxe-1) + cn*(ndep-1) + cx*(xj-1)          (process params / nominal)
      n    = n0 * (1 + nT*(T/300-1))

    It captures ~95 % of the variation of ln g (it alone is ~8 % off); the ML
    regressor then only learns the remaining smooth correction.
    """

    P0 = [np.log(1e-5), 1.0, -1.5, 0.6, -0.1, 0.3, 0.05, 0.05, 0.05, 0.0, 1.4, 0.5, 0.0]

    def __init__(self):
        self.p = None

    @staticmethod
    def _ln_g(p, v):
        c0, cu, cT, a, kt1, gam, eta, ct, cn, cx, n0, th, nT = p
        t = v["tk"] / 300.0
        vth = (a * v["vth0"] + kt1 * (t - 1)
               + gam * (np.sqrt(np.maximum(0.9 - v["vbs"], 0.05)) - np.sqrt(0.9))
               - eta * v["vds"] + ct * (v["toxe"] - 1) + cn * (v["ndep"] - 1) + cx * (v["xj"] - 1))
        n = np.maximum(n0 * (1 + nT * (t - 1)), 0.5)
        return (c0 + np.log(v["wl"]) + cu * np.log(v["u0"]) + cT * np.log(t)
                - np.log(1 + np.abs(th) * np.maximum(v["vgs"] - vth, 0))
                + _ln_ekv_conductance(v["vgs"], v["vgd"], v["vds"], vth, n, v["vt"]))

    def fit(self, v: dict, g: np.ndarray):
        ok = g > 0
        sub = {k: (x[ok] if isinstance(x, np.ndarray) else x) for k, x in v.items()}
        y = np.log(g[ok])
        r = least_squares(lambda p: self._ln_g(p, sub) - y, self.P0,
                          loss="soft_l1", f_scale=0.1, max_nfev=5000)
        self.p = r.x
        return self

    def ln_g(self, v: dict) -> np.ndarray:
        return self._ln_g(self.p, v)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def rel_error(pred, true, floor=I_FLOOR):
    """
    |pred - true| / max(|true|, floor)   (fraction).

    The 1 fA floor keeps currents that are far below any gate-level leakage
    (and close to the SPICE resolution) from dominating the statistics: e.g.
    an OFF-state source current of 1e-19 A, where channel and junction
    currents cancel, would otherwise count a 1e-18 A miss as a "1000 % error".
    For a current whose typical value is below the floor (NMOS_OFF Ig ~ 1e-22 A)
    the floor is lowered to that current's 1st percentile.
    """
    return np.abs(pred - true) / np.maximum(np.abs(true), floor)


def r2(pred, true):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return 1.0 - ss_res / ss_tot


def metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """Metrics on the samples that are not SPICE noise."""
    keep = ~noise_mask(true)
    p, t = pred[keep], true[keep]
    a = np.abs(t)
    floor = I_FLOOR if np.median(a) > I_FLOOR else float(np.quantile(a, 0.01))
    e = rel_error(p, t, floor)
    e_plain = np.abs(p - t) / a
    return {
        "n": int(keep.sum()),
        "floor_A": floor,
        "r2_log": r2(np.arcsinh(p / floor), np.arcsinh(t / floor)),   # R2 on a log-like scale
        "r2_lin": r2(p, t),                                             # R2 on raw amperes
        "med_err_pct": 100 * np.median(e),
        "p95_err_pct": 100 * np.quantile(e, 0.95),
        "mean_err_pct": 100 * np.mean(e),
        "med_err_plain_pct": 100 * np.median(e_plain),
        "p95_err_plain_pct": 100 * np.quantile(e_plain, 0.95),
    }


# ----------------------------------------------------------------------------
# One model per case
# ----------------------------------------------------------------------------
class TransistorModel:
    """
    Predicts (id, ig, is, ib) for one case (e.g. NMOS_OFF) from a DataFrame
    with the 12 CSV input columns. Currents are INTO the device terminal, in A.

    ON cases ("kcl" mode): drain and source currents are dominated by the
    channel current, which flips from about -1e-4 A to +1e-4 A within a few mV
    of Vd = Vs. The model therefore learns three smooth quantities:
        ln(g / g_baseline),  g = (id - is) / (vd - vs)   (correction to ChannelBaseline)
        ig, ib
    and rebuilds the rest with Kirchhoff's law (id + is = -(ig + ib)):
        id = ( g*(vd-vs) - ig - ib) / 2,   is = (-g*(vd-vs) - ig - ib) / 2
    so id + ig + is + ib = 0 holds exactly and Vd = Vs is handled smoothly.
    (g is positive whichever terminal is higher: channel current always enters
    the terminal at the higher potential.)

    OFF cases ("direct" mode): channel and junction currents are both ~1e-14 A,
    so g is not smooth; id, ig, is and ib are learned directly.
    """

    def __init__(self, case: str, model_name: str, make_regressor):
        self.case = case
        self.dev = config.CASES[case]["type"]
        self.mode = TARGET_MODE[case]
        self.model_name = model_name
        self.learned = ["g", "ig", "ib"] if self.mode == "kcl" else list(TARGETS)
        self._make = make_regressor
        self.baseline = ChannelBaseline() if self.mode == "kcl" else None
        self.regs, self.scale, self.unresolved = {}, {}, {}

    # -- helpers ------------------------------------------------------------
    def _x(self, v):
        x = features(v)
        if self.baseline is not None:
            x = np.column_stack([x, self.baseline.ln_g(v)])
        return x

    @staticmethod
    def _targets(df, v):
        out = {t: df[t].to_numpy(dtype=float) for t in TARGETS}
        out["g"] = (out["id"] - out["is"]) / v["vds_raw"]
        return out

    def _encode(self, t, val, v):
        """Regression target: ln(g / g_baseline) for g, asinh(I / I0) for a current."""
        if t == "g":
            return np.log(val) - self.baseline.ln_g(v)
        return np.arcsinh(val / self.scale[t])

    def _decode(self, t, y, v):
        if t == "g":
            return np.exp(np.clip(y, -20, 20) + self.baseline.ln_g(v))
        return self.scale[t] * np.sinh(np.clip(y, -80, 80))

    # -- API ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, n_jobs: int = 1):
        v = device_vars(df, self.dev)
        tgt = self._targets(df, v)
        if self.baseline is not None:
            self.baseline.fit(v, tgt["g"])
        x = self._x(v)
        for t in self.learned:
            val = tgt[t]
            # g: the ~0.6 % of ON samples with g <= 0 (channel current smaller than the
            # drain/source junction difference, |id - is| ~ 1e-14 A) are left out
            good = (np.isfinite(val) & (val > 0)) if t == "g" else ~noise_mask(val)
            # a current that is mostly SPICE noise has no physical value to learn:
            # it is predicted as 0 A (it is below the simulator's resolution)
            self.unresolved[t] = good.mean() < 1 - UNRESOLVED_FRACTION
            if self.unresolved[t]:
                self.regs[t] = None
                continue
            self.scale[t] = float(np.quantile(np.abs(val[good]), 0.01))
            vg = {k: (a[good] if isinstance(a, np.ndarray) else a) for k, a in v.items()}
            reg = self._make(n_jobs=n_jobs)
            reg.fit(x[good], self._encode(t, val[good], vg))
            self.regs[t] = reg
        self._make = None          # not needed after training (keeps pickles small)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        v = device_vars(df, self.dev)
        x = self._x(v)
        p = {}
        for t in self.learned:
            if self.regs[t] is None:
                p[t] = np.zeros(len(df))
            else:
                p[t] = self._decode(t, np.asarray(self.regs[t].predict(x), float).ravel(), v)
        if self.mode == "kcl":
            s = p["g"] * v["vds_raw"]
            leak = -(p["ig"] + p["ib"])                       # = id + is
            p["id"], p["is"] = (s + leak) / 2, (-s + leak) / 2
        return pd.DataFrame({t: p[t] for t in TARGETS}, index=df.index)

    def evaluate(self, df: pd.DataFrame) -> dict:
        """{current: metrics dict, or None if the current is below SPICE resolution}."""
        pred = self.predict(df)
        return {t: None if self.unresolved.get(t) else
                metrics(pred[t].to_numpy(), df[t].to_numpy()) for t in TARGETS}


def selection_score(val_metrics: dict) -> float:
    """
    Number used to pick the best model on the validation set: the average,
    over the resolved terminal currents, of (median error + 95th-pct error)/2.
    Lower is better.
    """
    vals = [(m["med_err_pct"] + m["p95_err_pct"]) / 2 for m in val_metrics.values() if m]
    return float(np.mean(vals))


def train_one(case: str, model_module, n_jobs: int = 1):
    """Train one model type on one case -> (model, val_metrics, test_metrics, seconds)."""
    df = load_case(case)
    tr, va, te = split_indices(len(df))
    t0 = time.time()
    m = TransistorModel(case, model_module.NAME, model_module.make_regressor)
    m.fit(df.iloc[tr], n_jobs=n_jobs)
    return m, m.evaluate(df.iloc[va]), m.evaluate(df.iloc[te]), time.time() - t0


def format_metrics(res: dict) -> str:
    parts = []
    for t in TARGETS:
        m = res[t]
        parts.append(f"{t}: n/a" if m is None else
                     f"{t}: R2={m['r2_log']:.5f} med={m['med_err_pct']:.3f}% "
                     f"p95={m['p95_err_pct']:.3f}%")
    return " | ".join(parts)


def run_standalone(model_module):
    """Used by `python scripts/ml/model_<x>.py`: train on all 4 cases, print validation metrics."""
    for case in config.CASES:
        _, val, _, sec = train_one(case, model_module, n_jobs=-1)
        print(f"{case:9s} {model_module.LABEL}: {format_metrics(val)}")
        print(f"          score={selection_score(val):.3f}  ({sec:.0f} s)", flush=True)


def load_model(case: str) -> "TransistorModel":
    """Load the chosen model for a case from ml_models/<CASE>.joblib."""
    import joblib
    return joblib.load(config.ML_MODEL_DIR / f"{case}.joblib")
