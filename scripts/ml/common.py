"""
Shared code for the Part A transistor models (step 2: train models).

Approach: physics-based component models
----------------------------------------
A transistor's four terminal currents are sums of a few physical leakage /
conduction mechanisms. BSIM4 reports each of them separately, and
01_generate_transistor_data.py saves them next to every dataset
(data/transistor/components/<CASE>_components.csv). Instead of learning the
terminal currents directly, each case model learns every mechanism on its own
and adds them up with Kirchhoff's law:

  mechanism (BSIM4 name)         how it is modelled
  ---------------------------------------------------------------------------
  channel current       (id)     ON : EKV equation (strong + weak inversion)
                                 OFF: subthreshold equation
                                       I = I0 W/L exp((Vgs-Vth)/nVt)(1-exp(-Vds/Vt)),
                                       Vth with body effect, DIBL, temperature,
                                       process; coefficients fitted   + ML correction
  drain junction        (ibd)    diode, Isat = 1e-14 A (BSIM4 default for this
                                 netlist): exact equation, no ML
  source junction       (ibs)    diode with BSIM4 temperature law
                                 Isat(T) ~ exp[(Eg(Tn)/kTn - Eg(T)/kT + XTI ln T/Tn)/N]
                                 fitted                                + ML correction
  GIDL / GISL           (igidl, igisl)
                                 BSIM4 GIDL equation with the model-card
                                 constants: exact, no ML
  gate-body tunnelling  (igb)    direct-tunnelling form ln I = f(W, toxe, Vox, T)
                                 fitted                                + ML correction
  gate-channel tunnelling (igcs, igcd)  same form, source and drain side
                                                                       + ML correction
  impact ionisation     (isub)   ln I = f(channel current, Vds, ...) fitted + ML correction
  gate-edge tunnelling  (igs, igd) zero for this model card, not modelled

Every learned mechanism is single-signed and smooth, so it is learned as
ln|I| - ln|I_physics| (the ML only corrects the fitted physics equation).
Terminal currents are then (pol = +1 NMOS / -1 PMOS, s = +1 if the device is in
normal mode, -1 if drain and source have swapped roles):

  id = pol*( s*Ich + Isub_d + Igidl - Ibd - Igc_d)
  is = pol*(-s*Ich + Isub_s + Igisl - Ibs - Igc_s)
  ig = pol*( Igb + Igcs + Igcd)
  ib = pol*( Ibd + Ibs - Isub - Igidl - Igisl - Igb)

so id + ig + is + ib = 0 holds exactly. This identity was checked against
ngspice's terminal currents (error < 1e-11 relative).

This module also has: data loading and the fixed 70/15/15 split, features,
the SPICE noise-floor detection, error metrics and model selection.
Each model type (regressor) is a separate file: model_*.py.
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
T_NOM = 300.15                     # K, model-card TNOM (27 degC)
ULP_1 = 2.0 ** -52                 # spacing of doubles near 1.0 (= 2.22e-16)
NOISE_ABS = 1e-14                  # only values below this can be noise
SOFT_FLOOR = 1e-5                  # soft floor of each mechanism, as a fraction of its 90th percentile
CHANNEL_FLOOR = 1e-18              # A/V, upper limit of the soft floor of the channel conductance
I_FLOOR = 1e-15                    # A, floor in the relative-error denominator (see rel_error)

# model-card constants used by the exact (not learned) equations
DRAIN_ISAT = 1e-14                                     # A
GIDL = {"agidl": 2e-4, "bgidl": 2.1e9, "cgidl": 2e-4, "egidl": 0.8}   # same for nmos / pmos

LEARNED = ["ch", "js", "gb", "gcs", "gcd", "sub"]     # mechanisms with an ML correction
COMP_NAME = {"ch": "id", "js": "ibs", "gb": "igb", "gcs": "igcs", "gcd": "igcd", "sub": "isub"}
COMP_LABEL = {"ch": "channel current (EKV / subthreshold)", "js": "source junction diode",
              "gb": "gate-body tunnelling", "gcs": "gate-channel tunnelling, source side",
              "gcd": "gate-channel tunnelling, drain side", "sub": "impact ionisation"}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_case(case: str) -> pd.DataFrame:
    return pd.read_csv(config.DATA_DIR / "transistor" / f"{case}.csv")


def load_components(case: str) -> pd.DataFrame:
    return pd.read_csv(config.DATA_DIR / "transistor" / "components" / f"{case}_components.csv")


def split_indices(n: int, seed: int = config.SEED_SPLIT, frac=(0.70, 0.15, 0.15)):
    """Fixed random 70/15/15 split -> (train_idx, val_idx, test_idx)."""
    perm = np.random.default_rng(seed).permutation(n)
    n_tr = int(round(frac[0] * n))
    n_va = int(round(frac[1] * n))
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]


def noise_mask(i: np.ndarray) -> np.ndarray:
    """
    True where a SPICE terminal current is numerical noise: exactly 0 or a small
    integer multiple of 2^-52 A (2.2e-16, 4.4e-16, ...), the result of a
    cancellation in double precision. Happens for the gate current of NMOS_ON
    at small Vgs and all of PMOS_OFF.
    """
    i = np.asarray(i, dtype=float)
    k = np.round(np.abs(i) / ULP_1)
    quantised = (np.abs(i) < NOISE_ABS) & (k >= 1) & (np.abs(np.abs(i) / ULP_1 - k) < 1e-3)
    return (i == 0.0) | quantised


# ----------------------------------------------------------------------------
# Device variables
# ----------------------------------------------------------------------------
def device_vars(df: pd.DataFrame, dev: str) -> dict:
    """
    Polarity-corrected voltages (x(+1) NMOS, x(-1) PMOS, so that "more positive
    Vgs" means "more ON" for both). Mode-oriented quantities (vgs, vgd, vds,
    vbs, vbd) use the terminal at the higher corrected potential as drain, as
    BSIM4 does internally; `fwd` is True when that is the netlist drain.
    Junction voltages vbd_p / vbs_p refer to the physical drain / source.
    """
    pol = 1.0 if dev == "nmos" else -1.0
    nom = config.NOMINAL[dev]
    vd, vs, vg, vb = (pol * df[c].to_numpy(dtype=float) for c in ("vd", "vs", "vg", "vb"))
    fwd = vd >= vs
    hi, lo = np.where(fwd, vd, vs), np.where(fwd, vs, vd)
    tk = df["temp"].to_numpy(dtype=float) + 273.15
    return {
        "pol": pol, "fwd": fwd, "vd": vd, "vs": vs, "vg": vg, "vb": vb, "hi": hi, "lo": lo,
        "vgs": vg - lo, "vgd": vg - hi, "vds": hi - lo, "vbs": vb - lo, "vbd": vb - hi,
        "vgb": vg - vb, "vbd_p": vb - vd, "vbs_p": vb - vs,
        "vdd": df["vdd"].to_numpy(dtype=float), "tk": tk, "vt": K_OVER_Q * tk,
        "w": df["w"].to_numpy(dtype=float), "wl": df["w"].to_numpy(dtype=float) / config.L_MIN,
        # process parameters normalised to the model-card value (1.0 = nominal)
        "vth0": np.abs(df["vth0"].to_numpy(dtype=float)) / abs(nom["vth0"]),
        "toxe": df["toxe"].to_numpy(dtype=float) / nom["toxe"],
        "ndep": df["ndep"].to_numpy(dtype=float) / nom["ndep"],
        "u0": df["u0"].to_numpy(dtype=float) / nom["u0"],
        "xj": df["xj"].to_numpy(dtype=float) / nom["xj"],
        "toxe_m": df["toxe"].to_numpy(dtype=float),
        "vth0_nom": abs(nom["vth0"]),
    }


def _sub(v: dict, mask) -> dict:
    return {k: (a[mask] if isinstance(a, np.ndarray) else a) for k, a in v.items()}


# ----------------------------------------------------------------------------
# Exact equations (no ML)
# ----------------------------------------------------------------------------
def drain_junction(v):
    """BSIM4 drain diode with the default 1e-14 A saturation current (AD = PD = 0)."""
    return DRAIN_ISAT * np.expm1(v["vbd_p"] / v["vt"])          # negative = reverse leakage


def gidl_current(v, terminal):
    """
    BSIM4 GIDL (terminal='d') / GISL ('s'):
      I = AGIDL*W*(Vxg - EGIDL)/(3 toxe) * exp(-3 toxe BGIDL/(Vxg - EGIDL)) * Vxb^3/(CGIDL + Vxb^3)
    for Vxg > EGIDL and Vxb > 0, else 0 (x = physical drain or source).
    """
    vx = v["vd"] if terminal == "d" else v["vs"]
    t = vx - v["vg"] - GIDL["egidl"]
    vxb = vx - v["vb"]
    on = (t > 0) & (vxb > 0)
    ts = np.where(on, t, 1.0)
    toxe = v["toxe_m"]
    val = (GIDL["agidl"] * v["w"] * ts / (3 * toxe) * np.exp(-3 * toxe * GIDL["bgidl"] / ts)
           * vxb ** 3 / (GIDL["cgidl"] + np.where(on, vxb, 0.0) ** 3))
    return np.where(on, val, 0.0)


# ----------------------------------------------------------------------------
# Fitted physics equations (baselines the ML corrects)
# ----------------------------------------------------------------------------
def _ln_ekv_g(vgs, vgd, vds, vth, n, vt):
    """ln[(F(vgs-vth) - F(vgd-vth)) / vds], F(x) = ln^2(1+exp(x/2nVt)), with the vds->0 limit."""
    a = 2 * n * vt
    us, ud = (vgs - vth) / a, (vgd - vth) / a
    fs, fd = np.logaddexp(0.0, us) ** 2, np.logaddexp(0.0, ud) ** 2
    dfdv = 2 * np.logaddexp(0.0, us) / (1 + np.exp(-us)) / a
    small = vds < 1e-6
    g = np.where(small, dfdv, (fs - fd) / np.where(small, 1.0, vds))
    return np.log(np.maximum(g, 1e-300))


def _vth(p, v):
    """Threshold voltage: vth0, temperature (KT1), body effect, DIBL, process sensitivities."""
    a, kt1, gam, eta, ct, cn, cx = p
    t = v["tk"] / 300.0
    return (a * v["vth0"] + kt1 * (t - 1)
            + gam * (np.sqrt(np.maximum(0.9 - v["vbs"], 0.05)) - np.sqrt(0.9))
            - eta * v["vds"] + ct * (v["toxe"] - 1) + cn * (v["ndep"] - 1) + cx * (v["xj"] - 1))


class Baseline:
    """A fitted physics equation returning ln|I_physics| (or ln g for the channel)."""
    name = ""
    equation = ""

    def fit(self, v, ln_y):
        raise NotImplementedError

    def ln(self, v):
        raise NotImplementedError


class EKVChannel(Baseline):
    """
    ON-state channel conductance g = Ich/Vds (EKV, valid from weak to strong inversion):
      ln g = c0 + ln(W/L) + cu ln(u0) + cT ln(T/300) - ln(1 + theta*max(Vgs-Vth,0))
             + ln[(F(Vgs-Vth) - F(Vgd-Vth))/Vds],   F(x) = ln^2(1 + exp(x/2nVt)),
      n = n0 (1 + nT (T/300-1))
    """
    name = "EKV channel"
    P0 = [np.log(1e-5), 1.0, -1.5, 0.5, 0.0, 1.4, 0.6, -0.1, 0.3, 0.05, 0.05, 0.05, 0.0]

    def _f(self, p, v):
        c0, cu, cT, th, nT, n0 = p[:6]
        vth = _vth(p[6:], v)
        n = np.maximum(n0 * (1 + nT * (v["tk"] / 300 - 1)), 0.5)
        return (c0 + np.log(v["wl"]) + cu * np.log(v["u0"]) + cT * np.log(v["tk"] / 300)
                - np.log(1 + np.abs(th) * np.maximum(v["vgs"] - vth, 0))
                + _ln_ekv_g(v["vgs"], v["vgd"], v["vds"], vth, n, v["vt"]))

    def fit(self, v, ln_y):
        self.p = least_squares(lambda p: self._f(p, v) - ln_y, self.P0, loss="soft_l1",
                               f_scale=0.1, max_nfev=5000).x
        return self

    def ln(self, v):
        return self._f(self.p, v)


class SubthresholdChannel(Baseline):
    """
    OFF-state channel conductance g = Ich/Vds (subthreshold / weak-inversion equation):
      Ich = I0 (W/L) u0^cu (T/300)^cT exp((Vgs - Vth)/(n Vt)) (1 - exp(-Vds/Vt))
      ln g = ln I0 + ln(W/L) + cu ln(u0) + cT ln(T/300) + (Vgs-Vth)/(nVt) + ln[(1-exp(-Vds/Vt))/Vds]
      n = n0 (1 + nT (T/300-1)) + nb (sqrt(0.9 - Vbs) - sqrt(0.9))
      Vth = a vth0 + KT1 (T/300-1) + gamma body effect - eta Vds (DIBL) + process terms
    """
    name = "subthreshold channel"
    P0 = [np.log(1e-6), 1.0, 1.0, 1.4, 0.0, 0.0, 0.6, -0.1, 0.3, 0.05, 0.05, 0.05, 0.0]

    def _f(self, p, v):
        c0, cu, cT, n0, nT, nb = p[:6]
        vth = _vth(p[6:], v)
        n = np.maximum(n0 * (1 + nT * (v["tk"] / 300 - 1))
                       + nb * (np.sqrt(np.maximum(0.9 - v["vbs"], 0.05)) - np.sqrt(0.9)), 0.5)
        vds = np.maximum(v["vds"], 1e-12)
        return (c0 + np.log(v["wl"]) + cu * np.log(v["u0"]) + cT * np.log(v["tk"] / 300)
                + (v["vgs"] - vth) / (n * v["vt"]) + np.log(-np.expm1(-vds / v["vt"]) / vds))

    fit = EKVChannel.fit
    ln = EKVChannel.ln


class SourceDiode(Baseline):
    """
    Source junction (reverse-biased diode) with the BSIM4 temperature law:
      I = Isat(T) (1 - exp(Vbs/Vt)),
      ln Isat = a0 + a1 ln(W/L) + [Eg(Tn)/(k Tn) - Eg(T)/(k T) + XTI ln(T/Tn)] / N,
      Eg(T) = 1.16 - 7.02e-4 T^2/(T + 1108)
    """
    name = "source diode"
    P0 = [np.log(1e-15), 0.0, 3.0, 1.0]

    @staticmethod
    def _eg(tk):
        return 1.16 - 7.02e-4 * tk ** 2 / (tk + 1108.0)

    def _f(self, p, v):
        a0, a1, xti, nj = p
        tk = v["tk"]
        arg = (self._eg(T_NOM) / (K_OVER_Q * T_NOM) - self._eg(tk) / (K_OVER_Q * tk)
               + xti * np.log(tk / T_NOM)) / max(nj, 0.3)
        rev = -np.expm1(v["vbs_p"] / v["vt"])
        return a0 + a1 * np.log(v["wl"]) + arg + np.log(np.maximum(rev, 1e-300))

    fit = EKVChannel.fit
    ln = EKVChannel.ln


class LogLinear(Baseline):
    """ln|I| = theta . phi(x), fitted by least squares (phi = physics-motivated terms)."""

    def fit(self, v, ln_y):
        self.theta, *_ = np.linalg.lstsq(self.phi(v), ln_y, rcond=None)
        return self

    def ln(self, v):
        return self.phi(v) @ self.theta


class Tunnelling(LogLinear):
    """
    Direct gate-oxide tunnelling (BSIM4 Igb / Igc form
      I ~ W L Vox Vaux exp(-B toxe (A - B' Vox)(1 + C Vox))):
      ln I = theta . [1, ln W, toxe, toxe*Vox, Vox, Vox^2, ln Vaux-like terms, T terms, process]
    with Vox the relevant gate voltage (Vgb for gate-body, Vgs / Vgd for gate-channel).
    """

    def __init__(self, vox_key):
        self.vox_key = vox_key
        self.name = f"tunnelling ({vox_key})"

    def phi(self, v):
        vox = v[self.vox_key]
        t = v["toxe"]
        return np.column_stack([
            np.ones_like(t), np.log(v["wl"]), t, t * vox, vox, vox ** 2, vox ** 3,
            v["vgs"], v["vgd"], v["vgb"], np.logaddexp(0, (v["vgs"] - 0.5) / 0.05),
            np.log(v["tk"] / 300), 300 / v["tk"], v["vth0"], v["ndep"], v["vds"],
        ])


class ImpactIonisation(LogLinear):
    """
    Substrate current from impact ionisation, Isub ~ (Vds - Vdsat) Ich exp(-beta/(Vds - Vdsat)):
      ln Isub = theta . [1, ln Ich_physics, ln(Vds + 1e-3), 1/(Vds + 0.05), Vds, Vgs, T terms]
    where ln Ich_physics comes from the fitted channel equation.
    """
    name = "impact ionisation"

    def __init__(self, channel: Baseline):
        self.channel = channel

    def phi(self, v):
        vds = v["vds"]
        ln_ich = self.channel.ln(v) + np.log(np.maximum(vds, 1e-12))
        return np.column_stack([
            np.ones_like(vds), ln_ich, np.log(vds + 1e-3), 1 / (vds + 0.05), vds, vds ** 2,
            v["vgs"], v["vgd"], np.log(v["tk"] / 300), 300 / v["tk"], v["vth0"], v["toxe"],
        ])


# ----------------------------------------------------------------------------
# Features for the ML corrections
# ----------------------------------------------------------------------------
def features(v: dict) -> np.ndarray:
    """Inputs common to every ML correction (polarity-corrected, mode-oriented)."""
    vt = v["vt"]
    vth = v["vth0"] * v["vth0_nom"]
    cols = [
        v["vdd"], v["tk"] - 273.15, np.log(v["wl"]), v["vd"], v["vs"],
        v["vgs"], v["vgd"], v["vds"], v["vbs"], v["vbd"], v["vgb"],
        v["vth0"], v["toxe"], v["ndep"], v["u0"], v["xj"],
        vt * 1e2, 1e3 / v["tk"],
        (v["vgs"] - vth) / vt, (v["vgd"] - vth) / vt, v["vds"] / vt,
        np.exp(np.minimum(v["vbs_p"], 0) / vt), np.exp(np.minimum(v["vbd_p"], 0) / vt),
        np.exp(-v["vds"] / vt), v["fwd"].astype(float),
    ]
    return np.column_stack(cols).astype(float)


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
    For a current whose typical value is below the floor (OFF-state Ig
    ~ 1e-22 A) the floor is lowered to a tenth of that current's median.
    """
    return np.abs(pred - true) / np.maximum(np.abs(true), floor)


def r2(pred, true):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return 1.0 - ss_res / ss_tot


def metrics(pred: np.ndarray, true: np.ndarray, keep=None) -> dict:
    keep = np.ones(len(true), bool) if keep is None else keep
    p, t = pred[keep], true[keep]
    a = np.abs(t)
    floor = I_FLOOR if np.median(a) > I_FLOOR else 0.1 * float(np.median(a))
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


def reference_currents(df: pd.DataFrame, comp: pd.DataFrame, dev: str) -> pd.DataFrame:
    """
    The SPICE values the models are scored against: id and is are the terminal
    currents from the CSV; ig and ib are BSIM4's own gate and body currents
    (sums of their components, saved by 01_generate_transistor_data.py). The
    two agree except where the terminal value carries double-precision rounding
    noise of ~1e-16 A: the gate current of every PMOS_OFF sample and some
    NMOS_ON ones (exact multiples of 2.2e-16 A), and the body current of PMOS
    samples whose |ib| is below ~1e-15 A (gate and body sit at VDD).
    """
    pol = 1.0 if dev == "nmos" else -1.0
    ref = df[TARGETS].copy()
    ref["ig"] = pol * (comp["igb"] + comp["igs"] + comp["igd"] + comp["igcs"] + comp["igcd"])
    ref["ib"] = pol * (comp["ibd"] + comp["ibs"] - comp["isub"] - comp["igidl"] - comp["igisl"]
                       - comp["igb"])
    return ref


# ----------------------------------------------------------------------------
# One model per case
# ----------------------------------------------------------------------------
def x_ln(model, k, v, mask):
    return model._ln_base(k, _sub(v, mask))


class TransistorModel:
    """
    Predicts (id, ig, is, ib) for one case from a DataFrame with the 12 CSV input
    columns. Currents are INTO the device terminal, in A. See the module
    docstring for the physics decomposition.
    """

    def __init__(self, case: str, model_name: str, make_regressor):
        self.case = case
        self.dev = config.CASES[case]["type"]
        self.is_on = case.endswith("_ON")
        self.model_name = model_name
        self._make = make_regressor
        self.base, self.regs, self.sign, self.n_fit, self.ln_eps = {}, {}, {}, {}, {}

    # targets: positive magnitudes of each learned mechanism
    def _magnitudes(self, comp: pd.DataFrame, v: dict) -> dict:
        out = {}
        for k in LEARNED:
            x = comp[COMP_NAME[k]].to_numpy(dtype=float)
            if k == "ch":
                out[k] = x / np.maximum(v["vds"], 1e-12)        # conductance Ich / Vds
            elif k == "js":
                out[k] = -x                                    # reverse diode current
            else:
                out[k] = x
        return out

    def _x(self, v):
        """
        ML inputs: the features plus the six physics-equation values. Each input
        is limited to the range seen in training. Inside a gate a transistor often
        sits exactly on a rail (e.g. source = body, where ln of the junction
        current is -infinity); without the limit such inputs would push the ML
        correction far outside what it has learned.
        """
        lns = [self._ln_base(k, v) for k in LEARNED]
        x = np.column_stack([features(v)] + lns)
        if getattr(self, "x_lo", None) is not None:
            x = np.clip(np.nan_to_num(x, nan=0.0, posinf=np.inf, neginf=-np.inf), self.x_lo, self.x_hi)
        return x

    def _ln_base(self, k, v):
        """ln(I_physics + eps_k): the physics value with the soft floor eps_k (see fit)."""
        return np.logaddexp(self.base[k].ln(v), self.ln_eps[k])

    def set_input_range(self, df: pd.DataFrame):
        """Record the range of every ML input over the training rows `df`."""
        self.x_lo = self.x_hi = None
        x = self._x(device_vars(df, self.dev))
        x = np.where(np.isfinite(x), x, np.nan)
        self.x_lo, self.x_hi = np.nanmin(x, axis=0), np.nanmax(x, axis=0)

    def fit(self, df: pd.DataFrame, comp: pd.DataFrame, n_jobs: int = 1):
        v = device_vars(df, self.dev)
        mag = self._magnitudes(comp, v)
        # 1) fit the physics equations (training set only)
        ch = EKVChannel() if self.is_on else SubthresholdChannel()
        bases = {"ch": ch, "js": SourceDiode(), "gb": Tunnelling("vgb"),
                 "gcs": Tunnelling("vgs"), "gcd": Tunnelling("vgd"), "sub": ImpactIonisation(ch)}
        for k in LEARNED:
            m = mag[k]
            self.sign[k] = 1.0 if np.median(m) >= 0 else -1.0
            good = np.isfinite(m) & (self.sign[k] * m > 0)
            self.n_fit[k] = int(good.sum())
            bases[k].fit(_sub(v, good), np.log(self.sign[k] * m[good]))
            self.base[k] = bases[k]
            # soft floor: 1e-5 of the mechanism's 90th-percentile magnitude. Below it
            # the current is negligible, and without a floor ln|I| would run over
            # 20-40 decades (down to exactly zero bias) and use up the ML's accuracy there.
            # (The channel conductance spans many useful decades, so its floor is at most 1e-18 A/V.)
            eps = SOFT_FLOOR * np.quantile(self.sign[k] * m[good], 0.90)
            self.ln_eps[k] = float(np.log(min(eps, CHANNEL_FLOOR) if k == "ch" else eps))
        # 2) ML learns the correction ln(|I| + eps) - ln(|I_physics| + eps) for each mechanism
        x = self._x(v)
        for k in LEARNED:
            m = mag[k]
            good = np.isfinite(m) & (self.sign[k] * m > 0)      # exact zeros (Vds = 0, zero bias) carry no information
            y = np.logaddexp(np.log(self.sign[k] * m[good]), self.ln_eps[k]) - x_ln(self, k, v, good)
            reg = self._make(n_jobs=n_jobs)
            reg.fit(x[good], y)
            self.regs[k] = reg
        self._make = None
        self.set_input_range(df)
        return self

    def predict_components(self, df: pd.DataFrame, physics_only: bool = False) -> dict:
        v = device_vars(df, self.dev)
        x = None if physics_only else self._x(v)
        c = {}
        for k in LEARNED:
            if physics_only:
                c[k] = self.sign[k] * np.exp(np.clip(self.base[k].ln(v), -745, 700))
            else:
                ln = self._ln_base(k, v) + np.clip(np.asarray(self.regs[k].predict(x), float).ravel(), -20, 20)
                c[k] = self.sign[k] * np.maximum(np.exp(np.clip(ln, -745, 700)) - np.exp(self.ln_eps[k]), 0.0)
        c["ch"] = c["ch"] * v["vds"]                    # conductance -> current
        c["js"] = -c["js"]                             # back to BSIM4 sign (negative)
        c["jd"] = drain_junction(v)
        c["gidl"], c["gisl"] = gidl_current(v, "d"), gidl_current(v, "s")
        return c, v

    def predict(self, df: pd.DataFrame, physics_only: bool = False) -> pd.DataFrame:
        c, v = self.predict_components(df, physics_only)
        pol, fwd = v["pol"], v["fwd"]
        s = np.where(fwd, 1.0, -1.0)
        sub_d, sub_s = np.where(fwd, c["sub"], 0.0), np.where(fwd, 0.0, c["sub"])
        gc_d = np.where(fwd, c["gcd"], c["gcs"])          # BSIM4 reports igcs/igcd in mode order
        gc_s = np.where(fwd, c["gcs"], c["gcd"])
        i_d = pol * (s * c["ch"] + sub_d + c["gidl"] - c["jd"] - gc_d)
        i_s = pol * (-s * c["ch"] + sub_s + c["gisl"] - c["js"] - gc_s)
        i_g = pol * (c["gb"] + c["gcs"] + c["gcd"])
        i_b = pol * (c["jd"] + c["js"] - c["sub"] - c["gidl"] - c["gisl"] - c["gb"])
        return pd.DataFrame({"id": i_d, "ig": i_g, "is": i_s, "ib": i_b}, index=df.index)

    def evaluate(self, df: pd.DataFrame, comp: pd.DataFrame, physics_only: bool = False) -> dict:
        pred = self.predict(df, physics_only)
        ref = reference_currents(df, comp, self.dev)
        return {t: metrics(pred[t].to_numpy(), ref[t].to_numpy()) for t in TARGETS}

    def baseline_summary(self) -> dict:
        return {k: self.base[k].name for k in LEARNED}


def selection_score(val_metrics: dict) -> float:
    """
    Number used to pick the best model on the validation set: the average,
    over the terminal currents, of (median error + 95th-pct error)/2. Lower is better.
    """
    vals = [(m["med_err_pct"] + m["p95_err_pct"]) / 2 for m in val_metrics.values() if m]
    return float(np.mean(vals))


def train_one(case: str, model_module, n_jobs: int = 1):
    """Train one model type on one case -> (model, val_metrics, test_metrics, seconds)."""
    df, comp = load_case(case), load_components(case)
    tr, va, te = split_indices(len(df))
    t0 = time.time()
    m = TransistorModel(case, model_module.NAME, model_module.make_regressor)
    m.fit(df.iloc[tr], comp.iloc[tr], n_jobs=n_jobs)
    return (m, m.evaluate(df.iloc[va], comp.iloc[va]), m.evaluate(df.iloc[te], comp.iloc[te]),
            time.time() - t0)


def format_metrics(res: dict) -> str:
    return " | ".join(f"{t}: R2={m['r2_log']:.5f} med={m['med_err_pct']:.3f}% "
                      f"p95={m['p95_err_pct']:.3f}%" for t, m in res.items())


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
