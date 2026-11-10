"""
Part A, data check (deliverable 3).

1. Kirchhoff check: id + ig + is + ib should be 0 for every sample.
   Writes results/tables/kirchhoff_check.csv
2. Trend plots (results/plots/):
     trend_off_current_vs_temp.png  - OFF drain current vs temperature
     trend_off_current_vs_vth0.png  - OFF drain current vs |vth0|
     trend_gate_current_vs_toxe.png - gate current vs toxe (ON cases)
   OFF-current plots use samples with |Vds| > 0.5*VDD, where the channel
   (subthreshold) leakage dominates the drain current.

Usage:  python scripts/02_check_transistor_data.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config                     # noqa: E402

DATA = config.DATA_DIR / "transistor"
PLOTS = config.RESULTS_DIR / "plots"
TABLES = config.RESULTS_DIR / "tables"
COLORS = {"NMOS": "#2a6fdb", "PMOS": "#d9534f"}


def load(case):
    return pd.read_csv(DATA / f"{case}.csv")


def kirchhoff_table():
    rows = []
    for case in config.CASES:
        d = load(case)
        cur = d[["id", "ig", "is", "ib"]]
        res = cur.sum(axis=1).abs()
        scale = cur.abs().max(axis=1)
        rows.append({
            "case": case, "n_samples": len(d),
            "max_abs_residual_A": res.max(),
            "median_abs_residual_A": res.median(),
            "median_rel_residual": (res / scale).median(),
            "p99_rel_residual": (res / scale).quantile(0.99),
            "median_max_terminal_current_A": scale.median(),
        })
    t = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    t.to_csv(TABLES / "kirchhoff_check.csv", index=False, float_format="%.3e")
    return t


def binned_median(x, y, bins=15):
    edges = np.linspace(x.min(), x.max(), bins + 1)
    idx = np.digitize(x, edges[1:-1])
    xc = np.array([x[idx == k].median() for k in range(bins)])
    yc = np.array([np.median(y[idx == k]) for k in range(bins)])
    return xc, yc


def trend_plot(xcol, ycol, cases, fname, xlabel, ylabel, title, filt=None, xabs=False,
               xscale=1.0):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for case in cases:
        d = load(case)
        if filt is not None:
            d = d[filt(d)]
        x = d[xcol].abs() if xabs else d[xcol]
        x = x * xscale
        y = d[ycol].abs()
        dev = case.split("_")[0]
        ax.scatter(x, y, s=3, alpha=0.25, color=COLORS[dev], label=f"{case} (n={len(d)})")
        xc, yc = binned_median(x.reset_index(drop=True), y.reset_index(drop=True))
        ax.plot(xc, yc, color=COLORS[dev], lw=2, marker="o", ms=3)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(markerscale=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / fname, dpi=150)
    plt.close(fig)


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    t = kirchhoff_table()
    pd.set_option("display.width", 200)
    print("Kirchhoff check (id+ig+is+ib):")
    print(t.to_string(index=False))

    big_vds = lambda d: (d.vd - d.vs).abs() > 0.5 * d.vdd   # noqa: E731
    off = ["NMOS_OFF", "PMOS_OFF"]
    tag = f"{config.TECH}, VDD={config.VDD_NOM} V +/-10 %"
    trend_plot("temp", "id", off, "trend_off_current_vs_temp.png",
               "Temperature (°C)", "|Id| OFF (A)",
               f"OFF current vs temperature ({tag}); |Vds|>0.5 VDD, line = binned median",
               filt=big_vds)
    trend_plot("vth0", "id", off, "trend_off_current_vs_vth0.png",
               "|vth0| (V)", "|Id| OFF (A)",
               f"OFF current vs vth0 ({tag}); |Vds|>0.5 VDD, line = binned median",
               filt=big_vds, xabs=True)
    trend_plot("toxe", "ig", ["NMOS_ON", "PMOS_ON"], "trend_gate_current_vs_toxe.png",
               "toxe (nm)", "|Ig| (A)",
               f"Gate current vs toxe, ON cases ({tag}); line = binned median",
               xscale=1e9)
    print(f"plots written to {PLOTS.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
