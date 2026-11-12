"""
Part B deliverables (after running every scripts/gates/<gate>_gate.py 200):
the Gate Results Table (18 gate x input cases: SPICE median, ML median,
bias, median error, 95th-pct error) and the Gate parity plot (ML vs SPICE
leakage, all simulations, log-log).

Usage: python scripts/05_compile_gate_results.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GATES_DIR = HERE / "gates"
TABLES_DIR = ROOT / "results" / "tables"
PLOTS_DIR = ROOT / "results" / "plots"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FILES = ["inv_200_raw.csv", "nand2_200_raw.csv", "nor2_200_raw.csv",
         "and2_200_raw.csv", "or2_200_raw.csv"]

all_rows = pd.concat([pd.read_csv(GATES_DIR / f) for f in FILES], ignore_index=True)
# unify the input-pattern column name (INV uses "A", the others use "AB")
all_rows["input"] = all_rows["A"].fillna(all_rows.get("AB"))

rows = []
for (gate, inp), grp in all_rows.groupby(["gate", "input"]):
    bias = (grp["ml_leak"] - grp["spice_leak"]) / grp["spice_leak"].abs()
    rows.append(dict(
        gate=gate, input=inp, n=len(grp),
        spice_median=grp["spice_leak"].abs().median(),
        ml_median=grp["ml_leak"].abs().median(),
        bias_pct=100 * bias.median(),
        median_err_pct=grp["err_pct"].median(),
        p95_err_pct=grp["err_pct"].quantile(0.95),
    ))

table = pd.DataFrame(rows).sort_values(["gate", "input"])
table.to_csv(TABLES_DIR / "gate_results_table.csv", index=False)
pd.set_option("display.float_format", lambda x: f"{x:.4g}")
print(table.to_string(index=False))
print(f"\noverall median err% = {all_rows['err_pct'].median():.2f}   "
      f"overall p95 err% = {all_rows['err_pct'].quantile(0.95):.2f}")
print(f"total cases (corner x input): {len(all_rows)}   "
      f"({all_rows.groupby('gate').size().to_dict()})")

# Parity plot: ML vs SPICE leakage, all gates/corners/inputs, log-log
fig, ax = plt.subplots(figsize=(6, 6))
colors = {"INV": "C0", "NAND2": "C1", "NOR2": "C2", "AND2": "C3", "OR2": "C4"}
for gate, grp in all_rows.groupby("gate"):
    ax.scatter(grp["spice_leak"].abs(), grp["ml_leak"].abs(), s=6, alpha=0.35,
               label=gate, color=colors[gate])
lo = all_rows[["spice_leak", "ml_leak"]].abs().to_numpy()
lo, hi = lo[lo > 0].min(), lo.max()
ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("SPICE |leakage| (A)")
ax.set_ylabel("ML |leakage| (A)")
ax.set_title("Gate-level leakage: ML vs SPICE (all gates, 200 corners)")
ax.legend(markerscale=3)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "gate_parity_plot.png", dpi=150)
print(f"\nSaved: {TABLES_DIR / 'gate_results_table.csv'}, {PLOTS_DIR / 'gate_parity_plot.png'}")
