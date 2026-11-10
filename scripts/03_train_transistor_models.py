"""
Part A, step 2: train and compare the transistor ML models (deliverables 4 and 5).

For each case (NMOS_ON, NMOS_OFF, PMOS_ON, PMOS_OFF):
  1. split data/transistor/<CASE>.csv 70/15/15 (train/validation/test, seed 42),
  2. train the 4 model types in scripts/ml/ on the training set:
       model_poly_ridge.py     polynomial ridge regression (baseline)
       model_random_forest.py  random forest
       model_xgboost.py        gradient-boosted trees (XGBoost)
       model_mlp.py            multilayer-perceptron neural network
  3. compare them on the validation set (R2, median and 95th-pct error per current),
  4. pick the best model on the validation set and report it on the test set,
  5. save it to ml_models/<CASE>.joblib.

Outputs
  results/tables/model_comparison.txt            human-readable summary (for the report)
  results/tables/model_comparison_validation.csv every model x case x current, validation
  results/tables/chosen_models_test.csv          chosen model per case, test set
  results/plots/pred_vs_spice_<CASE>.png         chosen model, test set, predicted vs SPICE
  ml_models/<CASE>.joblib, ml_models/model_index.json

Usage:  python scripts/03_train_transistor_models.py [--jobs 8]
Loading a saved model (Parts B and C):
  sys.path.insert(0, "scripts/ml"); import common; m = common.load_model("NMOS_ON")
  m.predict(df)   # df with columns vdd,temp,w,vd,vs,vg,vb,vth0,toxe,ndep,u0,xj
"""
import argparse
import datetime as dt
import importlib
import json
import shutil
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "ml"))
import config                     # noqa: E402
import common                     # noqa: E402

MODEL_MODULES = ["model_poly_ridge", "model_random_forest", "model_xgboost", "model_mlp"]
TABLES = config.RESULTS_DIR / "tables"
PLOTS = config.RESULTS_DIR / "plots"
WORK = config.ML_MODEL_DIR / "_work"


def job(case, module_name):
    """Train one (case, model type); save the model to WORK; return its metrics."""
    sys.path.insert(0, str(HERE / "ml"))
    mod = importlib.import_module(module_name)
    model, val, test, sec = common.train_one(case, mod, n_jobs=1)
    joblib.dump(model, WORK / f"{case}__{mod.NAME}.joblib", compress=3)
    return {"case": case, "model": mod.NAME, "label": mod.LABEL, "hyper": mod.HYPERPARAMS,
            "val": val, "test": test, "seconds": sec,
            "score": common.selection_score(val)}


def long_table(results, split):
    rows = []
    for r in results:
        for t in common.TARGETS:
            m = r[split][t]
            row = {"case": r["case"], "model": r["model"], "current": t}
            if m is None:
                row["note"] = "below SPICE resolution (predicted as 0)"
            else:
                row.update(m)
            rows.append(row)
    return pd.DataFrame(rows)


def fmt_row(label, res, extra=""):
    cells = []
    for t in common.TARGETS:
        m = res[t]
        cells.append(f"{'n/a':>7s} {'n/a':>7s} {'n/a':>7s}" if m is None else
                     f"{m['r2_log']:7.4f} {m['med_err_pct']:7.2f} {m['p95_err_pct']:7.2f}")
    return f"  {label:24s}" + " |".join(cells) + extra


def header_row():
    h1 = f"  {'':24s}" + " |".join(f"{t.upper():^23s}" for t in common.TARGETS)
    h2 = f"  {'model':24s}" + " |".join(f"{'R2':>7s} {'med%':>7s} {'p95%':>7s}"
                                       for _ in common.TARGETS)
    return [h1, h2, "  " + "-" * (len(h2) - 2)]


def write_text(results, best, path):
    by_case = {c: [r for r in results if r["case"] == c] for c in config.CASES}
    L = []
    L += ["PART A STEP 2 - TRANSISTOR ML MODEL COMPARISON",
          "=" * 46,
          f"Generated   : {dt.datetime.now():%Y-%m-%d %H:%M} by scripts/03_train_transistor_models.py",
          f"Technology  : {config.TECH}, VDD nominal {config.VDD_NOM} V, L = {config.L_MIN*1e9:.0f} nm",
          "Data        : data/transistor/<CASE>.csv, 10,000 SPICE samples per case",
          f"Split       : 70 / 15 / 15 % = 7,000 train / 1,500 validation / 1,500 test "
          f"(random, seed {config.SEED_SPLIT})",
          "Inputs      : vdd, temp, w, vd, vs, vg, vb, vth0, toxe, ndep, u0, xj "
          "(+ derived physics features, scripts/ml/common.py)",
          "Outputs     : id, ig, is, ib (A, current into the terminal)",
          "",
          "Metrics (per terminal current):",
          "  R2    coefficient of determination on a log-like scale, asinh(I / 1 fA)",
          "  med%  median relative error  |I_ML - I_SPICE| / max(|I_SPICE|, 1 fA)  x 100",
          "  p95%  95th-percentile of the same relative error",
          "  The 1 fA floor stops currents far below any gate leakage (and near SPICE",
          "  resolution) from dominating; NMOS_OFF ig (~1e-22 A) uses its own 1st-percentile",
          "  value as the floor instead. SPICE-noise samples (ig values that are exact",
          "  multiples of 2.2e-16 A) are excluded. PMOS_OFF ig is entirely SPICE noise",
          "  (|ig| < 1 fA), so it is predicted as 0 and marked n/a.",
          "Selection rule: lowest validation score = mean over the currents of (med% + p95%)/2.",
          "",
          "Model types (one .py file each in scripts/ml/):"]
    seen = {}
    for r in results:
        seen.setdefault(r["model"], r)
    for name, r in seen.items():
        L.append(f"  {r['label']:24s} model_{name}.py: {r['hyper']}")
    L += ["  All four share the same features and target handling (scripts/ml/common.py):",
          "  ON cases learn ln[(id-is)/(vd-vs)] as a correction to a fitted EKV-style",
          "  baseline plus ig and ib, and rebuild id and is with Kirchhoff's law;",
          "  OFF cases learn id, ig, is, ib directly. Currents are learned as asinh(I/I0).",
          "", ""]

    L += ["1) VALIDATION SET (1,500 samples per case) - model comparison", "-" * 62]
    for case, rs in by_case.items():
        L += ["", f"{case}"] + header_row()
        for r in sorted(rs, key=lambda r: MODEL_MODULES.index("model_" + r["model"])):
            mark = "   <== BEST" if r["model"] == best[case]["model"] else ""
            L.append(fmt_row(r["label"], r["val"],
                             f" | score {r['score']:7.2f}  ({r['seconds']:.0f} s){mark}"))
    L += ["", "", "2) CHOSEN MODEL PER CASE - TEST SET (1,500 unseen samples per case)", "-" * 66]
    L += [""] + header_row()
    for case in config.CASES:
        b = best[case]
        L.append(fmt_row(f"{case}: {b['model']}", b["test"]))
    L += ["", "Test set, plain relative error |I_ML - I_SPICE| / |I_SPICE| (no 1 fA floor):"]
    for case in config.CASES:
        b = best[case]
        parts = [f"{t} n/a" if b["test"][t] is None else
                 f"{t} med {b['test'][t]['med_err_plain_pct']:.2f}% / p95 "
                 f"{b['test'][t]['p95_err_plain_pct']:.1f}%" for t in common.TARGETS]
        L.append(f"  {case:9s} " + ", ".join(parts))
    L += ["", "Saved models: ml_models/<CASE>.joblib (see ml_models/model_index.json)",
          "Plots       : results/plots/pred_vs_spice_<CASE>.png (test set)",
          "Full numbers: results/tables/model_comparison_validation.csv, chosen_models_test.csv",
          ""]
    path.write_text("\n".join(L), encoding="utf-8")
    return "\n".join(L)


def pred_vs_spice_plot(case, model):
    df = common.load_case(case)
    _, _, te = common.split_indices(len(df))
    d = df.iloc[te]
    p = model.predict(d)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, t in zip(axes.ravel(), common.TARGETS):
        true, pred = d[t].to_numpy(), p[t].to_numpy()
        keep = ~common.noise_mask(true)
        if model.unresolved.get(t) or keep.sum() == 0:
            ax.text(0.5, 0.5, f"{t}: below SPICE resolution\n(|{t}| < 1 fA, predicted as 0)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        true, pred = true[keep], pred[keep]
        same = np.sign(true) == np.sign(pred)
        ax.scatter(np.abs(true[same]), np.abs(pred[same]), s=4, alpha=0.4, color="#2a6fdb",
                   label="same sign")
        if (~same).any():
            ax.scatter(np.abs(true[~same]), np.abs(pred[~same]), s=8, color="#d9534f",
                       label=f"sign differs ({(~same).sum()})")
        lo = max(min(np.abs(true).min(), np.abs(pred[pred != 0]).min()), 1e-24)
        hi = max(np.abs(true).max(), np.abs(pred).max())
        xs = np.array([lo, hi])
        ax.plot(xs, xs, "k-", lw=1)
        ax.plot(xs, xs * 1.1, "k:", lw=0.8)
        ax.plot(xs, xs / 1.1, "k:", lw=0.8, label="±10 %")
        m = common.metrics(p[t].to_numpy(), d[t].to_numpy())
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"|{t}| SPICE (A)"); ax.set_ylabel(f"|{t}| ML (A)")
        ax.set_title(f"{t}: median {m['med_err_pct']:.2f} %, p95 {m['p95_err_pct']:.1f} %",
                     fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, markerscale=2)
    fig.suptitle(f"{case} - {model.model_name} - test set (1,500 samples), predicted vs SPICE",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / f"pred_vs_spice_{case}.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8, help="parallel training jobs (each needs up to ~2 GB RAM)")
    args = ap.parse_args()

    from joblib import Parallel, delayed
    for p in (TABLES, PLOTS, config.ML_MODEL_DIR):
        p.mkdir(parents=True, exist_ok=True)
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    print(f"training {len(MODEL_MODULES)} model types x {len(config.CASES)} cases "
          f"({args.jobs} parallel jobs) ...", flush=True)
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(job)(c, m) for m in MODEL_MODULES for c in config.CASES)

    best = {c: min((r for r in results if r["case"] == c), key=lambda r: r["score"])
            for c in config.CASES}

    long_table(results, "val").to_csv(TABLES / "model_comparison_validation.csv",
                                      index=False, float_format="%.6g")
    long_table(list(best.values()), "test").to_csv(TABLES / "chosen_models_test.csv",
                                                   index=False, float_format="%.6g")
    txt = write_text(results, best, TABLES / "model_comparison.txt")

    index = {}
    for c, b in best.items():
        dst = config.ML_MODEL_DIR / f"{c}.joblib"
        shutil.copy2(WORK / f"{c}__{b['model']}.joblib", dst)
        model = joblib.load(dst)
        pred_vs_spice_plot(c, model)
        index[c] = {"file": dst.name, "model_type": b["model"], "label": b["label"],
                    "validation_score": round(b["score"], 4),
                    "target_mode": common.TARGET_MODE[c],
                    "test": {t: (None if b["test"][t] is None else
                                 {k: round(float(v), 6) for k, v in b["test"][t].items()})
                             for t in common.TARGETS}}
    (config.ML_MODEL_DIR / "model_index.json").write_text(json.dumps(index, indent=2))
    shutil.rmtree(WORK)

    print(txt)


if __name__ == "__main__":
    main()
