# ML-Driven PVT Variation Aware Leakage Estimation

Digital VLSI Design course project.
**Team assignment:** 32nm LP (`models/32nm_LP.pm`) · nominal VDD = 1.0 V · test circuit = 3-bit CLA adder
**Simulator:** ngspice-44.2 (Windows, `ngspice_con.exe`) · Python 3.10

## Folder layout

| Folder | Contents |
|---|---|
| `docs/` | Project spec and annotated slides |
| `models/` | PTM model cards (`32nm_LP.pm` is ours); `archives/` has the original zip/tar |
| `netlists/reference/` | TA's example netlist |
| `netlists/transistor/` | One example single-FET netlist per case + parametrised model library |
| `netlists/gates/`, `netlists/circuit/` | Gate and CLA netlists (Parts B, C) |
| `data/transistor/` | `NMOS_ON.csv`, `NMOS_OFF.csv`, `PMOS_ON.csv`, `PMOS_OFF.csv` (10,000 rows each, deliverable 1) |
| `data/transistor/components/` | BSIM4 internal current components for the same samples (used to train the physics models) |
| `data/gates/`, `data/circuit/` | SPICE and ML results for Parts B, C |
| `scripts/` | All code, numbered in run order |
| `scripts/ml/` | Part A ML models: `common.py` (physics equations, features, metrics) + one file per model type |
| `ml_models/` | Chosen model per case (`<CASE>.joblib`) + `model_index.json` |
| `results/plots/`, `results/tables/` | Figures and tables for the report |
| `report/` | Final report |

## Run order

```
python scripts/01_generate_transistor_data.py   # Part A step 1: 4 x 10,000 SPICE samples (~3 min)
python scripts/02_check_transistor_data.py      # Kirchhoff check + 3 trend plots
python scripts/03_train_transistor_models.py    # Part A step 2: 4 model types x 4 cases (~25 min)
```

Each model type can also be run on its own (prints validation metrics, saves nothing):
`python scripts/ml/model_poly_ridge.py`, `model_random_forest.py`, `model_xgboost.py`, `model_mlp.py`.

Step 2 writes `results/tables/model_comparison.txt` (validation comparison, chosen model per case and
its test-set errors, ready for the report), the CSV versions of those tables, the
`results/plots/pred_vs_spice_<CASE>.png` plots and the chosen models in `ml_models/`.
To use a model later (Parts B, C):

```python
import sys; sys.path.insert(0, "scripts/ml"); import common
m = common.load_model("NMOS_OFF")
m.predict(df)   # df columns: vdd,temp,w,vd,vs,vg,vb,vth0,toxe,ndep,u0,xj -> id,ig,is,ib
```

`01_generate_transistor_data.py --out-dir <folder>` regenerates into another folder without
touching the project data (all draws use fixed seeds).

`scripts/config.py` holds every path, range and random seed. `scripts/spice_utils.py` has the
ngspice helpers. If ngspice lives elsewhere, set the `NGSPICE` environment variable to the full
path of `ngspice_con.exe`.

## Conventions

* Currents are **into the device terminal** (`id + ig + is + ib = 0`); ngspice's `I(Vx)` is negated.
* Units: V, °C, m (`w`, `toxe`, `xj`), cm⁻³ (`ndep`), m²/V·s (`u0`), A.
* Process parameters vth0, toxe, ndep, u0, xj are each drawn uniformly within ±10 % of the model
  card value (vth0 for PMOS is negative, as in the card). They're applied through
  `.param` variables in a parametrised copy of the model card (`netlists/transistor/32nm_LP_param.lib`).
* AD/AS/PD/PS aren't specified (same as the TA example), so BSIM4 gives the drain junction its
  default 1e-14 A saturation current; this is why Ib sits near 1e-14 A except at high temperature.

## Part A, step 1: data

Every sample is an independent random draw (VDD first, then Vd, Vs, W, T, process), as in Section 4
of the spec. All draws are uniform, except that the extremes of Vd, Vs and temperature get a few more
samples (`config.EDGE_*`): with a small probability a value is drawn inside the band at one end of
its range instead of over the whole range.

| Share of samples | Plain uniform | Our data |
|---|---|---|
| T within 10 °C of −40 °C or 125 °C | 12.1 % | ~15.7 % |
| Vd (or Vs) within 5 % of 0 V or VDD | 10.0 % | ~15.6 % |
| T in the middle, 0–85 °C | 51.5 % | ~50 % |

Besides the four CSVs, the script saves BSIM4's internal current components for every sample
(`data/transistor/components/<CASE>_components.csv`): channel current, drain/source junction
currents, gate-body and gate-channel tunnelling, GIDL/GISL and impact ionisation. They're extra SPICE
outputs, not model inputs; the training uses them to fit one physics equation per mechanism.

## Part A, step 2: transistor ML models

Full details are in `results/tables/model_comparison.txt`, written by `03_train_transistor_models.py`.
The numbers below are from the run on 2026-11-12.

**Split:** 7,000 train / 1,500 validation / 1,500 test per case (random, seed 42).

**Metrics:**
* **Error:** |I_ML − I_SPICE| / max(|I_SPICE|, 1 fA). The OFF-state Ig (~1e-22 A) uses its own
  1st-percentile value as the floor instead.
* **R²:** computed on asinh(I / 1 fA), a log-like scale.
* **Reference values:** Id and Is are the CSV terminal currents. Ig and Ib are BSIM4's own gate and
  body currents, which equal the terminal values except where those carry ~1e-16 A rounding noise
  (all PMOS_OFF Ig, part of NMOS_ON Ig, PMOS Ib below ~1e-15 A).

**How the best model is picked:** the lowest validation score, where score = mean over the four
currents of (median + p95) / 2.

### Method: physics-based component models (`scripts/ml/common.py`)

Each terminal current is a sum of physical mechanisms. Each mechanism gets its own physics
equation, whose coefficients are fitted on the training set. The ML model then learns only the
correction ln|I_SPICE| − ln|I_physics|.

| Mechanism | Physics equation | ML correction |
|---|---|---|
| Channel current, ON | EKV equation (weak to strong inversion), Vth with body effect, DIBL, temperature, process | yes |
| Channel current, OFF | subthreshold equation I0·W/L·exp((Vgs−Vth)/nVt)·(1−exp(−Vds/Vt)) | yes |
| Source junction | reverse diode, BSIM4 temperature law (Eg(T), XTI) | yes |
| Gate-body tunnelling | direct-tunnelling form f(W, toxe, Vgb, T) | yes |
| Gate-channel tunnelling (source and drain side) | same form with Vgs / Vgd | yes |
| Impact ionisation | f(channel current, Vds) | yes |
| Drain junction | diode with 1e-14 A saturation current (exact) | no |
| GIDL / GISL | BSIM4 GIDL equation with the model-card constants (exact) | no |

The four terminal currents are then summed from the mechanisms with Kirchhoff's law, so
Id + Ig + Is + Ib = 0 holds exactly. Gate-edge tunnelling (igs, igd) is zero for this model card.

### Model types (one file each in `scripts/ml/`)

| Model | File | Main settings |
|---|---|---|
| Polynomial ridge | `model_poly_ridge.py` | all degree-3 terms of the 31 standardised inputs, ridge α = 1e-3 |
| Random forest | `model_random_forest.py` | 300 trees, max_features 0.5 |
| XGBoost | `model_xgboost.py` | lr 0.03, depth 6, subsample/colsample 0.8, early stopping on 10 % of train |
| MLP | `model_mlp.py` | 3 × 128 tanh, Adam lr 1e-3, batch 64, early stopping on 10 % of train |

### Validation score per case (lower is better)

| Case | Physics only (no ML) | Poly ridge | Random forest | XGBoost | MLP | Chosen |
|---|---|---|---|---|---|---|
| NMOS_ON | 52.42 | 2.82 | 7.26 | 3.69 | **1.68** | MLP |
| NMOS_OFF | 12.42 | 3.10 | 2.70 | 2.07 | **0.94** | MLP |
| PMOS_ON | 38.21 | 2.76 | 6.01 | 3.02 | **1.07** | MLP |
| PMOS_OFF | 12.61 | 3.40 | 2.69 | **0.94** | 1.12 | XGBoost |

### Chosen models: test set (median / 95th-percentile error)

| Case | Id | Ig | Is | Ib |
|---|---|---|---|---|
| NMOS_ON (MLP) | 0.56 / 2.79 % | 1.02 / 3.99 % | 0.58 / 2.89 % | 0.34 / 1.54 % |
| NMOS_OFF (MLP) | 0.01 / 1.57 % | 0.71 / 3.18 % | 0.17 / 1.54 % | 0.01 / 0.56 % |
| PMOS_ON (MLP) | 0.35 / 1.34 % | 0.79 / 3.29 % | 0.36 / 1.37 % | 0.18 / 1.15 % |
| PMOS_OFF (XGBoost) | 0.01 / 2.38 % | 0.31 / 1.67 % | 0.15 / 2.46 % | 0.01 / 0.27 % |

Test-set R² ≥ 0.9997 for every current. Mean errors are 0.06–1.35 %.

### Notes for the report
* **The physics equations did most of the work.** On their own they reach 0.04–8 % median error
  (OFF Id is almost entirely the exact drain-junction term). The ML corrections bring every current
  to about 1 % median or better.
* **Remaining largest errors:** gate current in the ON cases (~1 % median, ~4 % p95), where the
  tunnelling equation is only a rough fit, and the 95th percentile of OFF-state Id/Is near Vd ≈ Vs.
* **Drain and source aren't symmetric in this model card.** The drain junction has a fixed 1e-14 A
  saturation current; the source junction's is much smaller and strongly temperature dependent.
* **Saved models and plots:** the models are in `ml_models/<CASE>.joblib`; `ml_models/model_index.json`
  has the model type and test metrics for each case. The predicted-vs-SPICE plots are in
  `results/plots/pred_vs_spice_<CASE>.png`.
