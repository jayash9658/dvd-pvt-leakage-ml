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
| `data/transistor/` | `NMOS_ON.csv`, `NMOS_OFF.csv`, `PMOS_ON.csv`, `PMOS_OFF.csv` (10,000 rows each) |
| `data/gates/`, `data/circuit/` | SPICE and ML results for Parts B, C |
| `scripts/` | All code, numbered in run order |
| `scripts/ml/` | Part A ML models: `common.py` (shared features/targets/metrics) + one file per model type |
| `ml_models/` | Chosen model per case (`<CASE>.joblib`) + `model_index.json` |
| `results/plots/`, `results/tables/` | Figures and tables for the report |
| `report/` | Final report |

## Run order

```
python scripts/01_generate_transistor_data.py   # Part A step 1: 4 x 10,000 SPICE samples (~10 min)
python scripts/02_check_transistor_data.py      # Kirchhoff check + 3 trend plots
python scripts/03_train_transistor_models.py    # Part A step 2: 4 model types x 4 cases (~10 min)
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

Reproducibility: `01_generate_transistor_data.py --out-dir <folder>` regenerates into another folder
without touching the project data. Verified on 2026-11-07: the regenerated CSVs and example netlists are
byte-for-byte identical to the files in `data/transistor/` and `netlists/transistor/`.

`scripts/config.py` holds every path, range and random seed. `scripts/spice_utils.py` has the
ngspice helpers. If ngspice lives elsewhere, set the `NGSPICE` environment variable to the full
path of `ngspice_con.exe`.

## Conventions

* Currents are **into the device terminal** (`id + ig + is + ib = 0`); ngspice's `I(Vx)` is negated.
* Units: V, °C, m (`w`, `toxe`, `xj`), cm⁻³ (`ndep`), m²/V·s (`u0`), A.
* Process parameters vth0, toxe, ndep, u0, xj are each drawn uniformly within ±10 % of the model
  card value (vth0 for PMOS is negative, as in the card). They're applied through
  `.param` variables in a parametrised copy of the model card (`netlists/transistor/32nm_LP_param.lib`).
* AD/AS/PD/PS aren't specified (same as the TA example), so BSIM4 uses its default ~1e-14 A
  junction saturation current; this is why Ib sits near 1e-14 A except at high temperature.

## Part A results: transistor ML models

Full details are in `results/tables/model_comparison.txt`, written by `03_train_transistor_models.py`.
The numbers below are from the run on 2026-11-09.

**Split:** 7,000 train / 1,500 validation / 1,500 test per case (random, seed 42).

**Metrics:**
* **Error:** |I_ML − I_SPICE| / max(|I_SPICE|, 1 fA).
* **R²:** computed on asinh(I / 1 fA), a log-like scale.
* **Excluded samples:** Ig values that are SPICE noise (exact multiples of 2.2e-16 A) aren't scored. PMOS_OFF Ig is entirely noise, so it's n/a and predicted as 0.

**How the best model is picked:** the lowest validation score, where score = mean over the currents of (median + p95) / 2.

### Model types (one file each in `scripts/ml/`)

| Model | File | Main settings |
|---|---|---|
| Polynomial ridge | `model_poly_ridge.py` | all degree-3 terms of the standardised features, ridge α = 1e-3 |
| Random forest | `model_random_forest.py` | 300 trees, max_features 0.5 |
| XGBoost | `model_xgboost.py` | lr 0.03, depth 6, subsample/colsample 0.8, early stopping on 10 % of train |
| MLP | `model_mlp.py` | 3 × 128 tanh, Adam lr 1e-3, batch 64, early stopping on 10 % of train |

All four use the same preprocessing (`scripts/ml/common.py`):
* **Inputs:** the 12 CSV inputs plus derived physics features, such as polarity-corrected Vgs/Vds/Vbs, Vt = kT/q and an EKV-style channel-current estimate.
* **Targets:** currents are learned as asinh(I/I0), which behaves like sign·log|I|.
* **ON cases:** the model learns ln[(Id − Is)/(Vd − Vs)] as a correction to a fitted EKV-style baseline, plus Ig and Ib. Id and Is are then rebuilt with Kirchhoff's law, so the predictions always satisfy Id + Ig + Is + Ib = 0.
* **OFF cases:** Id, Ig, Is and Ib are learned directly.

### Validation score per case (lower is better)

| Case | Poly ridge | Random forest | XGBoost | MLP | Chosen |
|---|---|---|---|---|---|
| NMOS_ON | 16.43 | 17.33 | 9.07 | **3.82** | MLP |
| NMOS_OFF | 196.50 | 33.16 | 24.45 | **8.76** | MLP |
| PMOS_ON | 12.69 | 14.52 | 7.35 | **4.30** | MLP |
| PMOS_OFF | 289.79 | 27.54 | 28.55 | **6.64** | MLP |

### Chosen models: test set (median / 95th-percentile error)

| Case | Id | Ig | Is | Ib |
|---|---|---|---|---|
| NMOS_ON | 1.76 / 7.83 % | 1.09 / 3.98 % | 1.76 / 7.61 % | 1.54 / 5.01 % |
| NMOS_OFF | 3.39 / 19.05 % | 1.07 / 4.12 % | 6.53 / 35.87 % | 0.55 / 4.04 % |
| PMOS_ON | 2.45 / 10.47 % | 0.68 / 2.94 % | 2.46 / 10.37 % | 1.47 / 6.77 % |
| PMOS_OFF | 1.35 / 6.93 % | n/a | 2.32 / 22.06 % | 0.68 / 4.51 % |

Test-set R² is ≥ 0.990 for every current.

### Notes for the report
* **Why the MLP was chosen:** XGBoost is slightly better on Id and Is in the ON cases (about 1 % median error), but its Ig and Ib errors are much worse. Overall, the MLP has the lowest score in all four cases.
* **Largest errors: Is in the OFF cases.** There, the channel and source-junction currents nearly cancel, leaving |Is| between 1e-21 and 1e-15 A. Without the 1 fA floor, the median Is error is 14.6 % (NMOS_OFF) and 6.8 % (PMOS_OFF).
* **Drain and source aren't symmetric in this model card.** The drain has a constant ~1e-14 A term that the source lacks, which is why the models keep the real drain and source labels.
* **Saved models and plots:** the models are in `ml_models/<CASE>.joblib`; `ml_models/model_index.json` has the model type and test metrics for each case. The predicted-vs-SPICE plots are in `results/plots/pred_vs_spice_<CASE>.png`.
