# ML-Driven PVT Variation Aware Leakage Estimation

Digital VLSI Design course project.
**Team assignment:** 32nm LP (`models/32nm_LP.pm`) · nominal VDD = 1.0 V · test circuit = 3-bit CLA adder
**Simulator:** ngspice-44.2 (Windows, `ngspice_con.exe`) · Python 3.10

## Results in one view

| Part | What is compared | Points | Median error | 95th-pct error | Mean accuracy |
|---|---|---|---|---|---|
| A | Transistor Id / Ig / Is / Ib, ON cases (test set) | 1,500 per case | 0.12–0.59 % | 1.1–2.8 % | 99.1–99.7 % |
| B | Leakage of INV, NAND2, NOR2, AND2, OR2 (18 cases × 200 corners) | 3,599 | 0.69 % | 2.07 % | **99.17 %** |
| C | Leakage of the 3-bit CLA (128 input vectors × 200 corners) | 25,491 | 0.73 % | 1.79 % | **99.20 %** |

## Folder layout

| Folder | Contents |
|---|---|
| `docs/` | Project spec and annotated slides |
| `models/` | PTM model cards (`32nm_LP.pm` is ours); `archives/` has the original zip/tar |
| `netlists/reference/` | TA's example netlist |
| `netlists/transistor/` | One example single-FET netlist per case + parametrised model library |
| `netlists/gates/` | One netlist per gate: `INV`, `NAND2`, `NOR2`, `AND2`, `OR2` `_example.cir` |
| `netlists/circuit/` | `CLA3.cir`, the 3-bit carry-lookahead adder (204 transistors) |
| `data/transistor/` | `NMOS_ON.csv`, `NMOS_OFF.csv`, `PMOS_ON.csv`, `PMOS_OFF.csv` (10,000 rows each, deliverable 1) |
| `data/transistor/components/` | BSIM4 internal current components for the same samples (used to train the physics models) |
| `data/gates/` | `corners.csv` (the 200 corners) and `gate_simulations.csv` (every gate simulation: voltages, SPICE and ML leakage) |
| `data/circuit/` | `cla3_simulations.csv` (every CLA operating point: SPICE and ML leakage) |
| `scripts/` | All code, numbered in run order |
| `scripts/ml/` | Part A ML models: `common.py` (physics equations, features, metrics) + one file per model type |
| `ml_models/` | Chosen model per case (`<CASE>.joblib`) + `model_index.json` |
| `results/plots/`, `results/tables/` | Figures and tables for the report |
| `report/` | `Report.md`, the written report |

## Run order

```
python scripts/01_generate_transistor_data.py   # Part A step 1: 4 x 10,000 SPICE samples (~3 min)
python scripts/02_check_transistor_data.py      # Kirchhoff check + 3 trend plots
python scripts/03_train_transistor_models.py    # Part A step 2: 4 model types x 4 cases (~25 min)
python scripts/04_gate_validation.py            # Part B: 5 gates, 18 cases x 200 corners (~6 min)
python scripts/05_circuit_validation.py         # Part C: 3-bit CLA, 128 vectors x 200 corners (~35 min)
```

`scripts/gate_lib.py` holds what Parts B and C share: the gate definitions, netlist builders, the
corner draw and the ML gate model.

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
of the spec. Two things differ from plain uniform sampling, both set in `config.py`:

* **Mild corner enrichment** (`EDGE_*`): with a small probability a value of Vd, Vs or T is drawn
  inside the band at one end of its range instead of over the whole range.
* **Gate operating points** (`GATE_POINT_FRAC = 0.20`): for 20 % of the samples, (Vd, Vs) is a point
  that occurs inside a logic gate: a terminal exactly on a rail, drain within a small step of the
  source (1e-8 V to 0.3 V, log-uniform), or a terminal a small step from a rail. VDD, T, W and the
  process parameters of these samples are drawn as for all others. The other 80 % of the samples
  are unchanged by this. Without these points the models were accurate on the test set but wrong
  by 20–60 % inside gates, where ON transistors have Vd ≈ Vs and OFF transistors sit exactly on the
  rails; random (Vd, Vs) pairs almost never land there.

| Share of samples | Plain uniform | Our data |
|---|---|---|
| T within 10 °C of −40 °C or 125 °C | 12.1 % | ~15.7 % |
| T in the middle, 0–85 °C | 51.5 % | ~50 % |
| Vd within 5 % of 0 V or VDD | 10.0 % | ~28 % (12 % exactly on a rail) |
| \|Vd − Vs\| below 1 mV | 0.2 % | ~8 % |

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
  median / 10 as the floor instead.
* **R²:** computed on asinh(I / 1 fA), a log-like scale.
* **Reference values:** Id and Is are the CSV terminal currents. Ig and Ib are BSIM4's own gate and
  body currents, which equal the terminal values except where those carry ~1e-16 A rounding noise
  (all PMOS_OFF Ig, part of NMOS_ON Ig, PMOS Ib below ~1e-15 A).

**How the best model is picked:** the lowest validation score, where score = mean over the four
currents of (median + p95) / 2.

### Method: physics-based component models (`scripts/ml/common.py`)

Each terminal current is a sum of physical mechanisms. Each mechanism gets its own physics
equation, whose coefficients are fitted on the training set. The ML model then learns only the
correction ln(I_SPICE + ε) − ln(I_physics + ε).

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

Two details matter for use inside gates:

* **Soft floor ε.** Each mechanism has a floor ε = 1e-5 of its 90th-percentile magnitude (at most
  1e-18 A/V for the channel conductance). Below it the current is negligible. Without the floor,
  ln|I| runs over 20–40 decades down to zero bias and the ML spends its accuracy there.
* **Input limits.** Every ML input is limited to the range seen in training, so an operating point
  exactly on a rail (where, for example, ln of the junction current is −∞) can't push the
  correction outside what it has learned.

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
| NMOS_ON | 53.71 | 1.78 | 7.50 | 3.81 | **1.28** | MLP |
| NMOS_OFF | 1762.73 | 5.25 | 9.35 | 4.73 | **3.82** | MLP |
| PMOS_ON | 40.86 | 1.47 | 6.05 | 3.13 | **1.02** | MLP |
| PMOS_OFF | 38.57 | 5.80 | 4.04 | **1.98** | 2.30 | XGBoost |

The OFF-case scores are dominated by the OFF-state gate current (~1e-22 A), which is far below
anything that affects leakage. The physics-only NMOS_OFF score is large for the same reason.

### Chosen models: test set (median / 95th-percentile error)

| Case | Id | Ig | Is | Ib |
|---|---|---|---|---|
| NMOS_ON (MLP) | 0.53 / 2.77 % | 0.59 / 2.68 % | 0.53 / 2.59 % | 0.12 / 1.19 % |
| NMOS_OFF (MLP) | 0.01 / 1.12 % | 4.88 / 22.80 % | 0.12 / 1.11 % | 0.01 / 0.34 % |
| PMOS_ON (MLP) | 0.32 / 1.56 % | 0.56 / 2.49 % | 0.34 / 1.55 % | 0.16 / 1.15 % |
| PMOS_OFF (XGBoost) | 0.01 / 2.90 % | 1.31 / 8.75 % | 0.15 / 3.18 % | 0.01 / 0.28 % |

Test-set R² ≥ 0.997 for every current. Mean errors are 0.06–0.89 %, except the OFF-state gate
current (7.8 % NMOS, 2.3 % PMOS).

## Part B: gate-level validation (`04_gate_validation.py`)

* **Gates:** static CMOS INV, NAND2, NOR2; AND2 = NAND2 + INV and OR2 = NOR2 + INV, simulated as
  complete gates. Widths (`config.GATE_W`): INV 2/4 Lmin (N/P), NAND2 4/4, NOR2 2/8.
  Input order is AB: in the NAND2, A drives the NMOS next to the output; in the NOR2, A drives the
  PMOS next to VDD.
* **Corners:** 200 draws of VDD, T, five NMOS and five PMOS process parameters (seed 2025), saved
  in `data/gates/corners.csv`. All NMOS in a gate share one draw and all PMOS another.
* **Leakage definition:** the current drawn from every source at VDD (the supply and the inputs
  that are high), which is the static power divided by VDD. Counting the high inputs makes gate
  leakages add up in a larger circuit, where a gate's input current comes from the gate driving it.
* **ML gate model:** the node voltages recorded from SPICE give each transistor's Vd, Vg, Vs, Vb;
  the Part A model (ON or OFF, chosen from the gate voltage) predicts its four terminal currents;
  leakage = Σ over transistors of (Vd·Id + Vg·Ig + Vs·Is + Vb·Ib) / VDD.
* **Also recorded:** the ML's own estimate of the internal node voltages (Kirchhoff's current law
  solved by bisection with the transistor models) and the leakage that follows from them. It
  gives the same accuracy (median 0.69 %, p95 2.07 %).
* **SPICE check:** a solution with a node voltage outside [0, VDD] is rerun with a different
  starting guess. 4 simulations needed this; 1 of 3,600 stayed unusable and is left out.

| Gate | Input | SPICE median (A) | ML median (A) | Bias % | Median error % | p95 error % |
|---|---|---|---|---|---|---|
| INV | 0 | 2.137e-11 | 2.133e-11 | 0.53 | 0.78 | 2.30 |
| INV | 1 | 4.278e-11 | 4.257e-11 | 0.26 | 0.68 | 2.60 |
| NAND2 | 00 | 2.522e-11 | 2.495e-11 | 0.56 | 0.78 | 2.52 |
| NAND2 | 01 | 9.406e-11 | 9.390e-11 | 0.67 | 0.74 | 1.77 |
| NAND2 | 10 | 2.432e-11 | 2.433e-11 | 0.30 | 0.60 | 2.19 |
| NAND2 | 11 | 1.330e-10 | 1.331e-10 | 0.66 | 0.87 | 2.54 |
| NOR2 | 00 | 7.204e-11 | 7.135e-11 | 0.40 | 0.73 | 2.29 |
| NOR2 | 01 | 9.522e-11 | 9.527e-11 | 0.16 | 0.70 | 2.25 |
| NOR2 | 10 | 4.314e-11 | 4.301e-11 | 0.55 | 0.69 | 1.78 |
| NOR2 | 11 | 4.456e-11 | 4.446e-11 | 0.45 | 0.65 | 1.64 |
| AND2 | 00 | 8.342e-11 | 8.391e-11 | 0.43 | 0.72 | 1.91 |
| AND2 | 01 | 1.405e-10 | 1.403e-10 | 0.51 | 0.67 | 1.73 |
| AND2 | 10 | 7.142e-11 | 7.199e-11 | 0.36 | 0.64 | 1.93 |
| AND2 | 11 | 1.718e-10 | 1.733e-10 | 0.66 | 0.82 | 2.05 |
| OR2 | 00 | 1.257e-10 | 1.261e-10 | 0.33 | 0.65 | 1.86 |
| OR2 | 01 | 1.162e-10 | 1.164e-10 | 0.14 | 0.69 | 2.01 |
| OR2 | 10 | 6.903e-11 | 6.955e-11 | 0.54 | 0.60 | 1.42 |
| OR2 | 11 | 8.478e-11 | 8.464e-11 | 0.47 | 0.59 | 1.53 |
| **All** | | 6.964e-11 | 6.987e-11 | 0.46 | **0.69** | **2.07** |

Mean error 0.83 %, so the mean accuracy is **99.17 %**. Full table: `results/tables/gate_results.txt`
and `gate_results_table.csv`; parity plot: `results/plots/gate_parity_plot.png`.

## Part C: test circuit, 3-bit carry-lookahead adder (`05_circuit_validation.py`)

* **Circuit:** g_i = a_i·b_i, p_i = a_i ⊕ b_i, s_i = p_i ⊕ c_i, with the lookahead carries
  c1 = g0 + p0c0, c2 = g1 + p1g0 + p1p0c0, c3 = g2 + p2g1 + p2p1g0 + p2p1p0c0. It uses only Part B
  gates (an XOR is four NAND2): 24 NAND2, 12 AND2, 6 OR2 = 42 gates, 204 transistors.
* **SPICE:** the complete netlist, all 128 input vectors at the same 200 corners (25,600 operating
  points; 1 did not converge). The sum and carry outputs are logically correct in every one.
* **ML:** for each input vector, logic simulation gives the inputs of every gate; the Part B gate
  model is called for each of the 42 gates and the leakages are summed.

| | SPICE median (A) | ML median (A) | Bias % | Median error % | p95 error % | Mean error % |
|---|---|---|---|---|---|---|
| All 25,491 points | 4.517e-09 | 4.532e-09 | 0.60 | **0.73** | **1.79** | 0.80 |

Mean accuracy **99.20 %**. Per input vector the median error is between 0.69 % and 0.77 %
(`results/tables/circuit_results_table.csv` has all 128 rows). Parity plot:
`results/plots/circuit_parity_plot.png`.

**Composition check:** adding the SPICE leakages of the standalone gates reproduces the SPICE
leakage of the whole circuit to 0.002 % (median). So ignoring the loading between gates costs
nothing here, and the circuit error is the transistor-model error carried through.

## Notes for the report
* **The physics equations did most of the work.** The ML corrections bring every current that
  matters for leakage to about 0.5 % median error.
* **Gate leakage is mostly ON-transistor gate tunnelling** in this low-power technology (for the
  INV at one corner: 3.7e-11 A of tunnelling against 5e-12 A of subthreshold current). The
  ON-state Ig model therefore sets the gate and circuit accuracy.
* **Largest errors:** hot, high-VDD corners (corner 74: 113 °C, 1.098 V, about 4–6 %), and the
  OFF-state gate current (~1e-22 A), which doesn't affect leakage.
* **Errors don't average out in the circuit:** all 42 gates share the same corner, so the same
  transistor-model error appears in every gate. The circuit bias (+0.60 %) is close to the gate
  bias (+0.46 %).
* **Drain and source aren't symmetric in this model card.** The drain junction has a fixed 1e-14 A
  saturation current; the source junction's is much smaller and strongly temperature dependent.
* **Saved models and plots:** the models are in `ml_models/<CASE>.joblib`; `ml_models/model_index.json`
  has the model type and test metrics for each case. The predicted-vs-SPICE plots are in
  `results/plots/pred_vs_spice_<CASE>.png`.
