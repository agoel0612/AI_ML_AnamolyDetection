# FCA-PAD · Financial Consequence-Aware Process Anomaly Detection

Companion code and results for *Strange Is Not Costly: Financial Consequence-Aware Process Anomaly Detection under a Review Budget* by Krrish Yadav and Aditya Goel.

The experiments use four real BPI Challenge 2020 event logs with **synthetically injected anomalies**. Costly cases include duplicate payments, approval bypasses and other financial violations. Harmless decoys include rework, reordered approvals and delays. The financial layer combines audit-rule evidence, learned weights, the amount at stake and a consequence grade into an expected-loss ranking. The experiments test how ranking by process unusualness compares with ranking by financial consequence when an auditor can review only 50 cases.

## Repository layout

Clone or download this repository and open its root directory. The Python notebooks sit alongside this README. The [results folder](results/) contains the extracted experiment assets and saved outputs.

```text
repository/
├── README.md
├── FCA_PAD_*.ipynb                  # Seven Python notebooks
└── results/
    └── FCA_PAD_v2/                  # FCA_ROOT and the Domestic log directory
        ├── 00_frozen/              # Domestic frozen splits and ground truth
        ├── 01_raw/                 # Domestic XES event log
        ├── 02_processed/           # Domestic processed event tables
        ├── 03_features/            # Domestic features and scores
        ├── 04_models/              # Domestic saved models
        ├── 05_results/             # Domestic v2 tables
        ├── 06_figures/             # Domestic v2 plots
        ├── BPIC20_I/               # International
        ├── BPIC20_PR/              # Prepaid Travel Costs
        ├── BPIC20_R/               # Request for Payment
        ├── 16_gama_vs_fcapad/       # Cross-log benchmark comparison
        ├── *_summary/              # Cross-log summaries for v4–v8
        └── 23_hypothesis_tests/     # Consolidated tests and supporting outputs
```

The **Domestic** log uses `results/FCA_PAD_v2/` directly. The other logs have subdirectories inside it. The notebooks also use this shared directory for cross-log outputs, so keep its internal structure intact.

| Log ID | Dataset | XES under the log's `01_raw/` |
|---|---|---|
| `BPIC20_D` | Domestic Declarations | `DomesticDeclarations.xes` |
| `BPIC20_I` | International Declarations | `BPIC20_InternationalDeclarations.xes` |
| `BPIC20_PR` | Prepaid Travel Costs | `BPIC20_PrepaidTravelCost.xes` |
| `BPIC20_R` | Request for Payment | `BPIC20_RequestForPayment.xes` |

## Run the code

### 1. Start Jupyter or Colab

For local Jupyter, open a terminal at the repository root, create a virtual environment, and start Jupyter from that directory. The saved environments record Linux/Colab, Python 3.13.15 or 3.13.16, pandas 2.2.3 and NumPy 2.1.3 or 2.4.6 depending on the log. The bundle has no dependency lockfile. Use an interpreter supported by your TensorFlow and PyTorch builds for detector runs.

```bash
python -m venv .venv
source .venv/bin/activate           # bash / macOS / Linux
# Windows PowerShell uses: .\.venv\Scripts\Activate.ps1
python -m pip install jupyterlab numpy pandas scipy scikit-learn pyarrow matplotlib
python -m jupyterlab
```

The packages above are enough for the **hypothesis notebook**. For experiment notebooks, also install the relevant model packages into the same environment:

```bash
python -m pip install tensorflow joblib pm4py "pulp<4"
python -m pip install torch torch_geometric lxml networkx tqdm psutil
```

The first extra line supports v4–v8. The second supports BPAD-backed runs (v2 comparison and v5–v8). These runs also require Git and access to the upstream BPAD repository, or a local BPAD checkout. The notebooks can clone it, patch compatibility issues and write benchmark datasets into it, so point `BPAD_DIR` at a working copy. A GPU is helpful for training. If the setup cell downgrades PuLP after it was imported, restart the kernel.

In **Colab**, open a notebook from GitHub and place a copy of the repository in Drive so the runtime can access the assets. Authorize the Drive mount and set `FCA_ROOT` to the Drive path shown below.

### 2. Set the root and select the run

Add this cell **before the setup and configuration cells** in each notebook you run. The notebooks’ built-in defaults look for `FCA_PAD_v2/` directly, so this override points them to the repo’s `results/FCA_PAD_v2/` directory. Set the shared root for every log.

```python
import os
from pathlib import Path

# Local Jupyter started at the repository root
os.environ["FCA_ROOT"] = str(Path("results/FCA_PAD_v2").resolve())
# Colab with a copy of the repository in Drive
# os.environ["FCA_ROOT"] = "/content/drive/MyDrive/<repository>/results/FCA_PAD_v2"
os.environ["FCA_LOG"] = "BPIC20_D"  # v4–v8 select one log per run
# os.environ["GW_LOGS"] = "BPIC20_D,BPIC20_I"  # v2 comparison only
# os.environ["BPAD_DIR"] = r"/absolute/path/to/BPAD"
```

- **Inspect saved results:** Open the CSVs and PNGs directly. No notebook execution is needed.
- **Recompute the paper's hypothesis table from the archive:** Run every cell in `FCA_PAD_hypothesis_tests_(3).ipynb`. It reads per-log results across v5–v8 and uses saved v6 Parquet inputs and scores to recompute oracle F1 for H1. It writes to `results/FCA_PAD_v2/23_hypothesis_tests/` and does no detector training. Missing v8 inputs for PR and R are expected for this archive.
- **Rerun experiments:** Start from the committed assets in `results/FCA_PAD_v2/`. Use a separate working copy for reruns if you want to preserve the saved CSVs, figures and caches for comparison. Run the v2 comparison or v4 independently as needed. For the main paper workflow, run v5, v6, v7 and v8, then the hypothesis notebook. Run v5 before v6 if you want v6's clean-versus-polluted comparison. For v4–v8, run cells from the top in a fresh kernel for each selected log. The final cross-log cell aggregates available on-disk results, including previously saved runs. Run it after the intended per-log runs have finished.
- **Match the archived setup:** Set the log explicitly. The supplied v7 notebook defaults to R and v8 defaults to PR, while the archived v8 results are for D and I. Keep the standard test seeds (404, 505, 606 for v4–v7 and 707, 808, 909 for v8), default training epochs (20 for GAMA, 100 for benchmark DAE where configured), and budget of 50 review slots. Leave `FCA_FAST` and `FCA_SKIP_ROBUSTNESS` unset for full runs. Keep the frozen inputs intact. Many cells reuse saved intermediates or model scores when present.

The standalone original v2 training notebook is absent from the bundle. The v2 comparison reads the supplied processed artifacts and frozen test split. The Part A cells in v4–v8 check for saved v2 models, splits and features before proceeding. If they report missing files, check `FCA_ROOT` and confirm the expected files are present in your checkout.

## Notebook guide

Paths under **Per-log output** are relative to each log directory. Summary paths are relative to `FCA_ROOT`.

| Notebook | What it implements | Per-log output / cross-log summary | Archived coverage |
|---|---|---|---|
| [`FCA_PAD_v2_gama_vs_fcapad_multi_(1).ipynb`](FCA_PAD_v2_gama_vs_fcapad_multi_%281%29.ipynb) | Compares the original FCA-PAD fusion and anomaly lane with GAMA alone and GAMA plus the financial layer on frozen test cases. Runs available logs in one notebook. WAKE is optional in the code and disabled by default. | `14_gama_wake/` / `16_gama_vs_fcapad/` | D, I, PR, R |
| [`FCA_PAD_v4_realistic_anomalies.ipynb`](FCA_PAD_v4_realistic_anomalies.ipynb) | Generates realistic worlds containing costly anomalies and harmless decoys, sweeps decoy ratios and compares anomaly ranking, amount ranking, v2 fusion and financial-layer variants. Reports recovered euros and review-cost-adjusted benefit. | `18_v4_realistic/` / `18_v4_summary/` | D |
| [`FCA_PAD_v5_detectors_and_layer.ipynb`](FCA_PAD_v5_detectors_and_layer.ipynb) | Clean-training comparison of a next-event LSTM, GAMA (graph-based anomaly detector) and DAE (denoising autoencoder). Tests detector-only rankings, amount weighting, rules v1/v3, exploration and leave-one-type-out sensitivity. | `19_v5_detectors/` / `19_v5_summary/` | D, I, PR, R |
| [`FCA_PAD_v6_polluted_training.ipynb`](FCA_PAD_v6_polluted_training.ipynb) | Injects anomalies before a stratified 60/20/20 split. Retrains all three detectors on contaminated training data without anomaly labels, then evaluates them and the financial layer on held-out cases. Exports clean-versus-polluted comparisons. | `20_v6_polluted/` / `20_v6_summary/` | D, I, PR, R |
| [`FCA_PAD_v7_stress_tests (1).ipynb`](FCA_PAD_v7_stress_tests%20%281%29.ipynb) | Tests covered, rule-dodging, new and mixed fraud scenarios. Compares learned weights with equal weights and examines rule reliability, rule removal and exploration budgets. | `21_v7_stress/` / `21_v7_summary/` | D, I, PR, R |
| [`FCA_PAD_v8_two_lane (1).ipynb`](FCA_PAD_v8_two_lane%20%281%29.ipynb) | Tests a queue of 40 learned-rules × money cases plus 10 remaining cases ranked by GAMA on fresh seeds. Compares it with 50 slots ranked by learned rules × money, GAMA alone, blended and equal-weight alternatives. | `22_v8_two_lane/` / `22_v8_summary/` | D, I |
| [`FCA_PAD_hypothesis_tests_(3).ipynb`](FCA_PAD_hypothesis_tests_%283%29.ipynb) | Consolidates H1–H9 and E1 from v5–v8, including paired tests, bootstrap intervals, correlations, Holm adjustment and an equivalence test. | Shared `23_hypothesis_tests/` | Four logs for v5–v7 inputs, two for v8 inputs |

## Results archive map

| Path or file group | Contents |
|---|---|
| `00_frozen/` | Frozen case splits, sealed injected ground truth, original audit-rule registry and experiment preregistrations. |
| `01_raw/`, `02_processed/` | Original XES event logs and sorted or injected event tables in Parquet. |
| `03_features/` | Case skeleton, feature manifest, prefix features, case scores and next-event scores. |
| `04_models/` | Cached Keras LSTM/autoencoder models, Isolation Forest, scaler, calibration references and model metadata. |
| `05_results/`, `06_figures/` | v2 audit summaries, selected fusion, detector results, per-type performance, review-budget and early-warning analyses, natural-case review outputs and plots. Start with `07_headline_test.csv` and `12_summary.json`. |
| `14_gama_wake/` | Per-log GAMA scores, case ordering and test comparison. |
| `18_v4_realistic/` through `22_v8_two_lane/` | Per-log seeded worlds and detector scores where saved, plus detailed metrics, deltas, ablations and plots. The `_v2_core_outputs/` and `_v2_core_figures/` folders hold intermediate outputs from the shared v2 setup. |
| `*_summary/` | Cross-log tables. `metrics_by_seed_ratio.csv` lives in per-log experiment folders. `scenario_metrics_all_logs.csv` and headline pivot tables are in the relevant summary folders. |
| `23_hypothesis_tests/` | `hypothesis_tests.csv` has inference results, `H1_f1_vs_money.csv` and its PNG show the benchmark-to-money gap, and `H6_weights_vs_precision.csv` contains the rule-level correlation inputs. |

## Observations from the saved runs

**LossRecall@50 / MoneyRecall@50** is the fraction of injected monetary loss captured by the 50 highest-ranked cases in the evaluation pool. **Costly-case recall@50** counts costly cases equally. **Decoys in top 50** counts harmless cases occupying review slots. **Net benefit** subtracts review costs from recovered loss. AP and oracle F1 measure detection of any injected anomaly, including decoys. Oracle F1 uses test labels to pick its best threshold, so it is a retrospective benchmark. The v4–v6 headline comparison uses the 10:1 decoy-to-costly setting. Treat the v2, realistic-world and stress-test results as separate evaluations because the data generation and training protocols differ.

| Experiment | Observation | Source under `FCA_ROOT` |
|---|---|---|
| v2 frozen benchmark | Mean LossRecall@50 across four logs was **0.919** for full FCA-PAD, **0.912** for GAMA with the financial layer and **0.552** for GAMA alone. The benchmark GAMA run used the whole log including injected validation/test cases, while FCA-PAD used clean training cases. Read the accompanying protocol note when comparing these numbers. | [`16_gama_vs_fcapad/LossRecallat50_by_log.csv`](results/FCA_PAD_v2/16_gama_vs_fcapad/LossRecallat50_by_log.csv), [`protocol_note.txt`](results/FCA_PAD_v2/16_gama_vs_fcapad/protocol_note.txt) |
| v4 realistic mix | On Domestic, the v4 financial layer with audit tests recovered **82.9%** of loss versus **28.0%** for v2 fusion. At €25 per case the corresponding net benefits were approximately **€2,257** and **−€83**. | [`18_v4_summary/LossRecallat50_by_log.csv`](results/FCA_PAD_v2/18_v4_summary/LossRecallat50_by_log.csv), [`net_EUR_c25_by_log.csv`](results/FCA_PAD_v2/18_v4_summary/net_EUR_c25_by_log.csv) |
| v5 clean training | On International, GAMA alone scored **0.935 AP** for any anomaly and **0.000 LossRecall@50**. The v3 financial layer raised GAMA's monetary recall there to **0.998**. | [`19_v5_summary/AP_any_anomaly_by_log.csv`](results/FCA_PAD_v2/19_v5_summary/AP_any_anomaly_by_log.csv), [`LossRecallat50_by_log.csv`](results/FCA_PAD_v2/19_v5_summary/LossRecallat50_by_log.csv) |
| v6 polluted training | Learned rules × money recovered **0.818 / 0.936 / 0.996 / 0.974** across D/I/PR/R. GAMA alone recovered **0.451 / 0.054 / 0.196 / 0.645** in that order. The paired gain from adding the v3 financial layer to the detectors averaged **0.580**. | [`20_v6_summary/LossRecallat50_by_log.csv`](results/FCA_PAD_v2/20_v6_summary/LossRecallat50_by_log.csv), [`23_hypothesis_tests/hypothesis_tests.csv`](results/FCA_PAD_v2/23_hypothesis_tests/hypothesis_tests.csv) (H7) |
| v7 stress test | Learned rules × money recovered **0.854** on covered fraud, **0.099** on rule-dodging fraud, **0.171** on new fraud, and **0.631** in the mixed world. LSTM alone reached **0.528** on rule-dodging fraud. Amount alone reached **0.384** on new fraud. | [`21_v7_summary/money_recall_by_scenario.csv`](results/FCA_PAD_v2/21_v7_summary/money_recall_by_scenario.csv) |
| v8 fresh seeds | The rules-only queue recovered **0.935** on covered fraud and **0.568** in the mixed world. The two-lane queue recovered **0.905** and **0.562**. On new fraud, the two-lane queue recovered **0.126** versus **0.088** with rules alone. The paper selects learned rules × money for all 50 slots. | [`22_v8_summary/MoneyRecallat50_by_scenario.csv`](results/FCA_PAD_v2/22_v8_summary/MoneyRecallat50_by_scenario.csv), paper Table 5 |

### Hypothesis results

The consolidated notebook applies one-sided paired Wilcoxon tests and 10,000-resample bootstrap intervals for paired effects. It uses Spearman correlation for H6 and Holm adjustment across the 12 primary tests. The table below rounds the saved outputs. A positive monetary recall difference of 0.10 is an increase of 10 percentage points.

| Test | Saved mean effect or correlation | Pairs or points | Holm-adjusted p |
|---|---|---|---|
| H1 · Oracle F1 minus MoneyRecall@50 | 0.323 [0.252, 0.393] | 36 pairs | <0.001 |
| H2 · Extra decoys from detector-only ranking | 8.50 [6.75, 10.31] | 36 pairs | <0.001 |
| H3 · Layer gain, clean training | 0.687 [0.595, 0.778] | 36 pairs | <0.001 |
| H4 · Gain from audit tests F11–F15 | 0.297 [0.256, 0.338] | 36 pairs | <0.001 |
| H5a/H5b · Learned vs equal weights, covered / mixed | 0.289 [0.106, 0.504] / 0.152 [0.072, 0.235] | 6 pairs each | 0.09375 each |
| H6 · Rule weight vs rule precision | Spearman ρ = 0.865 | 34 log–rule points | <0.001 |
| H7 · Layer gain, polluted training | 0.580 [0.482, 0.673] | 36 pairs | <0.001 |
| H8a/H8b · Final rules-only system vs GAMA, covered / mixed | 0.667 [0.431, 0.881] / 0.308 [0.083, 0.550] | 6 pairs each | 0.09375 each |
| H9a/H9b · Covered minus rule-dodging / new fraud | 0.732 [0.600, 0.847] / 0.847 [0.782, 0.914] | 6 pairs each | 0.09375 each |

The intervals are 95% bootstrap intervals. H1–H4, H6 and H7 met the adjusted 0.05 threshold. H5, H8 and H9 had positive estimated effects and adjusted p-values above 0.05. H1's descriptive F1–money correlation was ρ = 0.415 (two-sided p = 0.012). E1 did not establish equivalence for the two-lane queue within ±0.05 (covered p = 0.219, mixed p = 0.078). The [`22_v8_summary/preregistered_tests.csv`](results/FCA_PAD_v2/22_v8_summary/preregistered_tests.csv) file contains raw one-sided tests for v8's own H28–H34 comparisons. Use [`23_hypothesis_tests/hypothesis_tests.csv`](results/FCA_PAD_v2/23_hypothesis_tests/hypothesis_tests.csv) for the consolidated paper claim tests.

The results apply to simulated losses on real expense and travel logs. Several audit rules were designed around the injected fraud types, and learned weights require labelled outcomes for calibration. Performance on rule-dodging and new fraud is therefore important when interpreting the results, particularly with the final fresh-seed evaluation limited to two logs.
