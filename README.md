# TC-FedIDS: Trajectory-Aware Trust and Environment-Risk Aggregation for Heterogeneous Federated Intrusion Detection

Code, measured results, figures and tables accompanying the manuscript
**"TC-FedIDS: Trajectory-Aware Trust and Environment-Risk Aggregation for Heterogeneous Federated Intrusion Detection"**.

**Authors:** Saumya Mishra, Shailendra Mishra, Megha Rathi, Ahmad Raza Khan

## DOI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

> Placeholder: replace `10.5281/zenodo.XXXXXXX` with the DOI that Zenodo assigns after the release is archived.

## Overview

TC-FedIDS keeps a rolling history of each client's five most recent updates, derives four temporal descriptors
(direction persistence, update-norm entropy, second-order acceleration, update-norm variance), and fuses them with geometric
deviation, environment alignment and validation residuals into continuous trust weights. The weights act as soft coefficients
in a Huber-regularised GroupDRO objective solved with bounded FISTA. No client is removed solely for a low trust score.

Setup: 28 clients (7 per environment) over four NetFlow environments (NF-CSE-CIC-IDS2018, NF-ToN-IoT, UNSW-NB15, CIDDS-001),
Dirichlet non-IID partitioning (alpha = 0.5), 30 rounds, 10 seeds (11-20), regimes: clean, ALIE, sign-flip and Gaussian noise
(30% Byzantine clients for the attacked regimes).

## Repository structure

```
TC-FedIDS/
├── README.md
├── requirements.txt
├── CITATION.cff
├── .zenodo.json
├── code/
│   ├── tc_fedids_experiment.py   # real-data federation benchmark (prepare / run / analyze / plan)
│   └── tc_fedids.py              # architectural validation and aggregation simulation
├── results/                      # measured outputs of the benchmark
│   ├── all_rounds.csv            # every round, trial, environment (43,200 rows)
│   ├── final_per_seed.csv        # round-30 results per seed (1,440 rows)
│   ├── summary_by_environment.csv
│   ├── summary_equal_environment.csv
│   ├── timing_summary.csv
│   └── trial_timing.csv
├── figures/                      # manuscript figures
└── tables/                       # manuscript tables as CSV + all_tables.xlsx
```

## Figures

| Manuscript | File | Content |
|---|---|---|
| Fig. 1 | `figures/fig1_architecture.jpeg` | TC-FedIDS architecture |
| Fig. 2 | `figures/fig2_bacc_clean.png` | Balanced-accuracy trajectories, clean (beta = 0) |
| Fig. 3 | `figures/fig3_bacc_alie.png` | Balanced-accuracy trajectories, 30% ALIE |
| Fig. 4 | `figures/fig4_bacc_gaussian.png` | Balanced-accuracy trajectories, 30% Gaussian noise |
| Fig. 5 | `figures/fig5_bacc_signflip.png` | Balanced-accuracy trajectories, 30% sign-flip |
| Fig. 6(a) | `figures/fig6a_trust_separation.png` | Mean trust separation across attack scenarios |
| Fig. 6(b) | `figures/fig6b_alie_recall_vs_fraction.png` | ALIE attack recall vs malicious participation |

## Tables

| Manuscript | File | Caption |
|---|---|---|
| 1 | `tables/table01.csv` | Mechanism-level positioning of TC-FedIDS relative to representative federated intrusion-detection and Byzantine-robust approaches. |
| 2 | `tables/table02.csv` | Characteristics of the four network environments used in the experiments. |
| 3 | `tables/table03.csv` | Equal-environment final-round performance under the clean regime (mean ± SD, 10 seeds) |
| 4 | `tables/table04.csv` | Equal-environment final-round performance under 30 % ALIE (mean ± SD, 10 seeds) |
| 5 | `tables/table05.csv` | Equal-environment final-round performance under sign-flip and Gaussian noise (mean SD, 10 seeds). |
| 6 | `tables/table06.csv` | TC-FedIDS equal-environment performance across ALIE participation rates (mean, 10 seeds). |
| 7 | `tables/table07.csv` | Mean wall-clock time per 30-round trial and per-round aggregation cost (CUDA). |
| 8 | `tables/table08.csv` | Component ablation under 30 % ALIE (equal-environment attack-class recall). |
| 9 | `tables/table09.csv` | Summary of per-environment balanced-accuracy behaviour across threat regimes. |
| 10 | `tables/table10.csv` | Final-round test-set composition per environment. |
| 11 | `tables/table11.csv` | Per-environment final-round balanced accuracy (mean over 10 seeds); Worst is the minimum over four environments. |
| 12 | `tables/table12.csv` | Per-environment final-round attack-class recall (mean over 10 seeds). |
| 13 | `tables/table13.csv` | Paired difference in equal-environment BAcc, TC-FedIDS minus baseline (mean difference; p-value). |
| 14 | `tables/table14.csv` | Change in equal-environment BAcc and false-positive rate. |
| 15 | `tables/table15.csv` | Mean TC-FedIDS trust weight over 30 rounds and 10 seeds (lower bound τ=0.35). |
| 16 | `tables/table16.csv` | Mechanism-level positioning of TC-FedIDS against representative recent federated IDS and Byzantine-robust learning frameworks. |

All sixteen tables are also in `tables/all_tables.xlsx` (one sheet per table).

## Reproducing the experiments

```bash
pip install -r requirements.txt
# 1. Place the four NetFlow/IDS datasets under ./Dataset (see manuscript Section 4.1 and dataset references)
python code/tc_fedids_experiment.py prepare --data-dir Dataset
python code/tc_fedids_experiment.py plan    --suite full
python code/tc_fedids_experiment.py run     --suite full --seeds 11 12 13 14 15 16 17 18 19 20 --rounds 30
python code/tc_fedids_experiment.py analyze --plots
```

Raw datasets are not redistributed here; obtain them from their original sources
(Sarhan et al. NetFlow collection; UNSW-NB15; CIDDS-001). The experiment script's docstring refers to a run guide
(`TC_FedIDS_RUN_GUIDE.md`) that describes the feature, split and label protocol.

`code/tc_fedids.py` is an architectural validation / aggregation simulation. As stated in its own header, its manuscript
tables and figures are fixed supplied values rather than computed results; the **measured results** in this repository come
from `code/tc_fedids_experiment.py` and are stored in `results/`.

## Citation

See `CITATION.cff`, or cite the Zenodo record once the DOI is issued.
