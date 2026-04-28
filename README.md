# Reproducibility Package for IQEA-Based Multi-Objective Welding Line Balancing
1. Overview
This repository contains the code, input data, and result files used to reproduce the experiments reported in the manuscript.
The repository supports reproduction of the following parts of the paper:
ablation study (Table 4)
benchmark verification on public ALBP instances (Table 5)
multi-objective comparison against NSGA-II and MOPSO (Table 7)
statistical significance tests using Wilcoxon rank-sum test with Holm correction (Table 8)
sensitivity analyses for key IQEA parameters
2. Input Data
Public benchmark instances:data/benchmark_instances/
Welding-line case study:data/welding_case/
3. Random Seeds：data/seeds.json
4. How to Reproduce the Main Results
python src/Benchmark/verification.py——————This reproduces the benchmark comparison on public ALBP instances and outputs CT and LoadSTD for the tested cases.
python src/iqea/run_iqea_comparison.py
python src/mopso/run_mopso_comparison.py
python src/nsga-ii/run_nsgaii_comparison.py——————These scripts generate the final Pareto fronts across 20 independent runs for IQEA, MOPSO, and NSGA-II.
python src/iqea/compute_hv_igd.py——————This script merges the Pareto solutions from all algorithms and all runs, performs nondominated filtering to obtain the empirical reference front, computes run-level HV and IGD and take the Wilcoxon rank-sum test and Holm correction for HV and IGD.
The resulting summary file is:data/hv_igd_mean_std.csv
The corresponding reference front is stored in:data/reference_front.csv
python src/sensitivity/Ablation_experiment.py——————This reproduces the ablation study including FULL, Abl-1, Abl-2, Abl-3, and Abl-4, and outputs the metrics.
src/sensitivity/
These reproduce the parameter sensitivity analyses reported in the manuscript, including:
initial rotation angle
forward rotation probability
mutation probability
mutation mode
polarization threshold
epsilon-dominance parameters
normalized quality-loss parameters
5. Contact
For any questions regarding reproduction, please contact:
Email：yuwang202303@163.com
