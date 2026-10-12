# E2SN — Echo-Euler State Network

Code and experimental results for a Bachelor's thesis in Computer Science at the University of Pisa (2026).

**Supervisor:** Prof. Claudio Gallicchio

## Overview

This project investigates heterogeneous reservoir computing architectures for time-series processing, combining:

- **Echo State Networks (ESNs):** leaky recurrent dynamics.
- **Euler State Networks (EuSNs):** recurrent dynamics based on Euler discretization and antisymmetric weight matrices.

The goal is to investigate how different recurrent dynamics can support the processing of temporal dependencies at multiple timescales.

## Architectures

The experiments consider:

- **ESN:** standard leaky reservoir.
- **EuSN:** Euler-based reservoir.
- **Hybrid E2SN:** heterogeneous reservoir containing ESN and EuSN units.
- **Coupled E2SN:** interacting ESN and EuSN reservoirs with directional coupling.
- **CoupledSymmetric E2SN:** coupled reservoirs with shared coupling strength.
- **Mixed E2SN:** combination of ESN and EuSN state updates.

Additional ESN baselines with increased reservoir dimensions are included for comparison.

## Repository structure

```text
E2SN/
├── data/                    # Generated datasets (not tracked)
├── src/
│   ├── reservoir.py         # Reservoir architectures
│   ├── memory_tasks.py      # Synthetic task generators
│   ├── prepare_datasets.py  # Dataset preparation
│   ├── run_experiments.py   # Real-world datasets
│   ├── run_memory_experiments.py
│   ├── run_synthetic_experiments.py
│   ├── run_shortlong_multiseed_all.py
│   ├── test_coupled_multiseed_shortlong.py
│   └── run_mixed_baselines.py
└── results/
    ├── datasets/
    ├── memory_capacity/
    ├── synthetic/
    └── short_long/
```

## Environment

The experiments were developed using Python 3.9.12.

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

The original experiments were run in a GPU-enabled environment. TensorFlow and GPU availability may vary across operating systems.

## Datasets

The real-world classification experiments use:

- FordA
- FordB
- Adiac
- Libras
- RacketSports

The datasets can be prepared using:

```bash
python -m src.prepare_datasets
```

The preparation script reconstructs stratified training, validation and test splits and saves the processed datasets locally in `data/`.

## Experimental tasks

**Memory Capacity:** evaluates the ability of reservoir states to reconstruct delayed inputs.

**Synthetic Long-Term Memorization:** evaluates retention of information presented at the beginning of a sequence under increasing padding lengths.

**Short+Long:** evaluates the joint processing of remote contextual information and recent temporal features.

**Real-world classification:** evaluates the architectures on five time-series classification datasets.

## Experimental protocol

Hyperparameters are selected using validation data. Final evaluation is performed on a separate test set.

Final results are reported across five reservoir initializations using mean and standard deviation.

Model selection is repeated for each padding length in the synthetic tasks. The Short+Long experiments use multiple reservoir initializations during model selection.

## Results

The `results/` directory contains compact CSV summaries of the final experiments:

- `datasets/summary.csv`
- `memory_capacity/summary.csv`
- `memory_capacity/per_delay.csv`
- `synthetic/summary.csv`
- `short_long/summary.csv`

These files contain aggregated results or per-seed measurements extracted from the original experimental outputs.

## Running experiments

The scripts in `src/` are the original research experiment drivers. Some retain model-specific entry points, historical configurations, and `DRY_RUN` settings.

Review the configuration at the bottom of each script before running a full experiment. The scripts have not yet been consolidated into a single command-line interface.

Running full model selection can be computationally expensive and may require a GPU-enabled environment.

## Acknowledgments

The ESN and EuSN implementations build on ideas and code from the Euler State Networks research project. Relevant upstream attribution and licensing information should be preserved when redistributing adapted code.
