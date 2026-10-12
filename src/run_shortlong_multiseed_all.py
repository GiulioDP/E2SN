#!/usr/bin/env python3
"""
Short+Long multi-seed model selection for all remaining architectures.

Default full protocol:
- paddings: 100, 200, ..., 1000
- 400 hyperparameter configurations per padding
- model-selection seeds: 42, 43, 44, 45, 46
- final seeds: 1001, 1002, 1003, 1004, 1005
- same Short+Long task seed: 42
- Ridge alpha grid: 1e-5 ... 1e5 (11 values)

Architectures:
- ESN
- EuSN
- Hybrid
- CoupledSymmetric
- Mixed
- ESN_2N and ESN_2N_to_N (paired worker, same 2N reservoir states per trial/seed)

Coupled is intentionally NOT rerun: its 5-seed experiment is already complete.

Parallel execution:
- master launches independent workers on the same selected GPU
- default MAX_PARALLEL=2 to reduce OOM risk on a 16 GB V100
- each worker has progressive checkpoints and resume support

Run from repository root, e.g.:
    tmux new -s shortlong_multi
    python -u src/run_shortlong_multiseed_all.py --gpu 3 --max-parallel 2

Detach tmux:
    Ctrl-b d

Reattach:
    tmux attach -t shortlong_multi
"""

import argparse
import csv
import gc
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

SEARCH_SEED = 42
MODEL_SELECTION_SEEDS = [42, 43, 44, 45, 46]
FINAL_SEEDS = [1001, 1002, 1003, 1004, 1005]

FULL_PADDINGS = list(range(100, 1001, 100))
FULL_TRIALS = 400

DRY_PADDINGS = [700]
DRY_TRIALS = 10
DRY_FINAL_SEEDS = [1001]

CHECKPOINT_EVERY = 10
SNAPSHOT_EVERY = 50
PROGRESS_EVERY = 25

MODEL_QUEUE = [
    "ESN",
    "EuSN",
    "Hybrid",
    "CoupledSymmetric",
    "Mixed",
    "ESN2NPair",
]

RESULT_DIR_NAMES = {
    "ESN": "ESNMultiSeed",
    "EuSN": "EuSNMultiSeed",
    "Hybrid": "HybridMultiSeed",
    "CoupledSymmetric": "CoupledSymmetricMultiSeed",
    "Mixed": "MixedMultiSeed",
    "ESN2NPair": "ESN2NBaselinesMultiSeed",
}

# ML globals: imported only inside worker processes.
np = None
tf = None
Ridge = None
accuracy_score = None
ESN = None
EuSN = None
HybridReservoir = None
CoupledSymmetricReservoir = None
MixedDynamicsReservoir = None
generate_contextual_short_long_task = None


# ---------------------------------------------------------------------
# Worker-only imports
# ---------------------------------------------------------------------

def load_ml_stack():
    global np, tf, Ridge, accuracy_score
    global ESN, EuSN, HybridReservoir
    global CoupledSymmetricReservoir, MixedDynamicsReservoir
    global generate_contextual_short_long_task

    import numpy as _np
    import tensorflow as _tf
    from sklearn.linear_model import Ridge as _Ridge
    from sklearn.metrics import accuracy_score as _accuracy_score

    from src.memory_tasks import generate_contextual_short_long_task as _gen_task
    from src.reservoir import (
        ESN as _ESN,
        EuSN as _EuSN,
        HybridReservoir as _HybridReservoir,
        CoupledSymmetricReservoir as _CoupledSymmetricReservoir,
        MixedDynamicsReservoir as _MixedDynamicsReservoir,
    )

    np = _np
    tf = _tf
    Ridge = _Ridge
    accuracy_score = _accuracy_score
    ESN = _ESN
    EuSN = _EuSN
    HybridReservoir = _HybridReservoir
    CoupledSymmetricReservoir = _CoupledSymmetricReservoir
    MixedDynamicsReservoir = _MixedDynamicsReservoir
    generate_contextual_short_long_task = _gen_task

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


# ---------------------------------------------------------------------
# Small logging / saving utilities
# ---------------------------------------------------------------------

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def atomic_pickle_dump(obj, path):
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def atomic_json_dump(obj, path):
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def save_csv(rows, path, fieldnames):
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def worker_log(model_name, root, message):
    line = f"[{model_name}] {message}"
    print(line, flush=True)
    ensure_dir(root)
    with open(Path(root) / "run.log", "a", buffering=1) as f:
        f.write(line + "\n")


def write_status(root, model_name, **kwargs):
    payload = {
        "model": model_name,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **kwargs,
    }
    atomic_json_dump(payload, Path(root) / "status.json")


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------
# Ridge readout: same logic as Coupled multi-seed script
# ---------------------------------------------------------------------

ALPHA_VALUES = None


def train_ridge_readout(x_train, y_train, num_classes, alpha):
    y_one_hot = np.eye(
        num_classes,
        dtype=np.float64,
    )[y_train]

    readout = Ridge(
        alpha=alpha,
        solver="cholesky",
    )
    readout.fit(x_train, y_one_hot)
    return readout


def evaluate_readout(readout, states, labels):
    scores = readout.predict(states)
    predictions = np.argmax(scores, axis=1)
    return float(accuracy_score(labels, predictions))


def select_ridge_alpha(h_train, y_train, h_val, y_val, num_classes):
    best_alpha = None
    best_accuracy = -np.inf

    for alpha in ALPHA_VALUES:
        readout = train_ridge_readout(
            h_train,
            y_train,
            num_classes,
            alpha,
        )
        accuracy = evaluate_readout(
            readout,
            h_val,
            y_val,
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_alpha = alpha

    return float(best_alpha), float(best_accuracy)


def evaluate_binary_probe(h_train, y_train, h_val, y_val):
    _, acc = select_ridge_alpha(
        h_train,
        y_train,
        h_val,
        y_val,
        num_classes=2,
    )
    return float(acc)


def evaluate_binary_probe_test(
    h_train,
    y_train,
    h_val,
    y_val,
    h_test,
    y_test,
):
    best_alpha, _ = select_ridge_alpha(
        h_train,
        y_train,
        h_val,
        y_val,
        num_classes=2,
    )
    readout = train_ridge_readout(
        h_train,
        y_train,
        num_classes=2,
        alpha=best_alpha,
    )
    return evaluate_readout(
        readout,
        h_test,
        y_test,
    )


# ---------------------------------------------------------------------
# Hyperparameter spaces
# ---------------------------------------------------------------------

def sample_log_uniform(rng, low, high):
    return float(
        10 ** rng.uniform(
            np.log10(low),
            np.log10(high),
        )
    )


def sample_esn_config(rng, units=100):
    return {
        "units": int(units),
        "leaky": float(rng.uniform(0.01, 1.0)),
        "input_scaling": float(rng.uniform(0.01, 1.5)),
        "spectral_radius": float(rng.uniform(0.01, 1.5)),
        "bias_scaling": float(rng.uniform(0.01, 1.5)),
        "return_sequences": False,
    }


def sample_config(model_type, rng):
    if model_type == "ESN":
        return sample_esn_config(rng, units=100)

    if model_type == "EuSN":
        return {
            "units": 100,
            "gamma": sample_log_uniform(rng, 1e-5, 1e-1),
            "epsilon": sample_log_uniform(rng, 1e-5, 1e-1),
            "input_scaling": float(rng.uniform(0.01, 1.5)),
            "recurrent_scaling": float(rng.uniform(0.01, 1.5)),
            "bias_scaling": float(rng.uniform(0.01, 1.5)),
            "return_sequences": False,
        }

    if model_type == "Hybrid":
        return {
            "units": 100,
            "esn_fraction": float(
                rng.choice(
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                )
            ),
            "leaky": float(rng.uniform(0.01, 1.0)),
            "spectral_radius": float(rng.uniform(0.01, 1.5)),
            "gamma": sample_log_uniform(rng, 1e-5, 1e-1),
            "epsilon": sample_log_uniform(rng, 1e-5, 1e-1),
            "recurrent_scaling": float(rng.uniform(0.01, 1.5)),
            "input_scaling": float(rng.uniform(0.01, 1.5)),
            "bias_scaling": float(rng.uniform(0.01, 1.5)),
            "return_sequences": False,
        }

    if model_type == "Mixed":
        return {
            "units": 100,
            "mixing": float(
                rng.choice(
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                )
            ),
            "leaky": float(rng.uniform(0.01, 1.0)),
            "spectral_radius": float(rng.uniform(0.01, 1.5)),
            "gamma": sample_log_uniform(rng, 1e-5, 1e-1),
            "epsilon": sample_log_uniform(rng, 1e-5, 1e-1),
            "recurrent_scaling": float(rng.uniform(0.01, 1.5)),
            "input_scaling": float(rng.uniform(0.01, 1.5)),
            "bias_scaling": float(rng.uniform(0.01, 1.5)),
            "return_sequences": False,
        }

    if model_type == "ESN2NPair":
        return sample_esn_config(rng, units=200)

    raise ValueError(f"Unsupported model_type: {model_type}")


def generate_configurations(model_type, num_trials, search_seed):
    rng = np.random.default_rng(search_seed)

    # Keep the original stratified search for symmetric coupling.
    if model_type == "CoupledSymmetric":
        beta_values = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
        base_count = num_trials // len(beta_values)
        remainder = num_trials % len(beta_values)

        configurations = []

        for i, coupling_strength in enumerate(beta_values):
            count = base_count + (1 if i < remainder else 0)

            for _ in range(count):
                configurations.append({
                    "units": 100,
                    "coupling_strength": float(coupling_strength),
                    "leaky": float(rng.uniform(0.01, 1.0)),
                    "spectral_radius": float(rng.uniform(0.01, 1.5)),
                    "gamma": sample_log_uniform(rng, 1e-5, 1e-1),
                    "epsilon": sample_log_uniform(rng, 1e-5, 1e-1),
                    "recurrent_scaling": float(rng.uniform(0.01, 1.5)),
                    "input_scaling": float(rng.uniform(0.01, 1.5)),
                    "bias_scaling": float(rng.uniform(0.01, 1.5)),
                    "return_sequences": False,
                })

        rng.shuffle(configurations)
        return configurations

    return [
        sample_config(model_type, rng)
        for _ in range(num_trials)
    ]


# ---------------------------------------------------------------------
# Reservoirs / state extraction
# ---------------------------------------------------------------------

def build_reservoir(model_type, model_params):
    if model_type == "ESN":
        return ESN(**model_params)

    if model_type == "EuSN":
        return EuSN(**model_params)

    if model_type == "Hybrid":
        return HybridReservoir(**model_params)

    if model_type == "CoupledSymmetric":
        return CoupledSymmetricReservoir(**model_params)

    if model_type == "Mixed":
        return MixedDynamicsReservoir(**model_params)

    if model_type == "ESN2NPair":
        return ESN(**model_params)

    raise ValueError(model_type)


def extract_final_states(model, x):
    states = model(x)
    return np.asarray(states, dtype=np.float64)


def cleanup_trial(*objects):
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    tf.keras.backend.clear_session()
    gc.collect()


# ---------------------------------------------------------------------
# Scoring one state representation
# ---------------------------------------------------------------------

def score_validation_view(task, h_train, h_val):
    best_alpha, val_acc = select_ridge_alpha(
        h_train,
        task["y_train"],
        h_val,
        task["y_val"],
        task["num_classes"],
    )

    context_acc = evaluate_binary_probe(
        h_train,
        task["context_train"],
        h_val,
        task["context_val"],
    )

    short_acc = evaluate_binary_probe(
        h_train,
        task["short_train"],
        h_val,
        task["short_val"],
    )

    return {
        "ridge_alpha": best_alpha,
        "validation_accuracy": val_acc,
        "context_accuracy": context_acc,
        "short_accuracy": short_acc,
    }


def score_test_view(task, h_train, h_val, h_test):
    best_alpha, val_acc = select_ridge_alpha(
        h_train,
        task["y_train"],
        h_val,
        task["y_val"],
        task["num_classes"],
    )

    readout = train_ridge_readout(
        h_train,
        task["y_train"],
        task["num_classes"],
        best_alpha,
    )

    test_acc = evaluate_readout(
        readout,
        h_test,
        task["y_test"],
    )

    context_test = evaluate_binary_probe_test(
        h_train,
        task["context_train"],
        h_val,
        task["context_val"],
        h_test,
        task["context_test"],
    )

    short_test = evaluate_binary_probe_test(
        h_train,
        task["short_train"],
        h_val,
        task["short_val"],
        h_test,
        task["short_test"],
    )

    return {
        "ridge_alpha": best_alpha,
        "validation_accuracy": val_acc,
        "test_accuracy": test_acc,
        "context_test_accuracy": context_test,
        "short_test_accuracy": short_test,
    }


# ---------------------------------------------------------------------
# Generic model-selection trial
# ---------------------------------------------------------------------

def run_seed_validation(model_type, task, model_params, seed):
    tf.keras.backend.clear_session()
    set_seed(seed)

    model = build_reservoir(model_type, model_params)

    h_train = extract_final_states(
        model,
        task["x_train"],
    )
    h_val = extract_final_states(
        model,
        task["x_val"],
    )

    result = score_validation_view(
        task,
        h_train,
        h_val,
    )
    result["seed"] = seed

    cleanup_trial(model, h_train, h_val)
    return result


def aggregate_seed_results(model_type, trial, params, seed_results):
    vals = [r["validation_accuracy"] for r in seed_results]
    context = [r["context_accuracy"] for r in seed_results]
    short = [r["short_accuracy"] for r in seed_results]

    return {
        "trial": trial,
        "model": model_type,
        "model_params": params,
        "selection_seeds": MODEL_SELECTION_SEEDS.copy(),
        "validation_accuracy": float(np.mean(vals)),
        "validation_std": float(np.std(vals)),
        "validation_accuracies": vals,
        "mean_context_accuracy": float(np.mean(context)),
        "std_context_accuracy": float(np.std(context)),
        "mean_short_accuracy": float(np.mean(short)),
        "std_short_accuracy": float(np.std(short)),
        "ridge_alphas": [r["ridge_alpha"] for r in seed_results],
    }


# ---------------------------------------------------------------------
# Generic model selection with progressive checkpoint / resume
# ---------------------------------------------------------------------

def model_selection_generic(
    model_type,
    task,
    configurations,
    root,
    padding,
    restart=False,
):
    pad_dir = Path(root) / f"long_{padding}"
    ensure_dir(pad_dir)

    checkpoint_path = pad_dir / "checkpoint_latest.pkl"
    snapshots_dir = pad_dir / "checkpoints"
    ensure_dir(snapshots_dir)

    best_result = None
    results = []
    start_trial = 1

    if checkpoint_path.exists() and not restart:
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        results = checkpoint["results"]
        best_result = checkpoint["best_result"]
        start_trial = checkpoint["completed_trials"] + 1

        worker_log(
            model_type,
            root,
            f"RESUME pad {padding} | trial {start_trial}/{len(configurations)}",
        )

    for trial in range(start_trial, len(configurations) + 1):
        params = configurations[trial - 1]

        seed_results = [
            run_seed_validation(
                model_type,
                task,
                params,
                seed,
            )
            for seed in MODEL_SELECTION_SEEDS
        ]

        result = aggregate_seed_results(
            model_type,
            trial,
            params,
            seed_results,
        )
        results.append(result)

        is_new_best = (
            best_result is None
            or result["validation_accuracy"]
            > best_result["validation_accuracy"]
        )

        if is_new_best:
            best_result = result
            worker_log(
                model_type,
                root,
                (
                    f"pad {padding} | NEW BEST {trial:03d}/{len(configurations)} | "
                    f"val {best_result['validation_accuracy']:.4f} "
                    f"+/- {best_result['validation_std']:.4f}"
                ),
            )

        if (
            trial % PROGRESS_EVERY == 0
            or trial == len(configurations)
        ):
            worker_log(
                model_type,
                root,
                (
                    f"pad {padding} | {trial:03d}/{len(configurations)} | "
                    f"best {best_result['validation_accuracy']:.4f} "
                    f"+/- {best_result['validation_std']:.4f}"
                ),
            )

        if (
            trial % CHECKPOINT_EVERY == 0
            or trial == len(configurations)
            or is_new_best
        ):
            checkpoint = {
                "model": model_type,
                "padding": padding,
                "completed_trials": trial,
                "best_result": best_result,
                "results": results,
            }
            atomic_pickle_dump(
                checkpoint,
                checkpoint_path,
            )

            write_status(
                root,
                model_type,
                phase="model_selection",
                padding=padding,
                trial=trial,
                total_trials=len(configurations),
                best_validation=best_result["validation_accuracy"],
                best_validation_std=best_result["validation_std"],
            )

        if trial % SNAPSHOT_EVERY == 0:
            atomic_pickle_dump(
                {
                    "model": model_type,
                    "padding": padding,
                    "completed_trials": trial,
                    "best_result": best_result,
                    "results": results,
                },
                snapshots_dir / f"checkpoint_trial_{trial:04d}.pkl",
            )

    return best_result, results


# ---------------------------------------------------------------------
# Final evaluation generic
# ---------------------------------------------------------------------

def final_evaluation_generic(
    model_type,
    task,
    best_result,
    final_seeds,
    root,
    padding,
):
    test_accs = []
    context_accs = []
    short_accs = []
    guesses = []

    worker_log(
        model_type,
        root,
        f"pad {padding} | FINAL x{len(final_seeds)}",
    )

    for i, seed in enumerate(final_seeds, start=1):
        tf.keras.backend.clear_session()
        set_seed(seed)

        model = build_reservoir(
            model_type,
            best_result["model_params"],
        )

        h_train = extract_final_states(model, task["x_train"])
        h_val = extract_final_states(model, task["x_val"])
        h_test = extract_final_states(model, task["x_test"])

        score = score_test_view(
            task,
            h_train,
            h_val,
            h_test,
        )

        test_accs.append(score["test_accuracy"])
        context_accs.append(score["context_test_accuracy"])
        short_accs.append(score["short_test_accuracy"])

        guesses.append({
            "guess": i,
            "seed": seed,
            **score,
        })

        worker_log(
            model_type,
            root,
            (
                f"pad {padding} | final {i}/{len(final_seeds)} | "
                f"test {score['test_accuracy']:.4f} | "
                f"val {score['validation_accuracy']:.4f}"
            ),
        )

        cleanup_trial(
            model,
            h_train,
            h_val,
            h_test,
        )

    return {
        "model": model_type,
        "mean_test_accuracy": float(np.mean(test_accs)),
        "std_test_accuracy": float(np.std(test_accs)),
        "mean_context_test_accuracy": float(np.mean(context_accs)),
        "std_context_test_accuracy": float(np.std(context_accs)),
        "mean_short_test_accuracy": float(np.mean(short_accs)),
        "std_short_test_accuracy": float(np.std(short_accs)),
        "guesses": guesses,
    }


# ---------------------------------------------------------------------
# ESN 2N paired model selection:
# same 200-state reservoir, two readout views
# ---------------------------------------------------------------------

def run_pair_seed_validation(task, model_params, seed):
    tf.keras.backend.clear_session()
    set_seed(seed)

    model = ESN(**model_params)

    h_train_2n = extract_final_states(
        model,
        task["x_train"],
    )
    h_val_2n = extract_final_states(
        model,
        task["x_val"],
    )

    full_score = score_validation_view(
        task,
        h_train_2n,
        h_val_2n,
    )

    first_n_score = score_validation_view(
        task,
        h_train_2n[:, :100],
        h_val_2n[:, :100],
    )

    full_score["seed"] = seed
    first_n_score["seed"] = seed

    cleanup_trial(
        model,
        h_train_2n,
        h_val_2n,
    )

    return {
        "ESN_2N": full_score,
        "ESN_2N_to_N": first_n_score,
    }


def aggregate_pair_branch(branch, trial, params, seed_results):
    selected = [
        r[branch]
        for r in seed_results
    ]

    vals = [r["validation_accuracy"] for r in selected]
    context = [r["context_accuracy"] for r in selected]
    short = [r["short_accuracy"] for r in selected]

    return {
        "trial": trial,
        "model": branch,
        "model_params": params,
        "selection_seeds": MODEL_SELECTION_SEEDS.copy(),
        "validation_accuracy": float(np.mean(vals)),
        "validation_std": float(np.std(vals)),
        "validation_accuracies": vals,
        "mean_context_accuracy": float(np.mean(context)),
        "std_context_accuracy": float(np.std(context)),
        "mean_short_accuracy": float(np.mean(short)),
        "std_short_accuracy": float(np.std(short)),
        "ridge_alphas": [r["ridge_alpha"] for r in selected],
    }


def model_selection_esn2n_pair(
    task,
    configurations,
    root,
    padding,
    restart=False,
):
    model_name = "ESN2NPair"
    pad_dir = Path(root) / f"long_{padding}"
    ensure_dir(pad_dir)

    checkpoint_path = pad_dir / "checkpoint_latest.pkl"
    snapshots_dir = pad_dir / "checkpoints"
    ensure_dir(snapshots_dir)

    best_full = None
    best_first_n = None
    results = []
    start_trial = 1

    if checkpoint_path.exists() and not restart:
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        results = checkpoint["results"]
        best_full = checkpoint["best_ESN_2N"]
        best_first_n = checkpoint["best_ESN_2N_to_N"]
        start_trial = checkpoint["completed_trials"] + 1

        worker_log(
            model_name,
            root,
            f"RESUME pad {padding} | trial {start_trial}/{len(configurations)}",
        )

    for trial in range(start_trial, len(configurations) + 1):
        params = configurations[trial - 1]

        seed_results = [
            run_pair_seed_validation(
                task,
                params,
                seed,
            )
            for seed in MODEL_SELECTION_SEEDS
        ]

        result_full = aggregate_pair_branch(
            "ESN_2N",
            trial,
            params,
            seed_results,
        )
        result_first_n = aggregate_pair_branch(
            "ESN_2N_to_N",
            trial,
            params,
            seed_results,
        )

        results.append({
            "trial": trial,
            "model_params": params,
            "ESN_2N": result_full,
            "ESN_2N_to_N": result_first_n,
        })

        new_full = (
            best_full is None
            or result_full["validation_accuracy"]
            > best_full["validation_accuracy"]
        )
        new_first = (
            best_first_n is None
            or result_first_n["validation_accuracy"]
            > best_first_n["validation_accuracy"]
        )

        if new_full:
            best_full = result_full
            worker_log(
                model_name,
                root,
                (
                    f"pad {padding} | NEW BEST ESN_2N {trial:03d}/{len(configurations)} | "
                    f"{best_full['validation_accuracy']:.4f} "
                    f"+/- {best_full['validation_std']:.4f}"
                ),
            )

        if new_first:
            best_first_n = result_first_n
            worker_log(
                model_name,
                root,
                (
                    f"pad {padding} | NEW BEST 2N->N {trial:03d}/{len(configurations)} | "
                    f"{best_first_n['validation_accuracy']:.4f} "
                    f"+/- {best_first_n['validation_std']:.4f}"
                ),
            )

        if (
            trial % PROGRESS_EVERY == 0
            or trial == len(configurations)
        ):
            worker_log(
                model_name,
                root,
                (
                    f"pad {padding} | {trial:03d}/{len(configurations)} | "
                    f"2N {best_full['validation_accuracy']:.4f} | "
                    f"2N->N {best_first_n['validation_accuracy']:.4f}"
                ),
            )

        if (
            trial % CHECKPOINT_EVERY == 0
            or trial == len(configurations)
            or new_full
            or new_first
        ):
            checkpoint = {
                "model": model_name,
                "padding": padding,
                "completed_trials": trial,
                "best_ESN_2N": best_full,
                "best_ESN_2N_to_N": best_first_n,
                "results": results,
            }
            atomic_pickle_dump(
                checkpoint,
                checkpoint_path,
            )

            write_status(
                root,
                model_name,
                phase="model_selection",
                padding=padding,
                trial=trial,
                total_trials=len(configurations),
                best_ESN_2N=best_full["validation_accuracy"],
                best_ESN_2N_to_N=best_first_n["validation_accuracy"],
            )

        if trial % SNAPSHOT_EVERY == 0:
            atomic_pickle_dump(
                {
                    "model": model_name,
                    "padding": padding,
                    "completed_trials": trial,
                    "best_ESN_2N": best_full,
                    "best_ESN_2N_to_N": best_first_n,
                    "results": results,
                },
                snapshots_dir / f"checkpoint_trial_{trial:04d}.pkl",
            )

    return best_full, best_first_n, results


def final_evaluation_esn2n_branch(
    branch,
    task,
    best_result,
    final_seeds,
    root,
    padding,
):
    test_accs = []
    context_accs = []
    short_accs = []
    guesses = []

    worker_log(
        "ESN2NPair",
        root,
        f"pad {padding} | FINAL {branch} x{len(final_seeds)}",
    )

    for i, seed in enumerate(final_seeds, start=1):
        tf.keras.backend.clear_session()
        set_seed(seed)

        model = ESN(**best_result["model_params"])

        h_train_2n = extract_final_states(model, task["x_train"])
        h_val_2n = extract_final_states(model, task["x_val"])
        h_test_2n = extract_final_states(model, task["x_test"])

        if branch == "ESN_2N":
            h_train = h_train_2n
            h_val = h_val_2n
            h_test = h_test_2n
        else:
            h_train = h_train_2n[:, :100]
            h_val = h_val_2n[:, :100]
            h_test = h_test_2n[:, :100]

        score = score_test_view(
            task,
            h_train,
            h_val,
            h_test,
        )

        test_accs.append(score["test_accuracy"])
        context_accs.append(score["context_test_accuracy"])
        short_accs.append(score["short_test_accuracy"])

        guesses.append({
            "guess": i,
            "seed": seed,
            **score,
        })

        worker_log(
            "ESN2NPair",
            root,
            (
                f"pad {padding} | {branch} final {i}/{len(final_seeds)} | "
                f"test {score['test_accuracy']:.4f}"
            ),
        )

        cleanup_trial(
            model,
            h_train_2n,
            h_val_2n,
            h_test_2n,
        )

    return {
        "model": branch,
        "mean_test_accuracy": float(np.mean(test_accs)),
        "std_test_accuracy": float(np.std(test_accs)),
        "mean_context_test_accuracy": float(np.mean(context_accs)),
        "std_context_test_accuracy": float(np.std(context_accs)),
        "mean_short_test_accuracy": float(np.mean(short_accs)),
        "std_short_test_accuracy": float(np.std(short_accs)),
        "guesses": guesses,
    }


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

SUMMARY_FIELDS = [
    "padding",
    "mean_validation_accuracy",
    "std_validation_accuracy",
    "mean_test_accuracy",
    "std_test_accuracy",
    "gap_validation_test",
]


def summary_row(best_result, final_result, padding):
    mean_test = final_result["mean_test_accuracy"]
    return {
        "padding": padding,
        "mean_validation_accuracy": best_result["validation_accuracy"],
        "std_validation_accuracy": best_result["validation_std"],
        "mean_test_accuracy": mean_test,
        "std_test_accuracy": final_result["std_test_accuracy"],
        "gap_validation_test": (
            best_result["validation_accuracy"]
            - mean_test
        ),
    }


# ---------------------------------------------------------------------
# One generic worker
# ---------------------------------------------------------------------

def run_generic_worker(
    model_type,
    dry_run,
    restart,
):
    root = Path(
        "results",
        "diagnostics",
        RESULT_DIR_NAMES[model_type],
    )
    ensure_dir(root)

    paddings = DRY_PADDINGS if dry_run else FULL_PADDINGS
    num_trials = DRY_TRIALS if dry_run else FULL_TRIALS
    final_seeds = DRY_FINAL_SEEDS if dry_run else FINAL_SEEDS

    worker_log(
        model_type,
        root,
        (
            f"START | pads={paddings} | trials={num_trials} | "
            f"selection seeds={MODEL_SELECTION_SEEDS} | final seeds={final_seeds}"
        ),
    )

    rows = []

    for padding in paddings:
        pad_dir = root / f"long_{padding}"
        final_name = "dry_run.pkl" if dry_run else "full_run.pkl"
        final_path = pad_dir / final_name

        if final_path.exists() and not restart:
            with open(final_path, "rb") as f:
                output = pickle.load(f)

            rows.append(
                summary_row(
                    output["best_result"],
                    output["final_result"],
                    padding,
                )
            )
            worker_log(
                model_type,
                root,
                f"SKIP pad {padding} | already complete",
            )
            continue

        worker_log(
            model_type,
            root,
            f"PAD {padding} | building task",
        )

        task = generate_contextual_short_long_task(
            long_padding_length=padding,
            seed=SEARCH_SEED,
        )

        configurations = generate_configurations(
            model_type,
            num_trials,
            SEARCH_SEED,
        )

        best_result, search_results = model_selection_generic(
            model_type,
            task,
            configurations,
            root,
            padding,
            restart=restart,
        )

        final_result = final_evaluation_generic(
            model_type,
            task,
            best_result,
            final_seeds,
            root,
            padding,
        )

        output = {
            "model": model_type,
            "padding": padding,
            "num_trials": num_trials,
            "model_selection_seeds": MODEL_SELECTION_SEEDS,
            "final_seeds": final_seeds,
            "best_result": best_result,
            "search_results": search_results,
            "final_result": final_result,
        }

        atomic_pickle_dump(
            output,
            final_path,
        )

        row = summary_row(
            best_result,
            final_result,
            padding,
        )
        rows.append(row)

        summary_file = (
            "summary_dry.csv"
            if dry_run
            else "summary.csv"
        )
        save_csv(
            sorted(rows, key=lambda x: int(x["padding"])),
            root / summary_file,
            SUMMARY_FIELDS,
        )

        worker_log(
            model_type,
            root,
            (
                f"DONE pad {padding} | "
                f"val {row['mean_validation_accuracy']:.4f} +/- "
                f"{row['std_validation_accuracy']:.4f} | "
                f"test {row['mean_test_accuracy']:.4f} +/- "
                f"{row['std_test_accuracy']:.4f} | "
                f"gap {row['gap_validation_test']:+.4f}"
            ),
        )

        write_status(
            root,
            model_type,
            phase="padding_complete",
            padding=padding,
            mean_validation=row["mean_validation_accuracy"],
            mean_test=row["mean_test_accuracy"],
            gap=row["gap_validation_test"],
        )

        del task, configurations, search_results, output
        gc.collect()

    worker_log(
        model_type,
        root,
        "ALL PADDINGS COMPLETE",
    )

    write_status(
        root,
        model_type,
        phase="complete",
        completed=True,
    )


# ---------------------------------------------------------------------
# ESN 2N paired worker
# ---------------------------------------------------------------------

def run_esn2n_pair_worker(
    dry_run,
    restart,
):
    model_type = "ESN2NPair"
    root = Path(
        "results",
        "diagnostics",
        RESULT_DIR_NAMES[model_type],
    )
    ensure_dir(root)

    paddings = DRY_PADDINGS if dry_run else FULL_PADDINGS
    num_trials = DRY_TRIALS if dry_run else FULL_TRIALS
    final_seeds = DRY_FINAL_SEEDS if dry_run else FINAL_SEEDS

    worker_log(
        model_type,
        root,
        (
            f"START | shared 200-unit ESN states | pads={paddings} | "
            f"trials={num_trials}"
        ),
    )

    rows_full = []
    rows_first_n = []

    for padding in paddings:
        pad_dir = root / f"long_{padding}"
        final_name = "dry_run.pkl" if dry_run else "full_run.pkl"
        final_path = pad_dir / final_name

        if final_path.exists() and not restart:
            with open(final_path, "rb") as f:
                output = pickle.load(f)

            rows_full.append(
                summary_row(
                    output["best_ESN_2N"],
                    output["final_ESN_2N"],
                    padding,
                )
            )
            rows_first_n.append(
                summary_row(
                    output["best_ESN_2N_to_N"],
                    output["final_ESN_2N_to_N"],
                    padding,
                )
            )

            worker_log(
                model_type,
                root,
                f"SKIP pad {padding} | already complete",
            )
            continue

        task = generate_contextual_short_long_task(
            long_padding_length=padding,
            seed=SEARCH_SEED,
        )

        configurations = generate_configurations(
            model_type,
            num_trials,
            SEARCH_SEED,
        )

        (
            best_full,
            best_first_n,
            search_results,
        ) = model_selection_esn2n_pair(
            task,
            configurations,
            root,
            padding,
            restart=restart,
        )

        final_full = final_evaluation_esn2n_branch(
            "ESN_2N",
            task,
            best_full,
            final_seeds,
            root,
            padding,
        )

        final_first_n = final_evaluation_esn2n_branch(
            "ESN_2N_to_N",
            task,
            best_first_n,
            final_seeds,
            root,
            padding,
        )

        output = {
            "model": model_type,
            "padding": padding,
            "num_trials": num_trials,
            "model_selection_seeds": MODEL_SELECTION_SEEDS,
            "final_seeds": final_seeds,
            "best_ESN_2N": best_full,
            "best_ESN_2N_to_N": best_first_n,
            "search_results": search_results,
            "final_ESN_2N": final_full,
            "final_ESN_2N_to_N": final_first_n,
        }

        atomic_pickle_dump(
            output,
            final_path,
        )

        row_full = summary_row(
            best_full,
            final_full,
            padding,
        )
        row_first_n = summary_row(
            best_first_n,
            final_first_n,
            padding,
        )

        rows_full.append(row_full)
        rows_first_n.append(row_first_n)

        suffix = "_dry" if dry_run else ""

        save_csv(
            sorted(rows_full, key=lambda x: int(x["padding"])),
            root / f"summary_ESN_2N{suffix}.csv",
            SUMMARY_FIELDS,
        )

        save_csv(
            sorted(rows_first_n, key=lambda x: int(x["padding"])),
            root / f"summary_ESN_2N_to_N{suffix}.csv",
            SUMMARY_FIELDS,
        )

        worker_log(
            model_type,
            root,
            (
                f"DONE pad {padding} | "
                f"2N test {row_full['mean_test_accuracy']:.4f} | "
                f"2N->N test {row_first_n['mean_test_accuracy']:.4f}"
            ),
        )

        write_status(
            root,
            model_type,
            phase="padding_complete",
            padding=padding,
            ESN_2N_test=row_full["mean_test_accuracy"],
            ESN_2N_to_N_test=row_first_n["mean_test_accuracy"],
        )

        del task, configurations, search_results, output
        gc.collect()

    worker_log(
        model_type,
        root,
        "ALL PADDINGS COMPLETE",
    )

    write_status(
        root,
        model_type,
        phase="complete",
        completed=True,
    )


# ---------------------------------------------------------------------
# Worker entry point
# ---------------------------------------------------------------------

def worker_main(args):
    # This worker process sees only the requested physical GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["PYTHONUNBUFFERED"] = "1"

    # Avoid CPU oversubscription when several workers run together.
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

    load_ml_stack()

    global ALPHA_VALUES
    ALPHA_VALUES = np.logspace(-5, 5, 11)

    if args.worker == "ESN2NPair":
        run_esn2n_pair_worker(
            dry_run=args.dry_run,
            restart=args.restart,
        )
    else:
        run_generic_worker(
            args.worker,
            dry_run=args.dry_run,
            restart=args.restart,
        )


# ---------------------------------------------------------------------
# Master: launch several independent model workers on GPU 3
# ---------------------------------------------------------------------

def master_main(args):
    requested = (
        [x.strip() for x in args.models.split(",") if x.strip()]
        if args.models
        else MODEL_QUEUE.copy()
    )

    unknown = [
        m for m in requested
        if m not in MODEL_QUEUE
    ]
    if unknown:
        raise ValueError(
            f"Unknown models: {unknown}. Allowed: {MODEL_QUEUE}"
        )

    print("=" * 72)
    print("SHORT+LONG MULTI-SEED MASTER")
    print("=" * 72)
    print(f"GPU: {args.gpu}")
    print(f"Max parallel workers: {args.max_parallel}")
    print(f"Models: {requested}")
    print(f"Dry run: {args.dry_run}")
    print("Coupled: skipped (already completed)")
    print("=" * 72)
    print(flush=True)

    script = os.path.abspath(__file__)
    pending = requested.copy()
    running = {}

    def start_model(model_name):
        cmd = [
            sys.executable,
            "-u",
            script,
            "--worker",
            model_name,
            "--gpu",
            str(args.gpu),
        ]

        if args.dry_run:
            cmd.append("--dry-run")

        if args.restart:
            cmd.append("--restart")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env["TF_CPP_MIN_LOG_LEVEL"] = "2"
        env["PYTHONUNBUFFERED"] = "1"

        # Same GPU shared by a small number of workers.
        env.setdefault("OMP_NUM_THREADS", "2")
        env.setdefault("OPENBLAS_NUM_THREADS", "2")
        env.setdefault("MKL_NUM_THREADS", "2")
        env.setdefault("NUMEXPR_NUM_THREADS", "2")

        process = subprocess.Popen(
            cmd,
            env=env,
        )

        running[model_name] = process

        print(
            f"[MASTER] START {model_name} | pid={process.pid}",
            flush=True,
        )

    while pending or running:
        while (
            pending
            and len(running) < args.max_parallel
        ):
            start_model(pending.pop(0))

        time.sleep(2)

        finished = []

        for model_name, process in running.items():
            code = process.poll()

            if code is not None:
                finished.append(model_name)

                if code == 0:
                    print(
                        f"[MASTER] DONE  {model_name}",
                        flush=True,
                    )
                else:
                    print(
                        f"[MASTER] FAIL  {model_name} | exit={code}",
                        flush=True,
                    )

        for model_name in finished:
            del running[model_name]

    print()
    print("=" * 72)
    print("[MASTER] ALL WORKERS FINISHED")
    print("=" * 72)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu",
        default="3",
        help="Physical GPU id. Default: 3",
    )

    parser.add_argument(
        "--max-parallel",
        type=int,
        default=2,
        help="Concurrent workers on the selected GPU. Default: 2",
    )

    parser.add_argument(
        "--models",
        default="",
        help=(
            "Comma-separated subset, e.g. "
            "Hybrid,CoupledSymmetric,Mixed"
        ),
    )

    parser.add_argument(
        "--worker",
        default="",
        choices=[""] + MODEL_QUEUE,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use only padding 700, 10 trials, one final seed.",
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore completed/checkpointed work and restart from scratch.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be >= 1")

    if args.worker:
        worker_main(args)
    else:
        master_main(args)

