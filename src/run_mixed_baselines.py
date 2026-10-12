"""
Additional ESN baselines for the MixedDynamics reservoir.

This script does NOT re-run ESN-N or Mixed. It evaluates only:

    1) ESN-2N       : ESN with 2N recurrent units, all 2N states to readout.
    2) ESN-2N->N    : the SAME ESN with 2N recurrent units, but only the
                      first N state components are given to the readout.

For each model-selection configuration, the two baselines share exactly the
same ESN realization and reservoir states. Their readouts are selected
independently and their best hyperparameter configurations are tracked
independently.

Protocols intentionally follow the existing project runners:

Datasets
--------
- FordA, FordB, Adiac, Libras, RacketSports
- Random search, 400 configurations in the full run
- N is a hyperparameter in {10, 20, ..., 200}; actual ESN size is 2N
- one reservoir realization per configuration, seed = 42 + trial
- RidgeClassifierCV readout
- final evaluation on train_all/test with seeds 1001..1005

Synthetic
---------
- padding lengths 100..1000 in steps of 100
- N = 100; actual ESN size = 200
- Random search, 400 trials per padding in the full run
- same ESN hyperparameter ranges as the original Synthetic ESN baseline
- fixed reservoir seed 42 during model selection
- multi-output Ridge readout, alpha selected on validation
- final seeds 1001..1005

Short+Long
----------
- long padding lengths 100..1000 in steps of 100
- N = 100; actual ESN size = 200
- Random search, 400 trials per padding in the full run
- same ESN hyperparameter ranges as the original Short+Long ESN baseline
- fixed reservoir seed 42 during model selection
- main 4-class Ridge readout + Context and Short diagnostic probes
- final seeds 1001..1005

Memory Capacity
---------------
- N = 100; actual ESN size = 200
- the original ESN MemCap GRID is retained (not replaced by random search)
- leaky x input_scaling grid, spectral_radius=0.9, bias=0
- one reservoir realization per configuration, seed = 42 + index
- unregularized multi-output Ridge (alpha=0, SVD)
- final seeds 1001..1005

Results are stored under results/MixedBaselines/ and never overwrite the
existing ESN/Mixed experiment directories.
"""

import csv
import itertools
import os
import pickle
import time
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from scipy.linalg import LinAlgWarning
from sklearn.linear_model import Ridge, RidgeClassifierCV
from sklearn.metrics import accuracy_score

from src.memory_tasks import (
    generate_contextual_short_long_task,
    generate_memcap_task,
    generate_synthetic_ltm_task,
)
from src.reservoir import ESN


# ============================================================
# Global names / execution switches
# ============================================================

BASELINE_FULL = "ESN_2N"
BASELINE_SUBSET = "ESN_2N_to_N"
BASELINES = (BASELINE_FULL, BASELINE_SUBSET)

# Start with a dry run. Change only this to False for the complete battery.
DRY_RUN = True

RUN_DATASETS = True
RUN_SYNTHETIC = True
RUN_SHORT_LONG = True
RUN_MEMCAP = True

SEARCH_SEED = 42
RESULTS_ROOT = os.path.join("results", "MixedBaselines")

# Same upper bound used by the dataset runner.
DATASET_MAX_TIME_HOURS = 10
DATASET_MAX_TIME_SECONDS = DATASET_MAX_TIME_HOURS * 60 * 60


# ============================================================
# GPU configuration
# ============================================================

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# ============================================================
# Generic utilities
# ============================================================


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_pickle(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def save_csv(rows, path):
    if not rows:
        return

    ensure_dir(os.path.dirname(path))

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_result(result):
    """Flatten a result dictionary for compact CSV output."""
    row = {}
    for key, value in result.items():
        if key == "model_params":
            for param_key, param_value in value.items():
                row[f"param_{param_key}"] = param_value
        elif isinstance(value, (list, tuple, np.ndarray, dict)):
            # Keep large/nested objects in pickle files, not CSV.
            continue
        else:
            row[key] = value
    return row


def representation(states, baseline, base_units):
    """Return either all 2N states or the fixed first-N subset."""
    if baseline == BASELINE_FULL:
        return states
    if baseline == BASELINE_SUBSET:
        return states[..., :base_units]
    raise ValueError(f"Unknown baseline: {baseline}")


def pair_result_template(
    baseline,
    base_units,
    model_params,
    seed,
    validation_score,
    readout_alpha=None,
    **extra,
):
    result = {
        "model": baseline,
        "base_units": int(base_units),
        "reservoir_units": int(model_params["units"]),
        "validation_accuracy": float(validation_score),
        "seed": int(seed),
        "model_params": dict(model_params),
    }
    if readout_alpha is not None:
        result["readout_alpha"] = float(readout_alpha)
    result.update(extra)
    return result


def print_pair_progress(trial, total, best_full, best_subset, metric="acc"):
    if metric == "mc":
        a = best_full["validation_mc"]
        b = best_subset["validation_mc"]
        print(
            f"Progress | {trial}/{total} | "
            f"best {BASELINE_FULL}: {a:.4f} | "
            f"best {BASELINE_SUBSET}: {b:.4f}"
        )
    else:
        a = best_full["validation_accuracy"]
        b = best_subset["validation_accuracy"]
        print(
            f"Progress | {trial}/{total} | "
            f"best {BASELINE_FULL}: {a:.4f} | "
            f"best {BASELINE_SUBSET}: {b:.4f}"
        )


# ============================================================
# Dataset protocol
# ============================================================


def load_dataset(dataset_name, datasets_path="./data"):
    """Same dataset layout used by the existing dataset runner."""
    dataset_file = os.path.join(
        datasets_path,
        dataset_name + "_dataset.p",
    )

    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset not found: {dataset_file}")

    with open(dataset_file, "rb") as file:
        dataset = pickle.load(file)

    x_train_all = dataset[0]
    y_train_all = dataset[1]
    x_test = dataset[2]
    y_test = dataset[3]
    x_train = dataset[4]
    x_val = dataset[5]
    y_train = dataset[6]
    y_val = dataset[7]

    return (
        x_train_all,
        y_train_all,
        x_test,
        y_test,
        x_train,
        x_val,
        y_train,
        y_val,
    )


def sample_dataset_esn_2n_configuration(rng):
    """
    Same ESN search space as the dataset runner, except that sampled N is
    stored as base_units and the actual ESN is built with 2N units.
    """
    base_units = int(10 * rng.integers(1, 21))

    model_params = {
        "units": 2 * base_units,
        "leaky": float(10.0 ** rng.integers(-5, 1)),
        "input_scaling": float(10.0 ** rng.integers(-3, 2)),
        "spectral_radius": float(0.1 * rng.integers(1, 16)),
        "bias_scaling": float(10.0 ** rng.integers(-3, 2)),
    }

    return base_units, model_params


def dataset_train_readout(train_states, y_train, alphas=None):
    if alphas is None:
        alphas = np.logspace(-5, 5, 11)

    readout = RidgeClassifierCV(
        alphas=alphas,
        cv=None,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=LinAlgWarning)
        warnings.filterwarnings(
            "ignore",
            message="Singular matrix in solving dual problem.*",
            category=UserWarning,
        )
        readout.fit(train_states, y_train)

    return readout


def dataset_extract_states(model, x):
    return np.asarray(model(x))


def run_dataset_pair_trial(
    x_train,
    y_train,
    x_val,
    y_val,
    base_units,
    model_params,
    seed,
):
    """One ESN realization, two independently trained readouts."""
    tf.keras.backend.clear_session()
    set_seed(seed)

    model = ESN(**model_params)

    start_time = time.time()
    train_states_2n = dataset_extract_states(model, x_train)
    val_states_2n = dataset_extract_states(model, x_val)
    reservoir_elapsed = time.time() - start_time

    results = {}

    for baseline in BASELINES:
        h_train = representation(train_states_2n, baseline, base_units)
        h_val = representation(val_states_2n, baseline, base_units)

        readout = dataset_train_readout(h_train, y_train)
        validation_accuracy = float(readout.score(h_val, y_val))

        results[baseline] = pair_result_template(
            baseline=baseline,
            base_units=base_units,
            model_params=model_params,
            seed=seed,
            validation_score=validation_accuracy,
            readout_alpha=readout.alpha_,
            reservoir_elapsed_time=float(reservoir_elapsed),
        )

    return results


def random_search_dataset_pair(
    dataset_name,
    x_train,
    y_train,
    x_val,
    y_val,
    num_trials,
    search_seed=42,
    max_time_seconds=None,
    save_checkpoints=False,
):
    rng = np.random.default_rng(search_seed)
    configurations = [
        sample_dataset_esn_2n_configuration(rng)
        for _ in range(num_trials)
    ]

    all_results = {baseline: [] for baseline in BASELINES}
    best = {baseline: None for baseline in BASELINES}

    start_time = time.time()

    print(f"Configurations: {num_trials}")

    for trial, (base_units, model_params) in enumerate(configurations, start=1):
        if max_time_seconds is not None:
            elapsed = time.time() - start_time
            if elapsed >= max_time_seconds:
                print(
                    "\nMaximum dataset model-selection time reached after "
                    f"{trial - 1} completed trials."
                )
                break

        # Same rule as the dataset ESN runner.
        trial_seed = search_seed + trial

        pair = run_dataset_pair_trial(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            base_units=base_units,
            model_params=model_params,
            seed=trial_seed,
        )

        new_best_messages = []

        for baseline in BASELINES:
            result = pair[baseline]
            result["trial"] = trial
            all_results[baseline].append(result)

            if (
                best[baseline] is None
                or result["validation_accuracy"]
                > best[baseline]["validation_accuracy"]
            ):
                best[baseline] = result
                new_best_messages.append(
                    f"{baseline}={result['validation_accuracy']:.4f} "
                    f"(N={base_units}, 2N={model_params['units']})"
                )

        if new_best_messages:
            print(
                f"New best | Trial {trial}/{num_trials} | "
                + " | ".join(new_best_messages)
            )
        elif trial % 25 == 0 or trial == num_trials:
            print_pair_progress(
                trial,
                num_trials,
                best[BASELINE_FULL],
                best[BASELINE_SUBSET],
            )

        if save_checkpoints and (trial % 25 == 0 or trial == num_trials):
            checkpoint_dir = ensure_dir(
                os.path.join(
                    RESULTS_ROOT,
                    "Datasets",
                    dataset_name,
                    "_checkpoint",
                )
            )
            save_pickle(
                {
                    "completed_trials": trial,
                    "total_trials": num_trials,
                    "best": best,
                    "results": all_results,
                },
                os.path.join(checkpoint_dir, "model_selection_checkpoint.pkl"),
            )

    return best, all_results


def final_dataset_evaluation(
    baseline,
    best_result,
    x_train_all,
    y_train_all,
    x_test,
    y_test,
    num_guesses=5,
    start_seed=1000,
):
    base_units = int(best_result["base_units"])
    model_params = dict(best_result["model_params"])

    test_accuracies = []
    elapsed_times = []
    guesses = []

    print(f"\n--- FINAL DATASET EVALUATION: {baseline} ---")
    print("Best hyperparameters:")
    print(best_result)

    for guess in range(1, num_guesses + 1):
        tf.keras.backend.clear_session()
        seed = start_seed + guess
        set_seed(seed)

        model = ESN(**model_params)

        start_time = time.time()
        train_states_2n = dataset_extract_states(model, x_train_all)
        test_states_2n = dataset_extract_states(model, x_test)

        h_train = representation(train_states_2n, baseline, base_units)
        h_test = representation(test_states_2n, baseline, base_units)

        readout = dataset_train_readout(h_train, y_train_all)
        test_accuracy = float(readout.score(h_test, y_test))
        elapsed_time = time.time() - start_time

        test_accuracies.append(test_accuracy)
        elapsed_times.append(elapsed_time)
        guesses.append(
            {
                "guess": guess,
                "seed": seed,
                "test_accuracy": test_accuracy,
                "readout_alpha": float(readout.alpha_),
                "elapsed_time": float(elapsed_time),
            }
        )

        print(
            f"Guess {guess}/{num_guesses} | "
            f"Test acc: {test_accuracy:.4f} | "
            f"Ridge alpha: {readout.alpha_} | "
            f"Seed: {seed}"
        )

    return {
        "model": baseline,
        "base_units": base_units,
        "reservoir_units": int(model_params["units"]),
        "best_params": model_params,
        "num_guesses": num_guesses,
        "test_accuracies": test_accuracies,
        "mean_test_accuracy": float(np.mean(test_accuracies)),
        "std_test_accuracy": float(np.std(test_accuracies)),
        "mean_elapsed_time": float(np.mean(elapsed_times)),
        "std_elapsed_time": float(np.std(elapsed_times)),
        "guesses": guesses,
    }


def save_dataset_baseline_results(
    dataset_name,
    baseline,
    best_result,
    search_results,
    final_result,
):
    path = ensure_dir(
        os.path.join(
            RESULTS_ROOT,
            "Datasets",
            dataset_name,
            baseline,
        )
    )

    save_pickle(best_result, os.path.join(path, "best_model_selection.pkl"))
    save_pickle(search_results, os.path.join(path, "model_selection_trials.pkl"))
    save_pickle(final_result, os.path.join(path, "final_evaluation.pkl"))
    save_csv(
        [flatten_result(row) for row in search_results],
        os.path.join(path, "model_selection_trials.csv"),
    )
    save_csv(final_result["guesses"], os.path.join(path, "final_guesses.csv"))


# ============================================================
# Synthetic protocol
# ============================================================


def sample_synthetic_esn_2n_configuration(rng):
    base_units = 100
    model_params = {
        "units": 2 * base_units,
        "leaky": float(rng.uniform(0.01, 1.0)),
        "input_scaling": float(rng.uniform(0.01, 1.5)),
        "spectral_radius": float(rng.uniform(0.01, 1.5)),
        "bias_scaling": float(rng.uniform(0.01, 1.5)),
        "return_sequences": False,
    }
    return base_units, model_params


def classification_train_ridge(
    x_train,
    y_train,
    num_classes,
    alpha,
    solver,
    target_dtype=np.float64,
):
    y_train_one_hot = np.eye(num_classes, dtype=target_dtype)[y_train]
    readout = Ridge(alpha=alpha, solver=solver)
    readout.fit(x_train, y_train_one_hot)
    return readout


def classification_evaluate_ridge(readout, states, labels):
    scores = readout.predict(states)
    predictions = np.argmax(scores, axis=1)
    return float(accuracy_score(labels, predictions))


def select_classification_ridge_alpha(
    h_train,
    y_train,
    h_val,
    y_val,
    num_classes,
    solver,
    target_dtype=np.float64,
):
    alpha_values = np.logspace(-5, 5, 11)
    best_alpha = None
    best_accuracy = -np.inf

    for alpha in alpha_values:
        readout = classification_train_ridge(
            h_train,
            y_train,
            num_classes,
            alpha,
            solver,
            target_dtype=target_dtype,
        )
        accuracy = classification_evaluate_ridge(readout, h_val, y_val)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_alpha = alpha

    return float(best_alpha), float(best_accuracy)


def run_synthetic_pair_trial(task, base_units, model_params, seed):
    tf.keras.backend.clear_session()
    set_seed(seed)

    model = ESN(**model_params)
    h_train_2n = np.asarray(model(task["x_train"]))
    h_val_2n = np.asarray(model(task["x_val"]))

    results = {}

    for baseline in BASELINES:
        h_train = representation(h_train_2n, baseline, base_units)
        h_val = representation(h_val_2n, baseline, base_units)

        best_alpha, validation_accuracy = select_classification_ridge_alpha(
            h_train=h_train,
            y_train=task["y_train"],
            h_val=h_val,
            y_val=task["y_val"],
            num_classes=task["num_classes"],
            solver="svd",
            target_dtype=np.float32,
        )

        results[baseline] = pair_result_template(
            baseline=baseline,
            base_units=base_units,
            model_params=model_params,
            seed=seed,
            validation_score=validation_accuracy,
            readout_alpha=best_alpha,
        )

    return results


def random_search_synthetic_pair(
    task,
    num_trials,
    search_seed=42,
    save_checkpoints=False,
):
    rng = np.random.default_rng(search_seed)

    all_results = {baseline: [] for baseline in BASELINES}
    best = {baseline: None for baseline in BASELINES}

    print(f"Trials: {num_trials}")

    for trial in range(1, num_trials + 1):
        base_units, model_params = sample_synthetic_esn_2n_configuration(rng)

        # Same rule as the existing Synthetic ESN baseline:
        # fixed reservoir seed during model selection.
        pair = run_synthetic_pair_trial(
            task=task,
            base_units=base_units,
            model_params=model_params,
            seed=search_seed,
        )

        new_best_messages = []
        for baseline in BASELINES:
            result = pair[baseline]
            result["trial"] = trial
            all_results[baseline].append(result)

            if (
                best[baseline] is None
                or result["validation_accuracy"]
                > best[baseline]["validation_accuracy"]
            ):
                best[baseline] = result
                new_best_messages.append(
                    f"{baseline}={result['validation_accuracy']:.4f} "
                    f"(alpha={result['readout_alpha']:.1e})"
                )

        if new_best_messages:
            print(
                f"New best | {trial}/{num_trials} | "
                + " | ".join(new_best_messages)
            )
        elif trial % 25 == 0 or trial == num_trials:
            print_pair_progress(
                trial,
                num_trials,
                best[BASELINE_FULL],
                best[BASELINE_SUBSET],
            )

        if save_checkpoints and (trial % 25 == 0 or trial == num_trials):
            path = ensure_dir(
                os.path.join(
                    RESULTS_ROOT,
                    "Synthetic",
                    f"padding_{task['padding_length']}",
                    "_checkpoint",
                )
            )
            save_pickle(
                {
                    "completed_trials": trial,
                    "total_trials": num_trials,
                    "best": best,
                    "results": all_results,
                },
                os.path.join(path, "model_selection_checkpoint.pkl"),
            )

    return best, all_results


def final_synthetic_baseline_evaluation(baseline, task, best_result, final_seeds):
    base_units = int(best_result["base_units"])
    model_params = dict(best_result["model_params"])

    test_accuracies = []
    guesses = []

    print(f"\n--- FINAL SYNTHETIC EVALUATION: {baseline} ---")
    print(f"Padding: {task['padding_length']}")
    print("Best hyperparameters:")
    print(best_result)

    for guess, seed in enumerate(final_seeds, start=1):
        tf.keras.backend.clear_session()
        set_seed(seed)

        model = ESN(**model_params)
        h_train_2n = np.asarray(model(task["x_train"]))
        h_val_2n = np.asarray(model(task["x_val"]))
        h_test_2n = np.asarray(model(task["x_test"]))

        h_train = representation(h_train_2n, baseline, base_units)
        h_val = representation(h_val_2n, baseline, base_units)
        h_test = representation(h_test_2n, baseline, base_units)

        best_alpha, validation_accuracy = select_classification_ridge_alpha(
            h_train=h_train,
            y_train=task["y_train"],
            h_val=h_val,
            y_val=task["y_val"],
            num_classes=task["num_classes"],
            solver="svd",
            target_dtype=np.float32,
        )

        readout = classification_train_ridge(
            x_train=h_train,
            y_train=task["y_train"],
            num_classes=task["num_classes"],
            alpha=best_alpha,
            solver="svd",
            target_dtype=np.float32,
        )

        test_accuracy = classification_evaluate_ridge(
            readout,
            h_test,
            task["y_test"],
        )

        test_accuracies.append(test_accuracy)
        guesses.append(
            {
                "guess": guess,
                "seed": seed,
                "ridge_alpha": best_alpha,
                "validation_accuracy": validation_accuracy,
                "test_accuracy": test_accuracy,
            }
        )

        print(
            f"Guess {guess}/{len(final_seeds)} | "
            f"Test acc: {test_accuracy:.4f} | "
            f"Val acc: {validation_accuracy:.4f} | "
            f"alpha: {best_alpha:.1e} | Seed: {seed}"
        )

    return {
        "model": baseline,
        "padding_length": task["padding_length"],
        "base_units": base_units,
        "reservoir_units": int(model_params["units"]),
        "best_model_params": model_params,
        "best_validation_accuracy": float(best_result["validation_accuracy"]),
        "mean_test_accuracy": float(np.mean(test_accuracies)),
        "std_test_accuracy": float(np.std(test_accuracies)),
        "guesses": guesses,
    }


def save_synthetic_baseline_results(
    padding_length,
    baseline,
    best_result,
    search_results,
    final_result,
):
    path = ensure_dir(
        os.path.join(
            RESULTS_ROOT,
            "Synthetic",
            f"padding_{padding_length}",
            baseline,
        )
    )
    save_pickle(best_result, os.path.join(path, "best_configuration.pkl"))
    save_pickle(search_results, os.path.join(path, "random_search.pkl"))
    save_pickle(final_result, os.path.join(path, "final_evaluation.pkl"))
    save_csv(
        [flatten_result(row) for row in search_results],
        os.path.join(path, "random_search.csv"),
    )
    save_csv(final_result["guesses"], os.path.join(path, "final_guesses.csv"))


# ============================================================
# Short+Long protocol
# ============================================================


def sample_short_long_esn_2n_configuration(rng):
    base_units = 100
    model_params = {
        "units": 2 * base_units,
        "leaky": float(rng.uniform(0.01, 1.0)),
        "input_scaling": float(rng.uniform(0.01, 1.5)),
        "spectral_radius": float(rng.uniform(0.01, 1.5)),
        "bias_scaling": float(rng.uniform(0.01, 1.5)),
        "return_sequences": False,
    }
    return base_units, model_params


def short_long_probe_validation(h_train, y_train, h_val, y_val):
    _, accuracy = select_classification_ridge_alpha(
        h_train=h_train,
        y_train=y_train,
        h_val=h_val,
        y_val=y_val,
        num_classes=2,
        solver="cholesky",
    )
    return float(accuracy)


def short_long_probe_test(h_train, y_train, h_val, y_val, h_test, y_test):
    best_alpha, _ = select_classification_ridge_alpha(
        h_train=h_train,
        y_train=y_train,
        h_val=h_val,
        y_val=y_val,
        num_classes=2,
        solver="cholesky",
    )

    readout = classification_train_ridge(
        x_train=h_train,
        y_train=y_train,
        num_classes=2,
        alpha=best_alpha,
        solver="cholesky",
    )

    return classification_evaluate_ridge(readout, h_test, y_test)


def run_short_long_pair_trial(task, base_units, model_params, seed):
    tf.keras.backend.clear_session()
    set_seed(seed)

    model = ESN(**model_params)
    h_train_2n = np.asarray(model(task["x_train"]), dtype=np.float64)
    h_val_2n = np.asarray(model(task["x_val"]), dtype=np.float64)

    results = {}

    for baseline in BASELINES:
        h_train = representation(h_train_2n, baseline, base_units)
        h_val = representation(h_val_2n, baseline, base_units)

        best_alpha, validation_accuracy = select_classification_ridge_alpha(
            h_train=h_train,
            y_train=task["y_train"],
            h_val=h_val,
            y_val=task["y_val"],
            num_classes=task["num_classes"],
            solver="cholesky",
        )

        context_accuracy = short_long_probe_validation(
            h_train,
            task["context_train"],
            h_val,
            task["context_val"],
        )

        short_accuracy = short_long_probe_validation(
            h_train,
            task["short_train"],
            h_val,
            task["short_val"],
        )

        results[baseline] = pair_result_template(
            baseline=baseline,
            base_units=base_units,
            model_params=model_params,
            seed=seed,
            validation_score=validation_accuracy,
            readout_alpha=best_alpha,
            context_accuracy=float(context_accuracy),
            short_accuracy=float(short_accuracy),
        )

    return results


def random_search_short_long_pair(
    task,
    num_trials,
    search_seed=42,
    save_checkpoints=False,
):
    rng = np.random.default_rng(search_seed)

    all_results = {baseline: [] for baseline in BASELINES}
    best = {baseline: None for baseline in BASELINES}

    print(f"Trials: {num_trials}")

    for trial in range(1, num_trials + 1):
        base_units, model_params = sample_short_long_esn_2n_configuration(rng)

        # Same rule as the current Short+Long ESN runner:
        # fixed model-selection seed.
        pair = run_short_long_pair_trial(
            task=task,
            base_units=base_units,
            model_params=model_params,
            seed=search_seed,
        )

        new_best_messages = []

        for baseline in BASELINES:
            result = pair[baseline]
            result["trial"] = trial
            all_results[baseline].append(result)

            # Exactly as in the existing runner: only MAIN validation
            # accuracy determines the best configuration. The two probes
            # are diagnostics and do not enter model selection.
            if (
                best[baseline] is None
                or result["validation_accuracy"]
                > best[baseline]["validation_accuracy"]
            ):
                best[baseline] = result
                new_best_messages.append(
                    f"{baseline}: main={result['validation_accuracy']:.4f}, "
                    f"context={result['context_accuracy']:.4f}, "
                    f"short={result['short_accuracy']:.4f}"
                )

        if new_best_messages:
            print(
                f"New best | {trial}/{num_trials} | "
                + " | ".join(new_best_messages)
            )
        elif trial % 25 == 0 or trial == num_trials:
            print_pair_progress(
                trial,
                num_trials,
                best[BASELINE_FULL],
                best[BASELINE_SUBSET],
            )

        if save_checkpoints and (trial % 25 == 0 or trial == num_trials):
            path = ensure_dir(
                os.path.join(
                    RESULTS_ROOT,
                    "ContextualShortLong",
                    f"long_{task['long_padding_length']}",
                    "_checkpoint",
                )
            )
            save_pickle(
                {
                    "completed_trials": trial,
                    "total_trials": num_trials,
                    "best": best,
                    "results": all_results,
                },
                os.path.join(path, "model_selection_checkpoint.pkl"),
            )

    return best, all_results


def final_short_long_baseline_evaluation(
    baseline,
    task,
    best_result,
    final_seeds,
):
    base_units = int(best_result["base_units"])
    model_params = dict(best_result["model_params"])

    test_accuracies = []
    context_test_accuracies = []
    short_test_accuracies = []
    guesses = []

    print(f"\n--- FINAL SHORT+LONG EVALUATION: {baseline} ---")
    print(f"Long padding: {task['long_padding_length']}")
    print("Best hyperparameters:")
    print(best_result)

    for guess, seed in enumerate(final_seeds, start=1):
        tf.keras.backend.clear_session()
        set_seed(seed)

        model = ESN(**model_params)
        h_train_2n = np.asarray(model(task["x_train"]), dtype=np.float64)
        h_val_2n = np.asarray(model(task["x_val"]), dtype=np.float64)
        h_test_2n = np.asarray(model(task["x_test"]), dtype=np.float64)

        h_train = representation(h_train_2n, baseline, base_units)
        h_val = representation(h_val_2n, baseline, base_units)
        h_test = representation(h_test_2n, baseline, base_units)

        best_alpha, validation_accuracy = select_classification_ridge_alpha(
            h_train=h_train,
            y_train=task["y_train"],
            h_val=h_val,
            y_val=task["y_val"],
            num_classes=task["num_classes"],
            solver="cholesky",
        )

        readout = classification_train_ridge(
            x_train=h_train,
            y_train=task["y_train"],
            num_classes=task["num_classes"],
            alpha=best_alpha,
            solver="cholesky",
        )

        test_accuracy = classification_evaluate_ridge(
            readout,
            h_test,
            task["y_test"],
        )

        context_test_accuracy = short_long_probe_test(
            h_train,
            task["context_train"],
            h_val,
            task["context_val"],
            h_test,
            task["context_test"],
        )

        short_test_accuracy = short_long_probe_test(
            h_train,
            task["short_train"],
            h_val,
            task["short_val"],
            h_test,
            task["short_test"],
        )

        test_accuracies.append(test_accuracy)
        context_test_accuracies.append(context_test_accuracy)
        short_test_accuracies.append(short_test_accuracy)

        guesses.append(
            {
                "guess": guess,
                "seed": seed,
                "ridge_alpha": best_alpha,
                "validation_accuracy": validation_accuracy,
                "test_accuracy": test_accuracy,
                "context_test_accuracy": context_test_accuracy,
                "short_test_accuracy": short_test_accuracy,
            }
        )

        print(
            f"Guess {guess}/{len(final_seeds)} | "
            f"Main: {test_accuracy:.4f} | "
            f"Context: {context_test_accuracy:.4f} | "
            f"Short: {short_test_accuracy:.4f} | "
            f"Val: {validation_accuracy:.4f} | "
            f"alpha: {best_alpha:.1e} | Seed: {seed}"
        )

    return {
        "model": baseline,
        "long_padding_length": task["long_padding_length"],
        "recent_window_length": task["recent_window_length"],
        "recent_delays": task["recent_delays"],
        "base_units": base_units,
        "reservoir_units": int(model_params["units"]),
        "best_model_params": model_params,
        "best_validation_accuracy": float(best_result["validation_accuracy"]),
        "mean_test_accuracy": float(np.mean(test_accuracies)),
        "std_test_accuracy": float(np.std(test_accuracies)),
        "mean_context_test_accuracy": float(np.mean(context_test_accuracies)),
        "std_context_test_accuracy": float(np.std(context_test_accuracies)),
        "mean_short_test_accuracy": float(np.mean(short_test_accuracies)),
        "std_short_test_accuracy": float(np.std(short_test_accuracies)),
        "guesses": guesses,
    }


def save_short_long_baseline_results(
    long_padding_length,
    baseline,
    best_result,
    search_results,
    final_result,
):
    path = ensure_dir(
        os.path.join(
            RESULTS_ROOT,
            "ContextualShortLong",
            f"long_{long_padding_length}",
            baseline,
        )
    )
    save_pickle(best_result, os.path.join(path, "best_configuration.pkl"))
    save_pickle(search_results, os.path.join(path, "random_search.pkl"))
    save_pickle(final_result, os.path.join(path, "final_evaluation.pkl"))
    save_csv(
        [flatten_result(row) for row in search_results],
        os.path.join(path, "random_search.csv"),
    )
    save_csv(final_result["guesses"], os.path.join(path, "final_guesses.csv"))


# ============================================================
# Memory Capacity protocol
# ============================================================


def squared_correlation(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0

    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    if not np.isfinite(correlation):
        return 0.0

    return float(correlation ** 2)


def evaluate_memcap(states, targets, train_range, evaluation_range):
    train_start, train_end = train_range
    eval_start, eval_end = evaluation_range

    x_train = states[train_start:train_end]
    y_train = targets[train_start:train_end]
    x_eval = states[eval_start:eval_end]
    y_eval = targets[eval_start:eval_end]

    readout = Ridge(alpha=0.0, solver="svd")
    readout.fit(x_train, y_train)
    predictions = readout.predict(x_eval)

    num_delays = targets.shape[1]
    mc_per_delay = np.zeros(num_delays)

    for column in range(num_delays):
        mc_per_delay[column] = squared_correlation(
            y_eval[:, column],
            predictions[:, column],
        )

    total_mc = float(np.sum(mc_per_delay))
    return total_mc, mc_per_delay


def generate_memcap_esn_2n_grid():
    """Original ESN MemCap grid, with actual reservoir size changed 100->200."""
    base_units = 100

    dynamic_values = [
        1.0,
        0.99,
        0.95,
        0.9,
        0.8,
        0.6,
        0.4,
        0.2,
        0.1,
        0.05,
        0.01,
        0.001,
        0.0001,
    ]

    input_scaling_values = [
        2.0,
        1.0,
        0.5,
        0.25,
    ]

    configurations = []

    for leaky, input_scaling in itertools.product(
        dynamic_values,
        input_scaling_values,
    ):
        configurations.append(
            (
                base_units,
                {
                    "units": 2 * base_units,
                    "leaky": leaky,
                    "input_scaling": input_scaling,
                    "spectral_radius": 0.9,
                    "bias_scaling": 0.0,
                    "return_sequences": True,
                },
            )
        )

    return configurations


def run_memcap_pair_trial(task, base_units, model_params, seed):
    tf.keras.backend.clear_session()
    set_seed(seed)

    model = ESN(**model_params)
    x = task["input"][np.newaxis, :, :]
    states_2n = np.asarray(model(x))[0]

    results = {}

    for baseline in BASELINES:
        states = representation(states_2n, baseline, base_units)
        validation_mc, validation_mc_per_delay = evaluate_memcap(
            states=states,
            targets=task["targets"],
            train_range=task["train_range"],
            evaluation_range=task["validation_range"],
        )

        results[baseline] = {
            "model": baseline,
            "base_units": base_units,
            "reservoir_units": int(model_params["units"]),
            "validation_mc": float(validation_mc),
            "validation_mc_per_delay": validation_mc_per_delay,
            "seed": seed,
            "model_params": dict(model_params),
        }

    return results


def grid_search_memcap_pair(
    task,
    search_seed=42,
    max_configurations=None,
    save_checkpoints=False,
):
    configurations = generate_memcap_esn_2n_grid()

    if max_configurations is not None:
        configurations = configurations[:max_configurations]

    num_configurations = len(configurations)
    print(f"Configurations: {num_configurations}")

    all_results = {baseline: [] for baseline in BASELINES}
    best = {baseline: None for baseline in BASELINES}

    for index, (base_units, model_params) in enumerate(configurations, start=1):
        # Same rule as the original ESN MemCap grid search.
        seed = search_seed + index

        pair = run_memcap_pair_trial(
            task=task,
            base_units=base_units,
            model_params=model_params,
            seed=seed,
        )

        new_best_messages = []

        for baseline in BASELINES:
            result = pair[baseline]
            result["configuration"] = index
            all_results[baseline].append(result)

            if (
                best[baseline] is None
                or result["validation_mc"] > best[baseline]["validation_mc"]
            ):
                best[baseline] = result
                new_best_messages.append(
                    f"{baseline}={result['validation_mc']:.4f}"
                )

        if new_best_messages:
            print(
                f"New best | {index}/{num_configurations} | "
                + " | ".join(new_best_messages)
            )
        elif index % 25 == 0 or index == num_configurations:
            print_pair_progress(
                index,
                num_configurations,
                best[BASELINE_FULL],
                best[BASELINE_SUBSET],
                metric="mc",
            )

        if save_checkpoints and (
            index % 25 == 0 or index == num_configurations
        ):
            path = ensure_dir(
                os.path.join(RESULTS_ROOT, "MemCap", "_checkpoint")
            )
            save_pickle(
                {
                    "completed_configurations": index,
                    "total_configurations": num_configurations,
                    "best": best,
                    "results": all_results,
                },
                os.path.join(path, "grid_search_checkpoint.pkl"),
            )

    return best, all_results


def final_memcap_baseline_evaluation(baseline, task, best_result, final_seeds):
    base_units = int(best_result["base_units"])
    model_params = dict(best_result["model_params"])

    test_mcs = []
    mc_curves = []
    guesses = []

    print(f"\n--- FINAL MEMCAP EVALUATION: {baseline} ---")
    print("Best hyperparameters:")
    print(best_result)

    for guess, seed in enumerate(final_seeds, start=1):
        tf.keras.backend.clear_session()
        set_seed(seed)

        model = ESN(**model_params)
        x = task["input"][np.newaxis, :, :]
        states_2n = np.asarray(model(x))[0]
        states = representation(states_2n, baseline, base_units)

        test_mc, test_mc_per_delay = evaluate_memcap(
            states=states,
            targets=task["targets"],
            train_range=task["train_range"],
            evaluation_range=task["test_range"],
        )

        test_mcs.append(test_mc)
        mc_curves.append(test_mc_per_delay)
        guesses.append(
            {
                "guess": guess,
                "seed": seed,
                "test_mc": float(test_mc),
                "mc_per_delay": test_mc_per_delay,
            }
        )

        print(
            f"Guess {guess}/{len(final_seeds)} | "
            f"Test MC: {test_mc:.4f} | Seed: {seed}"
        )

    test_mcs = np.asarray(test_mcs)
    mc_curves = np.asarray(mc_curves)

    return {
        "model": baseline,
        "base_units": base_units,
        "reservoir_units": int(model_params["units"]),
        "best_model_params": model_params,
        "best_validation_mc": float(best_result["validation_mc"]),
        "mean_test_mc": float(np.mean(test_mcs)),
        "std_test_mc": float(np.std(test_mcs)),
        "mean_mc_per_delay": np.mean(mc_curves, axis=0),
        "std_mc_per_delay": np.std(mc_curves, axis=0),
        "guesses": guesses,
    }


def save_memcap_baseline_results(
    baseline,
    best_result,
    search_results,
    final_result,
):
    path = ensure_dir(os.path.join(RESULTS_ROOT, "MemCap", baseline))

    save_pickle(best_result, os.path.join(path, "best_configuration.pkl"))
    save_pickle(search_results, os.path.join(path, "grid_search.pkl"))
    save_pickle(final_result, os.path.join(path, "final_evaluation.pkl"))
    save_csv(
        [flatten_result(row) for row in search_results],
        os.path.join(path, "grid_search.csv"),
    )

    guess_rows = [
        {
            "guess": row["guess"],
            "seed": row["seed"],
            "test_mc": row["test_mc"],
        }
        for row in final_result["guesses"]
    ]
    save_csv(guess_rows, os.path.join(path, "final_guesses.csv"))

    curve_rows = []
    for delay, (mean_mc, std_mc) in enumerate(
        zip(
            final_result["mean_mc_per_delay"],
            final_result["std_mc_per_delay"],
        ),
        start=1,
    ):
        curve_rows.append(
            {
                "delay": delay,
                "mean_mc": float(mean_mc),
                "std_mc": float(std_mc),
            }
        )
    save_csv(curve_rows, os.path.join(path, "mc_per_delay.csv"))


# ============================================================
# Complete battery
# ============================================================


def run_dataset_battery(summary_rows):
    if DRY_RUN:
        datasets = ["FordA"]
        num_trials = 12
        num_guesses = 2
    else:
        datasets = [
            "FordA",
            "FordB",
            "Adiac",
            "Libras",
            "RacketSports",
        ]
        num_trials = 400
        num_guesses = 5

    for dataset_name in datasets:
        print("\n" + "=" * 78)
        print(f"DATASET: {dataset_name}")
        print("=" * 78)

        try:
            (
                x_train_all,
                y_train_all,
                x_test,
                y_test,
                x_train,
                x_val,
                y_train,
                y_val,
            ) = load_dataset(dataset_name)

            print("Train shape:", x_train.shape)
            print("Validation shape:", x_val.shape)
            print("Test shape:", x_test.shape)

            best, all_results = random_search_dataset_pair(
                dataset_name=dataset_name,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                num_trials=num_trials,
                search_seed=SEARCH_SEED,
                max_time_seconds=DATASET_MAX_TIME_SECONDS,
                save_checkpoints=not DRY_RUN,
            )

            for baseline in BASELINES:
                print(f"\n--- BEST {baseline} / {dataset_name} ---")
                print(best[baseline])

                final_result = final_dataset_evaluation(
                    baseline=baseline,
                    best_result=best[baseline],
                    x_train_all=x_train_all,
                    y_train_all=y_train_all,
                    x_test=x_test,
                    y_test=y_test,
                    num_guesses=num_guesses,
                    # Same ESN final seed range as the dataset baseline.
                    start_seed=1000,
                )

                if not DRY_RUN:
                    save_dataset_baseline_results(
                        dataset_name=dataset_name,
                        baseline=baseline,
                        best_result=best[baseline],
                        search_results=all_results[baseline],
                        final_result=final_result,
                    )

                    summary_rows.append(
                        {
                            "experiment": "Dataset",
                            "setting": dataset_name,
                            "model": baseline,
                            "base_units": final_result["base_units"],
                            "reservoir_units": final_result["reservoir_units"],
                            "best_validation": best[baseline]["validation_accuracy"],
                            "mean_test": final_result["mean_test_accuracy"],
                            "std_test": final_result["std_test_accuracy"],
                        }
                    )

        except Exception as error:
            print("\n" + "!" * 78)
            print(f"ERROR while processing dataset {dataset_name}: {error}")
            print("Continuing with the next experiment...")
            print("!" * 78)


def run_synthetic_battery(summary_rows):
    if DRY_RUN:
        padding_lengths = [100]
        num_trials = 6
        final_seeds = [1001]
    else:
        padding_lengths = list(range(100, 1001, 100))
        num_trials = 400
        final_seeds = [1001, 1002, 1003, 1004, 1005]

    for padding_length in padding_lengths:
        print("\n" + "=" * 78)
        print(f"SYNTHETIC | PADDING: {padding_length}")
        print("=" * 78)

        task = generate_synthetic_ltm_task(
            padding_length=padding_length,
            seed=SEARCH_SEED,
        )

        best, all_results = random_search_synthetic_pair(
            task=task,
            num_trials=num_trials,
            search_seed=SEARCH_SEED,
            save_checkpoints=not DRY_RUN,
        )

        for baseline in BASELINES:
            print(f"\n--- BEST {baseline} / Synthetic {padding_length} ---")
            print(best[baseline])

            final_result = final_synthetic_baseline_evaluation(
                baseline=baseline,
                task=task,
                best_result=best[baseline],
                final_seeds=final_seeds,
            )

            if not DRY_RUN:
                save_synthetic_baseline_results(
                    padding_length=padding_length,
                    baseline=baseline,
                    best_result=best[baseline],
                    search_results=all_results[baseline],
                    final_result=final_result,
                )

                summary_rows.append(
                    {
                        "experiment": "Synthetic",
                        "setting": f"padding_{padding_length}",
                        "model": baseline,
                        "base_units": final_result["base_units"],
                        "reservoir_units": final_result["reservoir_units"],
                        "best_validation": best[baseline]["validation_accuracy"],
                        "mean_test": final_result["mean_test_accuracy"],
                        "std_test": final_result["std_test_accuracy"],
                    }
                )


def run_short_long_battery(summary_rows):
    if DRY_RUN:
        long_padding_lengths = [100, 1000]
        num_trials = 6
        final_seeds = [1001]
    else:
        long_padding_lengths = list(range(100, 1001, 100))
        num_trials = 400
        final_seeds = [1001, 1002, 1003, 1004, 1005]

    for long_padding_length in long_padding_lengths:
        print("\n" + "=" * 78)
        print(f"SHORT+LONG | LONG PADDING: {long_padding_length}")
        print("=" * 78)

        task = generate_contextual_short_long_task(
            long_padding_length=long_padding_length,
            seed=SEARCH_SEED,
        )

        print("Train:", task["x_train"].shape)
        print("Validation:", task["x_val"].shape)
        print("Test:", task["x_test"].shape)
        print("Classes:", task["num_classes"])

        best, all_results = random_search_short_long_pair(
            task=task,
            num_trials=num_trials,
            search_seed=SEARCH_SEED,
            save_checkpoints=not DRY_RUN,
        )

        for baseline in BASELINES:
            print(f"\n--- BEST {baseline} / Short+Long {long_padding_length} ---")
            print(best[baseline])

            final_result = final_short_long_baseline_evaluation(
                baseline=baseline,
                task=task,
                best_result=best[baseline],
                final_seeds=final_seeds,
            )

            if not DRY_RUN:
                save_short_long_baseline_results(
                    long_padding_length=long_padding_length,
                    baseline=baseline,
                    best_result=best[baseline],
                    search_results=all_results[baseline],
                    final_result=final_result,
                )

                summary_rows.append(
                    {
                        "experiment": "ShortLong",
                        "setting": f"long_{long_padding_length}",
                        "model": baseline,
                        "base_units": final_result["base_units"],
                        "reservoir_units": final_result["reservoir_units"],
                        "best_validation": best[baseline]["validation_accuracy"],
                        "mean_test": final_result["mean_test_accuracy"],
                        "std_test": final_result["std_test_accuracy"],
                        "mean_context_test": final_result[
                            "mean_context_test_accuracy"
                        ],
                        "mean_short_test": final_result[
                            "mean_short_test_accuracy"
                        ],
                    }
                )


def run_memcap_battery(summary_rows):
    task = generate_memcap_task()

    print("\n" + "=" * 78)
    print("MEMORY CAPACITY")
    print("=" * 78)

    if DRY_RUN:
        # Enough to verify the pipeline without running the full ESN grid.
        max_configurations = 8
        final_seeds = [1001]
    else:
        # Full original ESN grid: 13 leaky values x 4 input scalings = 52.
        max_configurations = None
        final_seeds = [1001, 1002, 1003, 1004, 1005]

    best, all_results = grid_search_memcap_pair(
        task=task,
        search_seed=SEARCH_SEED,
        max_configurations=max_configurations,
        save_checkpoints=not DRY_RUN,
    )

    for baseline in BASELINES:
        print(f"\n--- BEST {baseline} / MemCap ---")
        print(best[baseline])

        final_result = final_memcap_baseline_evaluation(
            baseline=baseline,
            task=task,
            best_result=best[baseline],
            final_seeds=final_seeds,
        )

        if not DRY_RUN:
            save_memcap_baseline_results(
                baseline=baseline,
                best_result=best[baseline],
                search_results=all_results[baseline],
                final_result=final_result,
            )

            summary_rows.append(
                {
                    "experiment": "MemCap",
                    "setting": "default",
                    "model": baseline,
                    "base_units": final_result["base_units"],
                    "reservoir_units": final_result["reservoir_units"],
                    "best_validation": best[baseline]["validation_mc"],
                    "mean_test": final_result["mean_test_mc"],
                    "std_test": final_result["std_test_mc"],
                }
            )


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
    print("\n" + "=" * 78)
    print("MIXED CONTROL BASELINES: ESN-2N AND ESN-2N->N")
    print("=" * 78)

    if DRY_RUN:
        print("*** DRY RUN MODE: no final results will be saved ***")
    else:
        print("*** FINAL RUN MODE ***")
        print(f"Results root: {RESULTS_ROOT}")

    summary_rows = []

    if RUN_DATASETS:
        run_dataset_battery(summary_rows)

    if RUN_SYNTHETIC:
        run_synthetic_battery(summary_rows)

    if RUN_SHORT_LONG:
        run_short_long_battery(summary_rows)

    if RUN_MEMCAP:
        run_memcap_battery(summary_rows)

    if not DRY_RUN:
        ensure_dir(RESULTS_ROOT)
        save_csv(summary_rows, os.path.join(RESULTS_ROOT, "summary.csv"))
        save_pickle(summary_rows, os.path.join(RESULTS_ROOT, "summary.pkl"))

        print("\n" + "=" * 78)
        print("ALL MIXED CONTROL BASELINES COMPLETED")
        print(f"Summary saved in: {RESULTS_ROOT}")
        print("=" * 78)
    else:
        print("\n" + "=" * 78)
        print("DRY RUN COMPLETED - NO FINAL RESULTS SAVED")
        print("=" * 78)
