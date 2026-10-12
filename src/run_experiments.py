"""
Experimental pipeline for real-world time-series classification.

This module handles:
- dataset loading
- reservoir construction
- reservoir state extraction
- RidgeClassifierCV readout training
- random-search model selection
- multi-seed final evaluation
- results saving

Supported reservoir architectures:
- ESN
- EuSN
- Hybrid E2SN
- Coupled E2SN
- Coupled-Symmetric E2SN
- Mixed E2SN
"""


import os
import pickle
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(
        gpu,
        True,
    )

import warnings
from scipy.linalg import LinAlgWarning

from sklearn.linear_model import RidgeClassifierCV

from src.reservoir import (
    ESN,
    EuSN,
    HybridReservoir,
    CoupledReservoir,
    CoupledSymmetricReservoir,
    MixedDynamicsReservoir,
)

# ============================================================
# Dataset loading
# ============================================================

def load_dataset(dataset_name, datasets_path="./data"):
    """
    Load a dataset stored in the format used by the
    EulerStateNetworks repository.

    Returns:
        x_train_all, y_train_all,
        x_test, y_test,
        x_train, x_val,
        y_train, y_val
    """

    dataset_file = os.path.join(
        datasets_path,
        dataset_name + "_dataset.p"
    )
    
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(
            f"Dataset not found: {dataset_file}"
        )

    with open(dataset_file, "rb") as f:
        dataset = pickle.load(f)

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
    
# ============================================================
# Results saving
# ============================================================

def get_results_path(
    dataset_name,
    model_type,
    results_root="./results",
):
    """
    Create and return the directory used to store the results
    for a specific dataset and reservoir model.
    """

    results_path = os.path.join(
        results_root,
        dataset_name,
        model_type,
    )

    os.makedirs(
        results_path,
        exist_ok=True,
    )

    return results_path


def save_model_selection_results(
    dataset_name,
    model_type,
    best_result,
    all_results,
    results_root="./results",
):
    """
    Save the complete random-search results and the best
    validation configuration.
    """

    results_path = get_results_path(
        dataset_name,
        model_type,
        results_root,
    )

    best_filename = os.path.join(
        results_path,
        "best_model_selection.pkl",
    )

    trials_filename = os.path.join(
        results_path,
        "model_selection_trials.pkl",
    )

    with open(best_filename, "wb") as f:
        pickle.dump(best_result, f)

    with open(trials_filename, "wb") as f:
        pickle.dump(all_results, f)

    print(
        f"Model selection results saved in: "
        f"{results_path}"
    )


def save_final_results(
    dataset_name,
    model_type,
    final_result,
    results_root="./results",
):
    """
    Save the final multi-guess test evaluation.
    """

    results_path = get_results_path(
        dataset_name,
        model_type,
        results_root,
    )

    final_filename = os.path.join(
        results_path,
        "final_evaluation.pkl",
    )

    with open(final_filename, "wb") as f:
        pickle.dump(final_result, f)

    print(
        f"Final evaluation saved in: "
        f"{final_filename}"
    )

# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    """
    Set NumPy and TensorFlow random seeds.
    """

    np.random.seed(seed)
    tf.random.set_seed(seed)

# ============================================================
# Hyperparameter sampling
# ============================================================

def sample_configuration(model_type, rng):
    """
    Sample one random reservoir configuration.

    The search ranges follow the experimental setup used
    in the original EulerStateNetworks repository.
    """

    if model_type == "ESN":

        return {
            "units": int(10 * rng.integers(1, 21)),
            "leaky": float(10.0 ** rng.integers(-5, 1)),
            "input_scaling": float(10.0 ** rng.integers(-3, 2)),
            "spectral_radius": float(0.1 * rng.integers(1, 16)),
            "bias_scaling": float(10.0 ** rng.integers(-3, 2)),
        }

    elif model_type == "EuSN":

        return {
            "units": int(10 * rng.integers(1, 21)),
            "gamma": float(10.0 ** rng.integers(-5, 1)),
            "epsilon": float(10.0 ** rng.integers(-5, 1)),
            "input_scaling": float(10.0 ** rng.integers(-3, 2)),
            "recurrent_scaling": float(10.0 ** rng.integers(-3, 2)),
            "bias_scaling": float(10.0 ** rng.integers(-3, 2)),
        }

    else:
        raise ValueError(
            f"Unknown model type: {model_type}"
        )
        
def generate_hybrid_configurations(
    seed=42,
    total_configurations=400,
):
    rng = np.random.default_rng(seed)

    esn_fraction_values = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ]

    base_count = (
        total_configurations
        // len(esn_fraction_values)
    )

    remainder = (
        total_configurations
        % len(esn_fraction_values)
    )

    configurations = []

    for i, esn_fraction in enumerate(
        esn_fraction_values
    ):

        num_configs = (
            base_count
            + (1 if i < remainder else 0)
        )

        for _ in range(num_configs):

            configurations.append({
                "units": int(
                    10 * rng.integers(1, 21)
                ),

                "esn_fraction":
                    esn_fraction,

                # ESN dynamics
                "leaky": float(
                    10.0 ** rng.integers(-5, 1)
                ),

                "spectral_radius": float(
                    0.1 * rng.integers(1, 16)
                ),

                # EuSN dynamics
                "gamma": float(
                    10.0 ** rng.integers(-5, 1)
                ),

                "epsilon": float(
                    10.0 ** rng.integers(-5, 1)
                ),

                "recurrent_scaling": float(
                    10.0 ** rng.integers(-3, 2)
                ),

                # Shared
                "input_scaling": float(
                    10.0 ** rng.integers(-3, 2)
                ),

                "bias_scaling": float(
                    10.0 ** rng.integers(-3, 2)
                ),
            })

    rng.shuffle(configurations)

    return configurations

def generate_coupled_configurations(
    seed=42,
    total_configurations=400,
):
    """
    Generate configurations for the Coupled reservoir.

    The search includes:
        - unidirectional coupling
        - bidirectional coupling
        - asymmetric coupling
        - symmetric coupling

    The fully uncoupled case (0, 0) is excluded.
    """

    rng = np.random.default_rng(seed)

    beta_values = [
        0.0,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
    ]

    beta_pairs = [
        (
            beta_esn_to_eusn,
            beta_eusn_to_esn,
        )
        for beta_esn_to_eusn in beta_values
        for beta_eusn_to_esn in beta_values
        if not (
            beta_esn_to_eusn == 0.0
            and beta_eusn_to_esn == 0.0
        )
    ]

    rng.shuffle(beta_pairs)

    base_count = (
        total_configurations
        // len(beta_pairs)
    )

    remainder = (
        total_configurations
        % len(beta_pairs)
    )

    configurations = []

    for i, (
        beta_esn_to_eusn,
        beta_eusn_to_esn,
    ) in enumerate(beta_pairs):

        num_configs = (
            base_count
            + (1 if i < remainder else 0)
        )

        for _ in range(num_configs):

            configurations.append({

                "units": int(
                    10 * rng.integers(1, 21)
                ),

                # Coupling
                "beta_esn_to_eusn":
                    beta_esn_to_eusn,

                "beta_eusn_to_esn":
                    beta_eusn_to_esn,

                # ESN dynamics
                "leaky": float(
                    10.0 ** rng.integers(-5, 1)
                ),

                "spectral_radius": float(
                    0.1 * rng.integers(1, 16)
                ),

                # EuSN dynamics
                "gamma": float(
                    10.0 ** rng.integers(-5, 1)
                ),

                "epsilon": float(
                    10.0 ** rng.integers(-5, 1)
                ),

                "recurrent_scaling": float(
                    10.0 ** rng.integers(-3, 2)
                ),

                # Shared
                "input_scaling": float(
                    10.0 ** rng.integers(-3, 2)
                ),

                "bias_scaling": float(
                    10.0 ** rng.integers(-3, 2)
                ),
            })

    rng.shuffle(configurations)

    return configurations

def generate_coupled_symmetric_configurations(
    seed=42,
    total_configurations=400,
):
    """
    Generate configurations for the Coupled-Symmetric reservoir.

    The search is stratified over the symmetric coupling strength:

        beta_ESN->EuSN = beta_EuSN->ESN = beta
    """

    rng = np.random.default_rng(seed)

    beta_values = [
        0.0,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
    ]

    base_count = (
        total_configurations
        // len(beta_values)
    )

    remainder = (
        total_configurations
        % len(beta_values)
    )

    configurations = []

    for i, coupling_strength in enumerate(
        beta_values
    ):

        num_configs = (
            base_count
            + (1 if i < remainder else 0)
        )

        for _ in range(num_configs):

            configurations.append({

                "units": int(
                    10 * rng.integers(1, 21)
                ),

                # Symmetric coupling
                "coupling_strength":
                    coupling_strength,

                # ESN dynamics
                "leaky": float(
                    10.0 ** rng.integers(-5, 1)
                ),

                "spectral_radius": float(
                    0.1 * rng.integers(1, 16)
                ),

                # EuSN dynamics
                "gamma": float(
                    10.0 ** rng.integers(-5, 1)
                ),

                "epsilon": float(
                    10.0 ** rng.integers(-5, 1)
                ),

                "recurrent_scaling": float(
                    10.0 ** rng.integers(-3, 2)
                ),

                # Shared
                "input_scaling": float(
                    10.0 ** rng.integers(-3, 2)
                ),

                "bias_scaling": float(
                    10.0 ** rng.integers(-3, 2)
                ),
            })

    rng.shuffle(configurations)

    return configurations

def generate_mixed_configurations(
    seed=42,
    total_configurations=400,
):
    """
    Generate configurations for the MixedDynamics reservoir.

    The search is stratified over non-trivial mixing values:
        0 < lambda < 1
    """

    rng = np.random.default_rng(seed)

    mixing_values = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ]

    base_count = (
        total_configurations
        // len(mixing_values)
    )

    remainder = (
        total_configurations
        % len(mixing_values)
    )

    configurations = []

    for i, mixing in enumerate(
        mixing_values
    ):

        num_configs = (
            base_count
            + (1 if i < remainder else 0)
        )

        for _ in range(num_configs):

            configurations.append({

                "units": int(
                    10 * rng.integers(1, 21)
                ),

                "mixing": mixing,

                # ESN dynamics
                "leaky": float(
                    10.0 ** rng.integers(-5, 1)
                ),

                "spectral_radius": float(
                    0.1 * rng.integers(1, 16)
                ),

                # EuSN dynamics
                "gamma": float(
                    10.0 ** rng.integers(-5, 1)
                ),

                "epsilon": float(
                    10.0 ** rng.integers(-5, 1)
                ),

                "recurrent_scaling": float(
                    10.0 ** rng.integers(-3, 2)
                ),

                # Shared
                "input_scaling": float(
                    10.0 ** rng.integers(-3, 2)
                ),

                "bias_scaling": float(
                    10.0 ** rng.integers(-3, 2)
                ),
            })

    rng.shuffle(configurations)

    return configurations

# ============================================================
# Reservoir construction
# ============================================================

def build_reservoir(model_type, **model_params):
    """
    Build the requested reservoir architecture.

    Supported models:
        - ESN
        - EuSN
        - HybridReservoir: DONE
        - CoupledReservoir
        - CoupledSymmetricReservoir
        - MixedDynamicsReservoir
    """

    if model_type == "ESN":
        return ESN(**model_params)

    if model_type == "EuSN":
        return EuSN(**model_params)
    
    if model_type == "Hybrid":
        return HybridReservoir(**model_params)

    if model_type == "Coupled":
        return CoupledReservoir(**model_params)

    if model_type == "Mixed":
        return MixedDynamicsReservoir(**model_params)
    
    if model_type == "CoupledSymmetric":
        return CoupledSymmetricReservoir(
            **model_params
        )
    
    raise ValueError(
        f"Unknown model type: {model_type}"
    )


# ============================================================
# Reservoir state extraction
# ============================================================

def extract_states(model, x):
    """
    Compute the final reservoir state h_T for each sequence.

    Input shape:
        (samples, timesteps, features)

    Output shape:
        (samples, reservoir_units)
    """

    states = model(x)

    return np.asarray(states)


# ============================================================
# Readout
# ============================================================

def train_readout(
    train_states,
    y_train,
    alphas=None,
):

    if alphas is None:
        alphas = np.logspace(-5, 5, 11)

    readout = RidgeClassifierCV(
        alphas=alphas,
        cv=None,
    )

    with warnings.catch_warnings():

        warnings.filterwarnings(
            "ignore",
            category=LinAlgWarning,
        )

        warnings.filterwarnings(
            "ignore",
            message="Singular matrix in solving dual problem.*",
            category=UserWarning,
        )

        readout.fit(
            train_states,
            y_train,
        )

    return readout


def evaluate_readout(
    readout,
    states,
    y,
):
    """
    Evaluate a fitted readout.
    """

    return readout.score(
        states,
        y,
    )


# ============================================================
# Single experiment
# ============================================================

def run_single_experiment(
    model_type,
    x_train,
    y_train,
    x_val,
    y_val,
    seed=42,
    readout_alphas=None,
    **model_params,
):
    """
    Run one ESN or EuSN experiment with a fixed configuration.

    Steps:
        1. set random seed
        2. build reservoir
        3. extract training states
        4. extract validation states
        5. train readout
        6. evaluate validation accuracy
    """

    tf.keras.backend.clear_session()

    set_seed(seed)

    model = build_reservoir(
        model_type,
        **model_params,
    )

    start_time = time.time()

    train_states = extract_states(
        model,
        x_train,
    )

    val_states = extract_states(
        model,
        x_val,
    )

    readout = train_readout(
        train_states,
        y_train,
        alphas=readout_alphas,
    )

    elapsed_time = (
        time.time() - start_time
    )

    validation_accuracy = evaluate_readout(
        readout,
        val_states,
        y_val,
    )

    return {
        "model": model_type,
        "validation_accuracy": validation_accuracy,
        "elapsed_time": elapsed_time,
        "seed": seed,
        "model_params": model_params,
        "readout_alpha": readout.alpha_,
    }

# ============================================================
# Random search
# ============================================================
def random_search(
    model_type,
    x_train,
    y_train,
    x_val,
    y_val,
    num_trials=5,
    search_seed=42,
    max_time_seconds=None,
    configurations=None,
):
    """
    Random search for reservoir hyperparameter selection.

    If configurations is None, configurations are sampled
    internally using sample_configuration().

    If configurations is provided, the supplied configurations
    are evaluated directly. This is useful for Hybrid, where
    configurations can be generated with a stratified search
    over esn_fraction.

    Each configuration is evaluated using one reservoir
    realization during model selection.
    """

    rng = np.random.default_rng(
        search_seed
    )

    # --------------------------------------------------------
    # Configuration generation
    # --------------------------------------------------------

    if configurations is None:

        configurations = [
            sample_configuration(
                model_type,
                rng,
            )
            for _ in range(num_trials)
        ]

    else:

        num_trials = len(
            configurations
        )

    results = []
    best_result = None

    start_time = time.time()

    print(
        f"Configurations: "
        f"{num_trials}"
    )

    # --------------------------------------------------------
    # Model selection
    # --------------------------------------------------------

    for trial, configuration in enumerate(
        configurations,
        start=1,
    ):

        # ----------------------------------------------------
        # Optional time limit
        # ----------------------------------------------------

        if max_time_seconds is not None:

            elapsed_time = (
                time.time()
                - start_time
            )

            if (
                elapsed_time
                >= max_time_seconds
            ):

                print(
                    "\nMaximum model-selection "
                    "time reached after "
                    f"{trial - 1} completed trials."
                )

                break

        # One reservoir realization per configuration
        trial_seed = (
            search_seed
            + trial
        )

        result = run_single_experiment(
            model_type=model_type,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            seed=trial_seed,
            **configuration,
        )

        result["trial"] = trial

        results.append(
            result
        )

        # ----------------------------------------------------
        # Best configuration
        # ----------------------------------------------------

        is_new_best = False

        if (
            best_result is None
            or result["validation_accuracy"]
            > best_result["validation_accuracy"]
        ):

            best_result = result
            is_new_best = True

        # ----------------------------------------------------
        # Terminal output
        # ----------------------------------------------------

        if is_new_best:

            params = result[
                "model_params"
            ]

            if model_type == "Hybrid":

                print(
                    f"New best | "
                    f"Trial {trial}/{num_trials} | "
                    f"Val acc: "
                    f"{result['validation_accuracy']:.4f} | "
                    f"ESN fraction: "
                    f"{params['esn_fraction']:.2f} | "
                    f"units: "
                    f"{params['units']}"
                )

            elif model_type == "Coupled":

                print(
                    f"New best | "
                    f"Trial {trial}/{num_trials} | "
                    f"Val acc: "
                    f"{result['validation_accuracy']:.4f} | "
                    f"beta E->U: "
                    f"{params['beta_esn_to_eusn']:.2f} | "
                    f"beta U->E: "
                    f"{params['beta_eusn_to_esn']:.2f} | "
                    f"units: "
                    f"{params['units']}"
                )
                
            elif model_type == "CoupledSymmetric":
                
                print(
                    f"New best | "
                    f"Trial {trial}/{num_trials} | "
                    f"Val acc: "
                    f"{result['validation_accuracy']:.4f} | "
                    f"beta: "
                    f"{params['coupling_strength']:.2f} | "
                    f"units: "
                    f"{params['units']}"
                )

            elif model_type == "Mixed":

                print(
                    f"New best | "
                    f"Trial {trial}/{num_trials} | "
                    f"Val acc: "
                    f"{result['validation_accuracy']:.4f} | "
                    f"lambda: "
                    f"{params['mixing']:.2f} | "
                    f"units: "
                    f"{params['units']}"
                )

            else:

                print(
                    f"New best | "
                    f"Trial {trial}/{num_trials} | "
                    f"Val acc: "
                    f"{result['validation_accuracy']:.4f}"
                )
        elif (
            trial % 25 == 0
            or trial == num_trials
        ):

            print(
                f"Progress | "
                f"Trial {trial}/{num_trials} | "
                f"Best val: "
                f"{best_result['validation_accuracy']:.4f}"
            )

    return best_result, results

# ============================================================
# Final evaluation
# ============================================================

def evaluate_best_configuration(
    best_result,
    x_train,
    y_train,
    x_test,
    y_test,
    num_guesses=5,
    start_seed=1000,
):
    """
    Evaluate the best reservoir configuration on the test set.

    The same hyperparameters are used for all guesses, while
    the random reservoir initialization changes at each guess.

    The RidgeClassifierCV readout is re-selected and trained
    independently for each reservoir realization.
    """

    model_type = best_result["model"]
    best_params = best_result["model_params"]

    test_accuracies = []
    elapsed_times = []
    guess_results = []

    print(
        f"\n--- FINAL EVALUATION: {model_type} ---"
    )

    print("Best hyperparameters:")
    print(best_params)

    for guess in range(1, num_guesses + 1):
        
        tf.keras.backend.clear_session()

        seed = start_seed + guess

        set_seed(seed)

        model = build_reservoir(
            model_type,
            **best_params,
        )

        start_time = time.time()

        # Reservoir representations
        train_states = extract_states(
            model,
            x_train,
        )

        test_states = extract_states(
            model,
            x_test,
        )

        # Select Ridge alpha using only the training data
        readout = train_readout(
            train_states,
            y_train,
        )

        test_accuracy = evaluate_readout(
            readout,
            test_states,
            y_test,
        )

        elapsed_time = time.time() - start_time

        test_accuracies.append(test_accuracy)
        elapsed_times.append(elapsed_time)

        guess_results.append({
            "guess": guess,
            "seed": seed,
            "test_accuracy": test_accuracy,
            "readout_alpha": readout.alpha_,
            "elapsed_time": elapsed_time,
        })

        print(
            f"Guess {guess}/{num_guesses} | "
            f"Test acc: {test_accuracy:.4f} | "
            f"Ridge alpha: {readout.alpha_} | "
            f"Seed: {seed}"
        )

    mean_accuracy = np.mean(test_accuracies)
    std_accuracy = np.std(test_accuracies)

    mean_time = np.mean(elapsed_times)
    std_time = np.std(elapsed_times)

    print(
        f"\n{model_type} FINAL RESULT | "
        f"Accuracy: {mean_accuracy:.4f} "
        f"+/- {std_accuracy:.4f}"
    )

    return {
        "model": model_type,
        "best_params": best_params,
        "num_guesses": num_guesses,
        "test_accuracies": test_accuracies,
        "mean_test_accuracy": mean_accuracy,
        "std_test_accuracy": std_accuracy,
        "mean_elapsed_time": mean_time,
        "std_elapsed_time": std_time,
        "guesses": guess_results,
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # Experiment configuration
    # ========================================================

    DRY_RUN = True

    MODELS = [
        "CoupledSymmetric"
    ]

    if DRY_RUN:

        DATASETS = [
            "FordA",
        ]

        NUM_TRIALS = 12
        NUM_GUESSES_FINAL = 2

        RESULTS_ROOT = (
            "./results_dry_run"
        )

    else:

        DATASETS = [
            "FordA",
            "FordB",
            "Adiac",
            "Libras",
            "RacketSports",
        ]

        NUM_TRIALS = 400
        NUM_GUESSES_FINAL = 5

        RESULTS_ROOT = (
            "./results"
        )


    MAX_TIME_HOURS = 10

    MAX_TIME_SECONDS = (
        MAX_TIME_HOURS
        * 60
        * 60
    )

    SEARCH_SEED = 42


    # Different seed ranges for final evaluation
    FINAL_START_SEEDS = {
        "CoupledSymmetric": 4000,
    }


    # ========================================================
    # Run datasets
    # ========================================================

    for dataset_name in DATASETS:

        print("\n" + "=" * 70)
        print(
            f"DATASET: {dataset_name}"
        )
        print("=" * 70)

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
            ) = load_dataset(
                dataset_name
            )

            print(
                "Train shape:",
                x_train.shape,
            )

            print(
                "Validation shape:",
                x_val.shape,
            )

            print(
                "Test shape:",
                x_test.shape,
            )


            # =================================================
            # Models
            # =================================================

            for model_type in MODELS:

                print(
                    "\n"
                    + "#" * 70
                )

                print(
                    f"MODEL: {model_type}"
                )

                print(
                    "#" * 70
                )


                # ---------------------------------------------
                # Generate configurations
                # ---------------------------------------------

                if model_type == "CoupledSymmetric":

                    configurations = (
                        generate_coupled_symmetric_configurations(
                            seed=SEARCH_SEED,
                            total_configurations=
                                NUM_TRIALS,
                        )
                    )

                elif model_type == "Mixed":

                    configurations = (
                        generate_mixed_configurations(
                            seed=SEARCH_SEED,
                            total_configurations=
                                NUM_TRIALS,
                        )
                    )

                else:

                    configurations = None


                # ---------------------------------------------
                # Model selection
                # ---------------------------------------------

                print(
                    f"\n--- RANDOM SEARCH: "
                    f"{model_type} ---"
                )

                (
                    best_result,
                    all_results,
                ) = random_search(

                    model_type=model_type,

                    x_train=x_train,
                    y_train=y_train,

                    x_val=x_val,
                    y_val=y_val,

                    search_seed=
                        SEARCH_SEED,

                    max_time_seconds=
                        MAX_TIME_SECONDS,

                    configurations=
                        configurations,
                )


                # ---------------------------------------------
                # Best configuration
                # ---------------------------------------------

                print(
                    f"\n--- BEST "
                    f"{model_type} "
                    f"CONFIGURATION ---"
                )

                print(
                    best_result
                )


                # ---------------------------------------------
                # Save model selection
                # ---------------------------------------------

                save_model_selection_results(

                    dataset_name=
                        dataset_name,

                    model_type=
                        model_type,

                    best_result=
                        best_result,

                    all_results=
                        all_results,

                    results_root=
                        RESULTS_ROOT,
                )


                # ---------------------------------------------
                # Final evaluation
                # ---------------------------------------------

                final_result = (
                    evaluate_best_configuration(

                        best_result=
                            best_result,

                        x_train=
                            x_train_all,

                        y_train=
                            y_train_all,

                        x_test=
                            x_test,

                        y_test=
                            y_test,

                        num_guesses=
                            NUM_GUESSES_FINAL,

                        start_seed=
                            FINAL_START_SEEDS[
                                model_type
                            ],
                    )
                )


                # ---------------------------------------------
                # Save final result
                # ---------------------------------------------

                save_final_results(

                    dataset_name=
                        dataset_name,

                    model_type=
                        model_type,

                    final_result=
                        final_result,

                    results_root=
                        RESULTS_ROOT,
                )


            print(
                "\n" + "-" * 70
            )

            print(
                f"COMPLETED DATASET: "
                f"{dataset_name}"
            )

            print(
                "-" * 70
            )


        except Exception as error:

            print(
                "\n" + "!" * 70
            )

            print(
                f"ERROR while processing "
                f"dataset: {dataset_name}"
            )

            print(
                error
            )

            print(
                "!" * 70
            )

"""
MAIN Hybrid experiments
if __name__ == "__main__":

    DRY_RUN = True

    if DRY_RUN:
        DATASETS = [
            "FordA",
        ]

        NUM_TRIALS = 9

    else:
        DATASETS = [
            "FordA",
            "FordB",
            "Adiac",
            "Libras",
            "RacketSports",
        ]

        NUM_TRIALS = 400

    NUM_GUESSES_FINAL = 5

    MAX_TIME_HOURS = 10
    MAX_TIME_SECONDS = (
        MAX_TIME_HOURS
        * 60
        * 60
    )

    SEARCH_SEED = 42

    for dataset_name in DATASETS:

        print("\n" + "=" * 70)
        print(f"DATASET: {dataset_name}")
        print("=" * 70)

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
            ) = load_dataset(
                dataset_name
            )

            print(
                "Train shape:",
                x_train.shape,
            )

            print(
                "Validation shape:",
                x_val.shape,
            )

            print(
                "Test shape:",
                x_test.shape,
            )

            # --------------------------------------------
            # Hybrid configurations
            # --------------------------------------------

            hybrid_configurations = (
                generate_hybrid_configurations(
                    seed=SEARCH_SEED,
                    total_configurations=
                        NUM_TRIALS,
                )
            )

            # --------------------------------------------
            # Model selection
            # --------------------------------------------

            print(
                "\n--- RANDOM SEARCH: Hybrid ---"
            )

            (
                best_hybrid,
                hybrid_results,
            ) = random_search(
                model_type="Hybrid",
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                search_seed=SEARCH_SEED,
                max_time_seconds=
                    MAX_TIME_SECONDS,
                configurations=
                    hybrid_configurations,
            )

            print(
                "\n--- BEST HYBRID "
                "CONFIGURATION ---"
            )

            print(
                best_hybrid
            )

            save_model_selection_results(
                dataset_name=
                    dataset_name,
                model_type="Hybrid",
                best_result=
                    best_hybrid,
                all_results=
                    hybrid_results,
            )

            # --------------------------------------------
            # Final evaluation
            # --------------------------------------------

            final_hybrid = (
                evaluate_best_configuration(
                    best_result=
                        best_hybrid,
                    x_train=
                        x_train_all,
                    y_train=
                        y_train_all,
                    x_test=
                        x_test,
                    y_test=
                        y_test,
                    num_guesses=
                        NUM_GUESSES_FINAL,
                    start_seed=3000,
                )
            )

            save_final_results(
                dataset_name=
                    dataset_name,
                model_type="Hybrid",
                final_result=
                    final_hybrid,
            )

        except Exception as error:

            print(
                f"ERROR on "
                f"{dataset_name}: "
                f"{error}"
            )

MAIN ESN experiments and EuSN experiments 
if __name__ == "__main__":

    # ============================================================
    # Experiment configuration
    # ============================================================

    DATASETS = [
        "FordA",
        "FordB",
        "Adiac",
        "Libras",
        "RacketSports",
    ]

    NUM_TRIALS = 400
    NUM_GUESSES_FINAL = 5

    MAX_TIME_HOURS = 10
    MAX_TIME_SECONDS = MAX_TIME_HOURS * 60 * 60

    SEARCH_SEED = 42

    # ============================================================
    # Run experiments on all datasets
    # ============================================================

    for dataset_name in DATASETS:

        print("\n" + "=" * 70)
        print(f"DATASET: {dataset_name}")
        print("=" * 70)

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

            # ====================================================
            # ESN - Model Selection
            # ====================================================

            print("\n--- RANDOM SEARCH: ESN ---")

            best_esn, esn_results = random_search(
                model_type="ESN",
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                num_trials=NUM_TRIALS,
                search_seed=SEARCH_SEED,
                max_time_seconds=MAX_TIME_SECONDS,
            )

            print("\n--- BEST ESN CONFIGURATION ---")
            print(best_esn)

            save_model_selection_results(
                dataset_name=dataset_name,
                model_type="ESN",
                best_result=best_esn,
                all_results=esn_results,
            )

            # ====================================================
            # ESN - Final Evaluation
            # ====================================================

            final_esn = evaluate_best_configuration(
                best_result=best_esn,
                x_train=x_train_all,
                y_train=y_train_all,
                x_test=x_test,
                y_test=y_test,
                num_guesses=NUM_GUESSES_FINAL,
                start_seed=1000,
            )

            save_final_results(
                dataset_name=dataset_name,
                model_type="ESN",
                final_result=final_esn,
            )

            # ====================================================
            # EuSN - Model Selection
            # ====================================================

            print("\n--- RANDOM SEARCH: EuSN ---")

            best_eusn, eusn_results = random_search(
                model_type="EuSN",
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                num_trials=NUM_TRIALS,
                search_seed=SEARCH_SEED,
                max_time_seconds=MAX_TIME_SECONDS,
            )

            print("\n--- BEST EuSN CONFIGURATION ---")
            print(best_eusn)

            save_model_selection_results(
                dataset_name=dataset_name,
                model_type="EuSN",
                best_result=best_eusn,
                all_results=eusn_results,
            )

            # ====================================================
            # EuSN - Final Evaluation
            # ====================================================

            final_eusn = evaluate_best_configuration(
                best_result=best_eusn,
                x_train=x_train_all,
                y_train=y_train_all,
                x_test=x_test,
                y_test=y_test,
                num_guesses=NUM_GUESSES_FINAL,
                start_seed=2000,
            )

            save_final_results(
                dataset_name=dataset_name,
                model_type="EuSN",
                final_result=final_eusn,
            )

            print("\n" + "-" * 70)
            print(f"COMPLETED DATASET: {dataset_name}")
            print("-" * 70)

        except Exception as error:

            print("\n" + "!" * 70)
            print(f"ERROR while processing dataset: {dataset_name}")
            print(error)
            print("Continuing with the next dataset...")
            print("!" * 70)
            
        """