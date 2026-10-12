import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import csv
import pickle

from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score

from src.memory_tasks import generate_synthetic_ltm_task
from src.reservoir import (
    ESN,
    EuSN,
    HybridReservoir,
    CoupledReservoir,
    CoupledSymmetricReservoir,
    MixedDynamicsReservoir,
)

# ============================================================
# GPU
# ============================================================

gpus = tf.config.list_physical_devices("GPU")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(
        gpu,
        True,
    )


# ============================================================
# Support functions
# ============================================================

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def extract_final_states(
    model,
    x,
):
    """
    Extract final reservoir state h_T.
    """

    states = model(x)

    return np.asarray(states)


def train_ridge_readout(
    x_train,
    y_train,
    num_classes,
    alpha,
):
    """
    Train a multi-output Ridge readout for multiclass
    classification.
    """

    y_train_one_hot = np.eye(
        num_classes,
        dtype=np.float32,
    )[y_train]

    readout = Ridge(
        alpha=alpha,
        solver="svd",
    )

    readout.fit(
        x_train,
        y_train_one_hot,
    )

    return readout

def select_ridge_alpha(
    h_train,
    y_train,
    h_val,
    y_val,
    num_classes,
):
    """
    Select Ridge regularization using validation accuracy.
    """

    alpha_values = np.logspace(
        -5,
        5,
        11,
    )

    best_alpha = None
    best_accuracy = -np.inf

    for alpha in alpha_values:

        readout = train_ridge_readout(
            x_train=h_train,
            y_train=y_train,
            num_classes=num_classes,
            alpha=alpha,
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


def evaluate_readout(
    readout,
    states,
    labels,
):
    """
    Predict class by argmax over Ridge outputs.
    """

    scores = readout.predict(
        states
    )

    predictions = np.argmax(
        scores,
        axis=1,
    )

    return accuracy_score(
        labels,
        predictions,
    )
    
def sample_log_uniform(
    rng,
    low,
    high,
):
    """
    Sample uniformly in logarithmic space.
    """

    return float(
        10 ** rng.uniform(
            np.log10(low),
            np.log10(high),
        )
    )

def build_reservoir(
    model_type,
    **model_params,
):

    if model_type == "ESN":
        return ESN(
            **model_params
        )

    if model_type == "EuSN":
        return EuSN(
            **model_params
        )

    if model_type == "Hybrid":
        return HybridReservoir(
            **model_params
        )

    if model_type == "Coupled":
        return CoupledReservoir(
            **model_params
        )
        
    if model_type == "CoupledSymmetric":
        return CoupledSymmetricReservoir(
            **model_params
        )

    if model_type == "Mixed":
        return MixedDynamicsReservoir(
            **model_params
        )

    raise ValueError(
        f"Unknown model type: {model_type}"
    )

def sample_synthetic_configuration(
    model_type,
    rng,
):
    """
    Sample reservoir hyperparameters using the ranges
    adopted for the LTM experiment in the EuSN paper.

    Reservoir size is fixed to 100.
    """

    if model_type == "ESN":

        return {
            "units": 100,

            "leaky": float(
                rng.uniform(
                    0.01,
                    1.0,
                )
            ),

            "input_scaling": float(
                rng.uniform(
                    0.01,
                    1.5,
                )
            ),

            "spectral_radius": float(
                rng.uniform(
                    0.01,
                    1.5,
                )
            ),

            "bias_scaling": float(
                rng.uniform(
                    0.01,
                    1.5,
                )
            ),

            "return_sequences": False,
        }

    if model_type == "EuSN":

        return {
            "units": 100,

            "gamma": sample_log_uniform(
                rng,
                1e-5,
                1e-1,
            ),

            "epsilon": sample_log_uniform(
                rng,
                1e-5,
                1e-1,
            ),

            "input_scaling": float(
                rng.uniform(
                    0.01,
                    1.5,
                )
            ),

            "recurrent_scaling": float(
                rng.uniform(
                    0.01,
                    1.5,
                )
            ),

            "bias_scaling": float(
                rng.uniform(
                    0.01,
                    1.5,
                )
            ),

            "return_sequences": False,
        }

    raise ValueError(
        f"Unknown model type: {model_type}"
    )
    
def generate_hybrid_synthetic_configurations(
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
                "units": 100,

                "esn_fraction":
                    esn_fraction,

                # ESN
                "leaky": float(
                    rng.uniform(
                        0.01,
                        1.0,
                    )
                ),

                "spectral_radius": float(
                    rng.uniform(
                        0.01,
                        1.5,
                    )
                ),

                # EuSN
                "gamma": sample_log_uniform(
                    rng,
                    1e-5,
                    1e-1,
                ),

                "epsilon": sample_log_uniform(
                    rng,
                    1e-5,
                    1e-1,
                ),

                "recurrent_scaling": float(
                    rng.uniform(
                        0.01,
                        1.5,
                    )
                ),

                # Shared
                "input_scaling": float(
                    rng.uniform(
                        0.01,
                        1.5,
                    )
                ),

                "bias_scaling": float(
                    rng.uniform(
                        0.01,
                        1.5,
                    )
                ),

                "return_sequences": False,
            })

    rng.shuffle(configurations)

    return configurations
    
def generate_coupled_synthetic_configurations(
    seed=42,
    total_configurations=400,
):
    """
    Generate stratified Coupled configurations for
    the Synthetic task.

    The completely uncoupled case (0, 0) is excluded.
    """

    rng = np.random.default_rng(
        seed
    )

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
        for beta_esn_to_eusn
        in beta_values
        for beta_eusn_to_esn
        in beta_values
        if not (
            beta_esn_to_eusn == 0.0
            and beta_eusn_to_esn == 0.0
        )
    ]

    rng.shuffle(
        beta_pairs
    )

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
    ) in enumerate(
        beta_pairs
    ):

        num_configs = (
            base_count
            + (
                1
                if i < remainder
                else 0
            )
        )

        for _ in range(
            num_configs
        ):

            configurations.append({
                "units": 100,

                # Coupling
                "beta_esn_to_eusn":
                    float(
                        beta_esn_to_eusn
                    ),

                "beta_eusn_to_esn":
                    float(
                        beta_eusn_to_esn
                    ),

                # ESN
                "leaky": float(
                    rng.uniform(
                        0.01,
                        1.0,
                    )
                ),

                "spectral_radius": float(
                    rng.uniform(
                        0.01,
                        1.5,
                    )
                ),

                # EuSN
                "gamma": sample_log_uniform(
                    rng,
                    1e-5,
                    1e-1,
                ),

                "epsilon": sample_log_uniform(
                    rng,
                    1e-5,
                    1e-1,
                ),

                "recurrent_scaling": float(
                    rng.uniform(
                        0.01,
                        1.5,
                    )
                ),

                # Shared
                "input_scaling": float(
                    rng.uniform(
                        0.01,
                        1.5,
                    )
                ),

                "bias_scaling": float(
                    rng.uniform(
                        0.01,
                        1.5,
                    )
                ),

                "return_sequences": False,
            })

    rng.shuffle(
        configurations
    )

    return configurations
    
def generate_coupled_symmetric_configurations(
    seed=42,
    total_configurations=400,
):
    """
    Generate configurations for the Coupled-Symmetric reservoir.

    The search is stratified over:

        beta_ESN->EuSN = beta_EuSN->ESN = beta

    Reservoir size is fixed to 100 units, consistently
    with the Synthetic experimental protocol.
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

                # Fixed by Synthetic protocol
                "units": 100,

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

def generate_mixed_synthetic_configurations(
    seed=42,
    total_configurations=400,
):
    """
    Generate stratified MixedDynamics configurations
    for the Synthetic task.

    Degenerate cases lambda = 0 and lambda = 1
    are excluded.
    """

    rng = np.random.default_rng(
        seed
    )

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
            + (
                1
                if i < remainder
                else 0
            )
        )

        for _ in range(
            num_configs
        ):

            configurations.append({
                "units": 100,

                # lambda
                "mixing":
                    float(mixing),

                # ESN
                "leaky": float(
                    rng.uniform(
                        0.01,
                        1.0,
                    )
                ),

                "spectral_radius": float(
                    rng.uniform(
                        0.01,
                        1.5,
                    )
                ),

                # EuSN
                "gamma": sample_log_uniform(
                    rng,
                    1e-5,
                    1e-1,
                ),

                "epsilon": sample_log_uniform(
                    rng,
                    1e-5,
                    1e-1,
                ),

                "recurrent_scaling": float(
                    rng.uniform(
                        0.01,
                        1.5,
                    )
                ),

                # Shared
                "input_scaling": float(
                    rng.uniform(
                        0.01,
                        1.5,
                    )
                ),

                "bias_scaling": float(
                    rng.uniform(
                        0.01,
                        1.5,
                    )
                ),

                "return_sequences": False,
            })

    rng.shuffle(
        configurations
    )

    return configurations
    
def run_synthetic_trial(
    model_type,
    task,
    model_params,
    seed,
):
    """
    Evaluate one reservoir configuration on the
    Synthetic-4 validation set.
    """
    
    tf.keras.backend.clear_session()

    set_seed(seed)

    model = build_reservoir(
        model_type,
        **model_params,
    )

    # --------------------------------------------------------
    # Reservoir states
    # --------------------------------------------------------

    h_train = extract_final_states(
        model,
        task["x_train"],
    )

    h_val = extract_final_states(
        model,
        task["x_val"],
    )

    # --------------------------------------------------------
    # Ridge model selection
    # --------------------------------------------------------

    best_alpha, validation_accuracy = select_ridge_alpha(
        h_train=h_train,
        y_train=task["y_train"],
        h_val=h_val,
        y_val=task["y_val"],
        num_classes=task["num_classes"],
    )

    return {
        "model": model_type,
        "model_params": model_params,
        "ridge_alpha": best_alpha,
        "validation_accuracy": validation_accuracy,
        "seed": seed,
    }

def random_search_synthetic(
    model_type,
    task,
    num_trials,
    search_seed=42,
):
    """
    Random search for the Synthetic-4 task.
    """

    rng = np.random.default_rng(
        search_seed
    )

    results = []
    best_result = None

    print(
        f"\n--- SYNTHETIC RANDOM SEARCH: "
        f"{model_type} ---"
    )

    print(
        f"Padding: {task['padding_length']}"
    )

    print(
        f"Trials: {num_trials}"
    )

    for trial in range(
        1,
        num_trials + 1,
    ):

        model_params = sample_synthetic_configuration(
            model_type,
            rng,
        )

        result = run_synthetic_trial(
            model_type=model_type,
            task=task,
            model_params=model_params,

            # one reservoir realization during
            # model selection
            seed=search_seed,
        )

        result["trial"] = trial

        results.append(
            result
        )

        is_new_best = False

        if (
            best_result is None
            or result["validation_accuracy"]
            > best_result["validation_accuracy"]
        ):

            best_result = result
            is_new_best = True

        if is_new_best:

            print(
                f"New best | "
                f"{trial}/{num_trials} | "
                f"Val acc: "
                f"{result['validation_accuracy']:.4f} | "
                f"alpha: "
                f"{result['ridge_alpha']:.1e}"
            )

        elif (
            trial % 25 == 0
            or trial == num_trials
        ):

            print(
                f"Progress | "
                f"{trial}/{num_trials} | "
                f"Best val: "
                f"{best_result['validation_accuracy']:.4f}"
            )
        
        if (
            trial % 25 == 0
            or trial == num_trials
        ):
            save_synthetic_checkpoint(
                model_type=model_type,
                padding_length=
                    task["padding_length"],
                completed_trials=trial,
                total_trials=num_trials,
                best_result=best_result,
                results=results,
            )

    return best_result, results

def random_search_synthetic_hybrid(
    task,
    num_trials=400,
    search_seed=42,
):

    configurations = (
        generate_hybrid_synthetic_configurations(
            seed=search_seed,
            total_configurations=num_trials,
        )
    )

    results = []
    best_result = None

    print(
        "\n--- SYNTHETIC RANDOM SEARCH: "
        "Hybrid ---"
    )

    print(
        f"Padding: {task['padding_length']}"
    )

    print(
        f"Trials: {len(configurations)}"
    )

    for trial, model_params in enumerate(
        configurations,
        start=1,
    ):

        result = run_synthetic_trial(
            model_type="Hybrid",
            task=task,
            model_params=model_params,
            seed=search_seed,
        )

        result["trial"] = trial

        results.append(
            result
        )

        is_new_best = False

        if (
            best_result is None
            or result["validation_accuracy"]
            > best_result["validation_accuracy"]
        ):
            best_result = result
            is_new_best = True

        if is_new_best:

            p = result[
                "model_params"
            ]["esn_fraction"]

            print(
                f"New best | "
                f"{trial}/{len(configurations)} | "
                f"Val acc: "
                f"{result['validation_accuracy']:.4f} | "
                f"ESN fraction: {p:.1f} | "
                f"alpha: "
                f"{result['ridge_alpha']:.1e}"
            )

        elif (
            trial % 25 == 0
            or trial == len(configurations)
        ):

            print(
                f"Progress | "
                f"{trial}/{len(configurations)} | "
                f"Best val: "
                f"{best_result['validation_accuracy']:.4f}"
            )

        if (
            trial % 25 == 0
            or trial == len(configurations)
        ):

            save_synthetic_checkpoint(
                model_type="Hybrid",
                padding_length=
                    task["padding_length"],
                completed_trials=trial,
                total_trials=
                    len(configurations),
                best_result=best_result,
                results=results,
            )

    return best_result, results

def random_search_synthetic_new_model(
    model_type,
    task,
    num_trials=400,
    search_seed=42,
    save_checkpoints=True,
):
    """
    Random search for Coupled and Mixed reservoirs
    on the Synthetic task.

    Model selection uses one fixed reservoir seed,
    consistently with the previous experiments.
    """

    if model_type == "Coupled":

        configurations = (
            generate_coupled_synthetic_configurations(
                seed=search_seed,
                total_configurations=
                    num_trials,
            )
        )
        
    elif model_type == "CoupledSymmetric":
        configurations = (
            generate_coupled_symmetric_configurations(
                seed=SEARCH_SEED,
                total_configurations=NUM_TRIALS,
            )
        )

    elif model_type == "Mixed":

        configurations = (
            generate_mixed_synthetic_configurations(
                seed=search_seed,
                total_configurations=
                    num_trials,
            )
        )

    else:

        raise ValueError(
            "This function supports only "
            "Coupled and Mixed."
        )

    results = []
    best_result = None

    print(
        f"\n--- SYNTHETIC RANDOM SEARCH: "
        f"{model_type} ---"
    )

    print(
        f"Padding: "
        f"{task['padding_length']}"
    )

    print(
        f"Trials: "
        f"{len(configurations)}"
    )

    for trial, model_params in enumerate(
        configurations,
        start=1,
    ):

        result = run_synthetic_trial(
            model_type=model_type,
            task=task,
            model_params=model_params,

            # Same realization during model selection
            seed=search_seed,
        )

        result[
            "trial"
        ] = trial

        results.append(
            result
        )

        is_new_best = False

        if (
            best_result is None
            or result["validation_accuracy"]
            > best_result["validation_accuracy"]
        ):

            best_result = result
            is_new_best = True

        if is_new_best:

            params = result[
                "model_params"
            ]

            if model_type == "Coupled":

                print(
                    f"New best | "
                    f"{trial}/"
                    f"{len(configurations)} | "
                    f"Val acc: "
                    f"{result['validation_accuracy']:.4f} | "
                    f"beta E->U: "
                    f"{params['beta_esn_to_eusn']:.2f} | "
                    f"beta U->E: "
                    f"{params['beta_eusn_to_esn']:.2f} | "
                    f"alpha: "
                    f"{result['ridge_alpha']:.1e}"
                )
                
            elif model_type == "CoupledSymmetric":

                print(
                    f"New best | "
                    f"Trial {trial}/{num_trials} | "
                    f"Val acc: "
                    f"{result['validation_accuracy']:.4f} | "
                    f"beta: "
                    f"{params['coupling_strength']:.2f}"
                )

            elif model_type == "Mixed":

                print(
                    f"New best | "
                    f"{trial}/"
                    f"{len(configurations)} | "
                    f"Val acc: "
                    f"{result['validation_accuracy']:.4f} | "
                    f"lambda: "
                    f"{params['mixing']:.2f} | "
                    f"alpha: "
                    f"{result['ridge_alpha']:.1e}"
                )

        elif (
            trial % 25 == 0
            or trial
            == len(configurations)
        ):

            print(
                f"Progress | "
                f"{trial}/"
                f"{len(configurations)} | "
                f"Best val: "
                f"{best_result['validation_accuracy']:.4f}"
            )

        if (
            save_checkpoints
            and (
                trial % 25 == 0
                or trial
                == len(configurations)
            )
        ):

            save_synthetic_checkpoint(
                model_type=model_type,
                padding_length=
                    task["padding_length"],
                completed_trials=
                    trial,
                total_trials=
                    len(configurations),
                best_result=
                    best_result,
                results=
                    results,
            )

    return (
        best_result,
        results,
    )

# ============================================================
# Results saving
# ============================================================

def get_synthetic_results_path(
    padding_length,
    model_type,
):
    path = os.path.join(
        "results",
        "Synthetic",
        f"padding_{padding_length}",
        model_type,
    )

    os.makedirs(
        path,
        exist_ok=True,
    )

    return path


def save_csv(
    rows,
    path,
):
    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        path,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def flatten_search_result(
    result,
):
    row = {
        key: value
        for key, value in result.items()
        if key != "model_params"
    }

    for key, value in result["model_params"].items():
        row[f"param_{key}"] = value

    return row


def save_synthetic_checkpoint(
    model_type,
    padding_length,
    completed_trials,
    total_trials,
    best_result,
    results,
):
    results_path = get_synthetic_results_path(
        padding_length,
        model_type,
    )

    checkpoint = {
        "model": model_type,
        "padding_length": padding_length,
        "completed_trials": completed_trials,
        "total_trials": total_trials,
        "best_result": best_result,
        "results": results,
    }

    with open(
        os.path.join(
            results_path,
            "random_search_checkpoint.pkl",
        ),
        "wb",
    ) as file:

        pickle.dump(
            checkpoint,
            file,
        )


def save_random_search_results(
    model_type,
    padding_length,
    best_result,
    results,
):
    results_path = get_synthetic_results_path(
        padding_length,
        model_type,
    )

    with open(
        os.path.join(
            results_path,
            "random_search.pkl",
        ),
        "wb",
    ) as file:

        pickle.dump(
            results,
            file,
        )

    with open(
        os.path.join(
            results_path,
            "best_configuration.pkl",
        ),
        "wb",
    ) as file:

        pickle.dump(
            best_result,
            file,
        )

    csv_rows = [
        flatten_search_result(result)
        for result in results
    ]

    save_csv(
        csv_rows,
        os.path.join(
            results_path,
            "random_search.csv",
        ),
    )


def save_final_synthetic_evaluation(
    final_result,
):
    results_path = get_synthetic_results_path(
        final_result["padding_length"],
        final_result["model"],
    )

    with open(
        os.path.join(
            results_path,
            "final_evaluation.pkl",
        ),
        "wb",
    ) as file:

        pickle.dump(
            final_result,
            file,
        )

    save_csv(
        final_result["guesses"],
        os.path.join(
            results_path,
            "final_guesses.csv",
        ),
    )


def update_synthetic_summary(
    new_row,
):
    path = os.path.join(
        "results",
        "Synthetic",
    )

    os.makedirs(
        path,
        exist_ok=True,
    )

    csv_path = os.path.join(
        path,
        "summary.csv",
    )

    rows = []

    if os.path.exists(
        csv_path
    ):

        with open(
            csv_path,
            "r",
            newline="",
        ) as file:

            reader = csv.DictReader(
                file
            )

            rows = list(
                reader
            )

    padding = str(
        new_row[
            "padding_length"
        ]
    )

    model = new_row[
        "model"
    ]

    rows = [
        row
        for row in rows
        if not (
            str(
                row[
                    "padding_length"
                ]
            )
            == padding
            and row[
                "model"
            ]
            == model
        )
    ]

    rows.append(
        new_row
    )

    model_order = {
        "ESN": 0,
        "EuSN": 1,
        "Hybrid": 2,
        "Coupled": 3,
        "Mixed": 4,
    }

    rows.sort(
        key=lambda row: (
            int(
                row[
                    "padding_length"
                ]
            ),
            model_order.get(
                row[
                    "model"
                ],
                99,
            ),
        )
    )

    fieldnames = [
        "padding_length",
        "model",
        "mean_test_accuracy",
        "std_test_accuracy",
        "best_validation_accuracy",
    ]

    with open(
        csv_path,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=
                fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )
    
# ============================================================
# Final evaluation
# ============================================================

def final_synthetic_evaluation(
    model_type,
    task,
    best_result,
    final_seeds,
):
    test_accuracies = []
    final_results = []

    print(
        f"\n--- FINAL SYNTHETIC EVALUATION: "
        f"{model_type} ---"
    )

    print(
        f"Padding: {task['padding_length']}"
    )

    print("Best hyperparameters:")
    print(
        best_result["model_params"]
    )

    for guess, seed in enumerate(
        final_seeds,
        start=1,
    ):

        tf.keras.backend.clear_session()

        set_seed(seed)

        model = build_reservoir(
            model_type,
            **best_result["model_params"],
        )

        h_train = extract_final_states(
            model,
            task["x_train"],
        )

        h_val = extract_final_states(
            model,
            task["x_val"],
        )

        h_test = extract_final_states(
            model,
            task["x_test"],
        )

        best_alpha, validation_accuracy = (
            select_ridge_alpha(
                h_train=h_train,
                y_train=task["y_train"],
                h_val=h_val,
                y_val=task["y_val"],
                num_classes=task["num_classes"],
            )
        )

        readout = train_ridge_readout(
            x_train=h_train,
            y_train=task["y_train"],
            num_classes=task["num_classes"],
            alpha=best_alpha,
        )

        test_accuracy = evaluate_readout(
            readout,
            h_test,
            task["y_test"],
        )

        test_accuracies.append(
            test_accuracy
        )

        final_results.append(
            {
                "guess": guess,
                "seed": seed,
                "ridge_alpha": best_alpha,
                "validation_accuracy":
                    validation_accuracy,
                "test_accuracy":
                    float(test_accuracy),
            }
        )

        print(
            f"Guess "
            f"{guess}/{len(final_seeds)} | "
            f"Test acc: "
            f"{test_accuracy:.4f} | "
            f"Val acc: "
            f"{validation_accuracy:.4f} | "
            f"alpha: "
            f"{best_alpha:.1e} | "
            f"Seed: {seed}"
        )

    mean_accuracy = float(
        np.mean(test_accuracies)
    )

    std_accuracy = float(
        np.std(test_accuracies)
    )

    print(
        f"\n{model_type} FINAL SYNTHETIC | "
        f"Padding: {task['padding_length']} | "
        f"Accuracy: "
        f"{mean_accuracy:.4f} "
        f"+/- {std_accuracy:.4f}"
    )

    return {
        "model": model_type,
        "padding_length":
            task["padding_length"],
        "best_model_params":
            best_result["model_params"],
        "mean_test_accuracy":
            mean_accuracy,
        "std_test_accuracy":
            std_accuracy,
        "guesses":
            final_results,
    }

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # Experiment configuration
    # ========================================================

    DRY_RUN = True

    SEARCH_SEED = 42

    MODELS_TO_RUN = [
    "CoupledSymmetric",
    ]

    # ========================================================
    # Dry / final configuration
    # ========================================================

    if DRY_RUN:

        PADDING_LENGTHS = [
            100,
        ]

        NUM_TRIALS = 6

        FINAL_SEEDS = [
            1001,
        ]

        print(
            "\n*** SYNTHETIC DRY RUN MODE ***"
        )

    else:

        PADDING_LENGTHS = list(
            range(
                100,
                1001,
                100,
            )
        )

        NUM_TRIALS = 400

        FINAL_SEEDS = [
            1001,
            1002,
            1003,
            1004,
            1005,
        ]

        print(
            "\n*** SYNTHETIC FINAL RUN MODE ***"
        )

    # ========================================================
    # Experiments
    # ========================================================

    for padding_length in PADDING_LENGTHS:

        print(
            "\n"
            "============================================"
        )

        print(
            f"PADDING LENGTH: "
            f"{padding_length}"
        )

        print(
            "============================================"
        )

        task = generate_synthetic_ltm_task(
            padding_length=
                padding_length,
            seed=
                SEARCH_SEED,
        )

        for model_type in MODELS_TO_RUN:

            # ------------------------------------------------
            # Model selection
            # ------------------------------------------------

            (
                best_result,
                search_results,
            ) = random_search_synthetic_new_model(
                model_type=
                    model_type,

                task=
                    task,

                num_trials=
                    NUM_TRIALS,

                search_seed=
                    SEARCH_SEED,

                save_checkpoints=
                    not DRY_RUN,
            )

            # ------------------------------------------------
            # Best configuration
            # ------------------------------------------------

            print(
                f"\n--- BEST "
                f"{model_type} ---"
            )

            print(
                f"Validation accuracy: "
                f"{best_result['validation_accuracy']:.4f}"
            )

            print(
                f"Ridge alpha: "
                f"{best_result['ridge_alpha']}"
            )

            print(
                best_result[
                    "model_params"
                ]
            )

            # ------------------------------------------------
            # Final evaluation
            # ------------------------------------------------

            final_result = (
                final_synthetic_evaluation(
                    model_type=
                        model_type,

                    task=
                        task,

                    best_result=
                        best_result,

                    final_seeds=
                        FINAL_SEEDS,
                )
            )

            # ------------------------------------------------
            # Save only real experiments
            # ------------------------------------------------

            if not DRY_RUN:

                save_random_search_results(
                    model_type=
                        model_type,

                    padding_length=
                        padding_length,

                    best_result=
                        best_result,

                    results=
                        search_results,
                )

                save_final_synthetic_evaluation(
                    final_result
                )

                summary_row = {
                    "padding_length":
                        padding_length,

                    "model":
                        model_type,

                    "mean_test_accuracy":
                        final_result[
                            "mean_test_accuracy"
                        ],

                    "std_test_accuracy":
                        final_result[
                            "std_test_accuracy"
                        ],

                    "best_validation_accuracy":
                        best_result[
                            "validation_accuracy"
                        ],
                }

                update_synthetic_summary(
                    summary_row
                )

    # ========================================================
    # End
    # ========================================================

    print(
        "\n"
        "============================================"
    )

    if DRY_RUN:

        print(
            "SYNTHETIC DRY RUN COMPLETED - "
            "NO FINAL RESULTS SAVED"
        )

    else:

        print(
            "COUPLED + MIXED SYNTHETIC "
            "EXPERIMENTS COMPLETED"
        )

    print(
        "============================================"
    )