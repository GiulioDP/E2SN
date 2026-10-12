import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import itertools
import numpy as np
import tensorflow as tf
import pickle
import csv

from sklearn.linear_model import Ridge

from src.memory_tasks import generate_memcap_task
from src.reservoir import (
    ESN,
    EuSN,
    HybridReservoir,
    CoupledReservoir,
    CoupledSymmetricReservoir,
    MixedDynamicsReservoir,
)


gpus = tf.config.list_physical_devices("GPU")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(
        gpu,
        True,
    )

# ============================================================
# Memory Capacity metric
# ============================================================

def squared_correlation(
    y_true,
    y_pred,
):
    """
    Compute the squared Pearson correlation coefficient.
    """

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if (
        np.std(y_true) == 0
        or np.std(y_pred) == 0
    ):
        return 0.0

    correlation = np.corrcoef(
        y_true,
        y_pred,
    )[0, 1]

    if not np.isfinite(correlation):
        return 0.0

    return float(correlation ** 2)


# ============================================================
# MemCap evaluation
# ============================================================

def evaluate_memcap(
    states,
    targets,
    train_range,
    evaluation_range,
):
    """
    Compute Memory Capacity on validation or test.

    A single multi-output linear readout reconstructs all
    delayed versions of the input simultaneously.
    """

    train_start, train_end = train_range
    eval_start, eval_end = evaluation_range

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    x_train = states[
        train_start:train_end
    ]

    y_train = targets[
        train_start:train_end
    ]

    # --------------------------------------------------------
    # Validation / test
    # --------------------------------------------------------

    x_eval = states[
        eval_start:eval_end
    ]

    y_eval = targets[
        eval_start:eval_end
    ]

    # --------------------------------------------------------
    # Linear readout
    # --------------------------------------------------------

    readout = Ridge(
        alpha=0.0,
        solver="svd",
    )

    readout.fit(
        x_train,
        y_train,
    )

    predictions = readout.predict(
        x_eval
    )

    # --------------------------------------------------------
    # MC_k for every delay
    # --------------------------------------------------------

    num_delays = targets.shape[1]

    mc_per_delay = np.zeros(
        num_delays
    )

    for column in range(num_delays):

        mc_per_delay[column] = squared_correlation(
            y_eval[:, column],
            predictions[:, column],
        )

    total_mc = np.sum(
        mc_per_delay
    )

    return total_mc, mc_per_delay

# ============================================================
# MemCap hyperparameter grid
# ============================================================

def generate_memcap_grid(
    model_type,
):
    """
    Generate the complete hyperparameter grid for MemCap.

    Fixed:
        - units = 100
        - bias_scaling = 0
        - ESN spectral_radius = 0.9
        - return_sequences = True
    """

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

    # ========================================================
    # ESN
    # ========================================================

    if model_type == "ESN":

        configurations = []

        for leaky, input_scaling in itertools.product(
            dynamic_values,
            input_scaling_values,
        ):

            configurations.append({
                "units": 100,
                "leaky": leaky,
                "input_scaling": input_scaling,
                "spectral_radius": 0.9,
                "bias_scaling": 0.0,
                "return_sequences": True,
            })

        return configurations

    # ========================================================
    # EuSN
    # ========================================================

    if model_type == "EuSN":

        recurrent_scaling_values = [
            2.0,
            1.0,
            0.5,
            0.25,
            0.1,
        ]

        configurations = []

        for (
            gamma,
            epsilon,
            input_scaling,
            recurrent_scaling,
        ) in itertools.product(
            dynamic_values,
            dynamic_values,
            input_scaling_values,
            recurrent_scaling_values,
        ):

            configurations.append({
                "units": 100,
                "gamma": gamma,
                "epsilon": epsilon,
                "input_scaling": input_scaling,
                "recurrent_scaling": recurrent_scaling,
                "bias_scaling": 0.0,
                "return_sequences": True,
            })

        return configurations

    raise ValueError(
        f"Unknown model type: {model_type}"
    )
    
def generate_hybrid_memcap_configurations(
    seed=42,
    total_configurations=3380,
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

    recurrent_scaling_values = [
        2.0,
        1.0,
        0.5,
        0.25,
        0.1,
    ]

    configurations = []

    base_count = (
        total_configurations
        // len(esn_fraction_values)
    )

    remainder = (
        total_configurations
        % len(esn_fraction_values)
    )

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

                # ESN dynamics
                "leaky": float(
                    rng.choice(dynamic_values)
                ),

                "spectral_radius": 0.9,

                # EuSN dynamics
                "gamma": float(
                    rng.choice(dynamic_values)
                ),

                "epsilon": float(
                    rng.choice(dynamic_values)
                ),

                "recurrent_scaling": float(
                    rng.choice(
                        recurrent_scaling_values
                    )
                ),

                # Shared
                "input_scaling": float(
                    rng.choice(
                        input_scaling_values
                    )
                ),

                "bias_scaling": 0.0,
                "return_sequences": True,
            })

    rng.shuffle(configurations)

    return configurations

def generate_coupled_memcap_configurations(
    seed=42,
    total_configurations=3380,
):
    """
    Generate Coupled reservoir configurations for MemCap.

    The search is stratified over coupling pairs.

    Fixed:
        - units = 100
        - spectral_radius = 0.9
        - bias_scaling = 0
        - return_sequences = True

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

    # Randomize the order so that a small dry run
    # does not always inspect the first coupling pairs.
    rng.shuffle(
        beta_pairs
    )

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

    recurrent_scaling_values = [
        2.0,
        1.0,
        0.5,
        0.25,
        0.1,
    ]

    configurations = []

    base_count = (
        total_configurations
        // len(beta_pairs)
    )

    remainder = (
        total_configurations
        % len(beta_pairs)
    )

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

                # ESN dynamics
                "leaky": float(
                    rng.choice(
                        dynamic_values
                    )
                ),

                "spectral_radius": 0.9,

                # EuSN dynamics
                "gamma": float(
                    rng.choice(
                        dynamic_values
                    )
                ),

                "epsilon": float(
                    rng.choice(
                        dynamic_values
                    )
                ),

                "recurrent_scaling": float(
                    rng.choice(
                        recurrent_scaling_values
                    )
                ),

                # Shared
                "input_scaling": float(
                    rng.choice(
                        input_scaling_values
                    )
                ),

                "bias_scaling": 0.0,

                "return_sequences": True,
            })

    rng.shuffle(
        configurations
    )

    return configurations

def generate_coupled_symmetric_memcap_configurations(
    seed=42,
    total_configurations=3380,
):
    """
    Generate CoupledSymmetric reservoir configurations for MemCap.

    The search is stratified over the symmetric coupling coefficient beta.

    Fixed:
        - units = 100
        - spectral_radius = 0.9
        - bias_scaling = 0
        - return_sequences = True

    beta = 0 is excluded, analogously to excluding (0, 0)
    in the asymmetric Coupled architecture.
    """

    rng = np.random.default_rng(seed)

    beta_values = [
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
    ]

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

    recurrent_scaling_values = [
        2.0,
        1.0,
        0.5,
        0.25,
        0.1,
    ]

    configurations = []

    base_count = (
        total_configurations
        // len(beta_values)
    )

    remainder = (
        total_configurations
        % len(beta_values)
    )

    for i, beta in enumerate(beta_values):

        num_configs = (
            base_count
            + (1 if i < remainder else 0)
        )

        for _ in range(num_configs):

            configurations.append({
                "units": 100,

                # Symmetric coupling
                "coupling_strength": float(beta),

                # ESN dynamics
                "leaky": float(
                    rng.choice(dynamic_values)
                ),

                "spectral_radius": 0.9,

                # EuSN dynamics
                "gamma": float(
                    rng.choice(dynamic_values)
                ),

                "epsilon": float(
                    rng.choice(dynamic_values)
                ),

                "recurrent_scaling": float(
                    rng.choice(
                        recurrent_scaling_values
                    )
                ),

                # Shared
                "input_scaling": float(
                    rng.choice(
                        input_scaling_values
                    )
                ),

                "bias_scaling": 0.0,

                "return_sequences": True,
            })

    rng.shuffle(configurations)

    return configurations

def generate_mixed_memcap_configurations(
    seed=42,
    total_configurations=3380,
):
    """
    Generate MixedDynamics reservoir configurations for MemCap.

    The search is stratified over the non-trivial mixing
    coefficient lambda:

        0 < lambda < 1

    Fixed:
        - units = 100
        - spectral_radius = 0.9
        - bias_scaling = 0
        - return_sequences = True
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

    recurrent_scaling_values = [
        2.0,
        1.0,
        0.5,
        0.25,
        0.1,
    ]

    configurations = []

    base_count = (
        total_configurations
        // len(mixing_values)
    )

    remainder = (
        total_configurations
        % len(mixing_values)
    )

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

                # ESN dynamics
                "leaky": float(
                    rng.choice(
                        dynamic_values
                    )
                ),

                "spectral_radius": 0.9,

                # EuSN dynamics
                "gamma": float(
                    rng.choice(
                        dynamic_values
                    )
                ),

                "epsilon": float(
                    rng.choice(
                        dynamic_values
                    )
                ),

                "recurrent_scaling": float(
                    rng.choice(
                        recurrent_scaling_values
                    )
                ),

                # Shared
                "input_scaling": float(
                    rng.choice(
                        input_scaling_values
                    )
                ),

                "bias_scaling": 0.0,

                "return_sequences": True,
            })

    rng.shuffle(
        configurations
    )

    return configurations

def random_search_memcap_hybrid(
    task,
    search_seed=42,
    total_configurations=3380,
    ):
    """
    Random search for the Hybrid reservoir on MemCap.

    The search is stratified by esn_fraction:
    each fraction receives the same number of configurations.
    """

    configurations = generate_hybrid_memcap_configurations(
        seed=search_seed,
        total_configurations=total_configurations,
    )

    num_configurations = len(
        configurations
    )

    print(
        "\n--- RANDOM SEARCH: Hybrid ---"
    )

    print(
        f"Configurations: {num_configurations}"
    )

    results = []
    best_result = None

    for index, configuration in enumerate(
        configurations,
        start=1,
    ):

        seed = search_seed + index

        result = run_memcap_experiment(
            model_type="Hybrid",
            task=task,
            seed=seed,
            **configuration,
        )

        result["configuration"] = index

        results.append(
            result
        )

        is_new_best = False

        if (
            best_result is None
            or result["validation_mc"]
            > best_result["validation_mc"]
        ):
            best_result = result
            is_new_best = True

        if is_new_best:

            p = result[
                "model_params"
            ]["esn_fraction"]

            print(
                f"New best | "
                f"{index}/{num_configurations} | "
                f"Validation MC: "
                f"{result['validation_mc']:.4f} | "
                f"ESN fraction: {p:.1f}"
            )

        elif (
            index % 25 == 0
            or index == num_configurations
        ):

            print(
                f"Progress | "
                f"{index}/{num_configurations} | "
                f"Best MC: "
                f"{best_result['validation_mc']:.4f}"
            )

            results_path = (
                get_memcap_results_path(
                    "Hybrid"
                )
            )

            checkpoint = {
                "model": "Hybrid",
                "completed_configurations":
                    index,
                "total_configurations":
                    num_configurations,
                "best_result":
                    best_result,
                "results":
                    results,
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

    return best_result, results

def random_search_memcap_new_model(
    model_type,
    task,
    search_seed=42,
    total_configurations=3380,
    save_checkpoints=True,
):
    """
    Random search for Coupled and MixedDynamics reservoirs
    on the Memory Capacity task.

    Both architectures receive the same total model-selection
    budget used for Hybrid.

    One independent reservoir realization is evaluated for
    each hyperparameter configuration.
    """

    if model_type == "Coupled":

        configurations = (
            generate_coupled_memcap_configurations(
                seed=search_seed,
                total_configurations=
                    total_configurations,
            )
        )

    elif model_type == "CoupledSymmetric":

        configurations = (
            generate_coupled_symmetric_memcap_configurations(
                seed=search_seed,
                total_configurations=total_configurations,
            )
        )

    elif model_type == "Mixed":

        configurations = (
            generate_mixed_memcap_configurations(
                seed=search_seed,
                total_configurations=
                    total_configurations,
            )
        )

    else:

        raise ValueError(
            "random_search_memcap_new_model "
            "supports only Coupled and Mixed."
        )

    num_configurations = len(
        configurations
    )

    print(
        f"\n--- RANDOM SEARCH: "
        f"{model_type} ---"
    )

    print(
        f"Configurations: "
        f"{num_configurations}"
    )

    results = []
    best_result = None

    for index, configuration in enumerate(
        configurations,
        start=1,
    ):

        # One reservoir realization
        # for each configuration.
        seed = (
            search_seed
            + index
        )

        result = run_memcap_experiment(
            model_type=model_type,
            task=task,
            seed=seed,
            **configuration,
        )

        result[
            "configuration"
        ] = index

        results.append(
            result
        )

        is_new_best = False

        if (
            best_result is None
            or result["validation_mc"]
            > best_result["validation_mc"]
        ):

            best_result = result
            is_new_best = True

        # ====================================================
        # Terminal output
        # ====================================================

        if is_new_best:

            params = result[
                "model_params"
            ]

            if model_type == "Coupled":

                print(
                    f"New best | "
                    f"{index}/"
                    f"{num_configurations} | "
                    f"Validation MC: "
                    f"{result['validation_mc']:.4f} | "
                    f"beta E->U: "
                    f"{params['beta_esn_to_eusn']:.2f} | "
                    f"beta U->E: "
                    f"{params['beta_eusn_to_esn']:.2f}"
                )
                
            elif model_type == "CoupledSymmetric":

                print(
                    f"New best | "
                    f"{index}/{num_configurations} | "
                    f"Validation MC: "
                    f"{result['validation_mc']:.4f} | "
                    f"beta:{params['coupling_strength']:.2f}"
                )

            elif model_type == "Mixed":

                print(
                    f"New best | "
                    f"{index}/"
                    f"{num_configurations} | "
                    f"Validation MC: "
                    f"{result['validation_mc']:.4f} | "
                    f"lambda: "
                    f"{params['mixing']:.2f}"
                )

        elif (
            index % 25 == 0
            or index == num_configurations
        ):

            print(
                f"Progress | "
                f"{index}/"
                f"{num_configurations} | "
                f"Best MC: "
                f"{best_result['validation_mc']:.4f}"
            )

        # ====================================================
        # Checkpoint
        # ====================================================

        if (
            save_checkpoints
            and (
                index % 25 == 0
                or index
                == num_configurations
            )
        ):

            results_path = (
                get_memcap_results_path(
                    model_type
                )
            )

            checkpoint = {
                "model":
                    model_type,

                "completed_configurations":
                    index,

                "total_configurations":
                    num_configurations,

                "best_result":
                    best_result,

                "results":
                    results,
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

    return (
        best_result,
        results,
    )


def run_memcap_experiment(
    model_type,
    task,
    seed,
    **model_params,
):
    """
    Evaluate one reservoir configuration on the
    MemCap validation set.
    """

    tf.keras.backend.clear_session()

    np.random.seed(
        seed
    )

    tf.random.set_seed(
        seed
    )

    model = build_reservoir(
        model_type,
        **model_params,
    )

    x = task["input"][
        np.newaxis,
        :,
        :
    ]

    states = model(
        x
    )

    states = np.asarray(
        states
    )[0]

    (
        validation_mc,
        validation_mc_per_delay,
    ) = evaluate_memcap(
        states=states,
        targets=task["targets"],
        train_range=
            task["train_range"],
        evaluation_range=
            task["validation_range"],
    )

    return {
        "model":
            model_type,

        "validation_mc":
            validation_mc,

        "validation_mc_per_delay":
            validation_mc_per_delay,

        "seed":
            seed,

        "model_params":
            model_params,
    }

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

    if model_type == "Mixed":
        return MixedDynamicsReservoir(
            **model_params
        )
        
    if model_type == "CoupledSymmetric":
        return CoupledSymmetricReservoir(
            **model_params
        )

    raise ValueError(
        f"Unknown model type: {model_type}"
    )

# ============================================================
# MemCap grid search
# ============================================================

def grid_search_memcap(
    model_type,
    task,
    search_seed=42,
    max_configurations=None,
):
    """
    Perform grid search for MemCap reservoir hyperparameters.

    One reservoir realization is evaluated for each
    hyperparameter configuration.

    The best configuration is selected according to
    validation Memory Capacity.
    """

    configurations = generate_memcap_grid(
        model_type
    )

    if max_configurations is not None:
        configurations = configurations[
            :max_configurations
        ]

    num_configurations = len(
        configurations
    )

    print(
        f"\n--- GRID SEARCH: {model_type} ---"
    )

    print(
        f"Configurations: {num_configurations}"
    )

    results = []
    best_result = None

    for index, configuration in enumerate(
        configurations,
        start=1,
    ):

        # One reservoir realization per configuration
        seed = search_seed + index

        result = run_memcap_experiment(
            model_type=model_type,
            task=task,
            seed=seed,
            **configuration,
        )

        result["configuration"] = index

        results.append(
            result
        )

        # ----------------------------------------------------
        # Best configuration
        # ----------------------------------------------------

        is_new_best = False

        if (
            best_result is None
            or result["validation_mc"]
            > best_result["validation_mc"]
        ):

            best_result = result
            is_new_best = True

        # ----------------------------------------------------
        # Terminal output
        # ----------------------------------------------------

        if is_new_best:

            print(
                f"New best | "
                f"{index}/{num_configurations} | "
                f"Validation MC: "
                f"{result['validation_mc']:.4f}"
            )

        elif (
            index % 25 == 0
            or index == num_configurations
        ):
            results_path = get_memcap_results_path(
                model_type
            )

            checkpoint = {
                "model": model_type,
                "completed_configurations": index,
                "total_configurations": num_configurations,
                "best_result": best_result,
                "results": results,
            }

            with open(
                os.path.join(
                    results_path,
                    "grid_search_checkpoint.pkl",
                ),
                "wb",
            ) as file:
                pickle.dump(
                    checkpoint,
                    file,
                )

    return best_result, results

# ============================================================
# Final MemCap evaluation
# ============================================================

def evaluate_best_memcap_configuration(
    best_result,
    task,
    final_seeds,
):
    """
    Evaluate the best MemCap reservoir configuration on the
    test set using multiple independent reservoir realizations.
    """

    model_type = best_result[
        "model"
    ]

    best_params = best_result[
        "model_params"
    ]

    test_mcs = []
    mc_curves = []
    guess_results = []

    print(
        f"\n--- FINAL MEMCAP EVALUATION: "
        f"{model_type} ---"
    )

    print(
        "Best hyperparameters:"
    )

    print(
        best_params
    )

    num_guesses = len(
        final_seeds
    )

    for guess, seed in enumerate(
        final_seeds,
        start=1,
    ):

        tf.keras.backend.clear_session()

        np.random.seed(
            seed
        )

        tf.random.set_seed(
            seed
        )

        model = build_reservoir(
            model_type,
            **best_params,
        )

        x = task["input"][
            np.newaxis,
            :,
            :
        ]

        states = model(
            x
        )

        states = np.asarray(
            states
        )[0]

        (
            test_mc,
            test_mc_per_delay,
        ) = evaluate_memcap(
            states=states,
            targets=task["targets"],
            train_range=
                task["train_range"],
            evaluation_range=
                task["test_range"],
        )

        test_mcs.append(
            test_mc
        )

        mc_curves.append(
            test_mc_per_delay
        )

        guess_results.append({
            "guess":
                guess,

            "seed":
                seed,

            "test_mc":
                test_mc,

            "mc_per_delay":
                test_mc_per_delay,
        })

        print(
            f"Guess "
            f"{guess}/{num_guesses} | "
            f"Test MC: "
            f"{test_mc:.4f} | "
            f"Seed: {seed}"
        )

    test_mcs = np.asarray(
        test_mcs
    )

    mc_curves = np.asarray(
        mc_curves
    )

    mean_mc = np.mean(
        test_mcs
    )

    std_mc = np.std(
        test_mcs
    )

    mean_mc_per_delay = np.mean(
        mc_curves,
        axis=0,
    )

    std_mc_per_delay = np.std(
        mc_curves,
        axis=0,
    )

    print(
        f"\n{model_type} FINAL MEMCAP | "
        f"MC: {mean_mc:.4f} "
        f"+/- {std_mc:.4f}"
    )

    return {
        "model":
            model_type,

        "best_params":
            best_params,

        "num_guesses":
            num_guesses,

        "test_mcs":
            test_mcs,

        "mean_test_mc":
            mean_mc,

        "std_test_mc":
            std_mc,

        "mean_mc_per_delay":
            mean_mc_per_delay,

        "std_mc_per_delay":
            std_mc_per_delay,

        "guesses":
            guess_results,
    }
    
# ============================================================
# Results saving
# ============================================================

def get_memcap_results_path(
    model_type,
    results_root="./results",
):
    """
    Return the directory used to store MemCap results.
    """

    path = os.path.join(
        results_root,
        "MemCap",
        model_type,
    )

    os.makedirs(
        path,
        exist_ok=True,
    )

    return path

def save_memcap_grid_search(
    model_type,
    best_result,
    all_results,
    results_root="./results",
):
    """
    Save complete MemCap grid-search results.
    """

    results_path = get_memcap_results_path(
        model_type,
        results_root,
    )

    # --------------------------------------------------------
    # Full Python objects
    # --------------------------------------------------------

    with open(
        os.path.join(
            results_path,
            "grid_search.pkl",
        ),
        "wb",
    ) as file:

        pickle.dump(
            all_results,
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

    # --------------------------------------------------------
    # Human-readable CSV
    # --------------------------------------------------------

    csv_path = os.path.join(
        results_path,
        "grid_search.csv",
    )

    rows = []

    for result in all_results:

        row = {
            "configuration": result["configuration"],
            "seed": result["seed"],
            "validation_mc": result["validation_mc"],
        }

        for key, value in result["model_params"].items():
            row[key] = value

        rows.append(row)

    if rows:

        fieldnames = list(
            rows[0].keys()
        )

        with open(
            csv_path,
            "w",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)

    print(
        f"Grid-search results saved in "
        f"{results_path}"
    )
    
def save_memcap_random_search(
    model_type,
    best_result,
    all_results,
    results_root="./results",
):
    """
    Save complete MemCap random-search results.
    """

    results_path = get_memcap_results_path(
        model_type,
        results_root,
    )

    # --------------------------------------------------------
    # Full Python objects
    # --------------------------------------------------------

    with open(
        os.path.join(
            results_path,
            "random_search.pkl",
        ),
        "wb",
    ) as file:

        pickle.dump(
            all_results,
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

    # --------------------------------------------------------
    # Human-readable CSV
    # --------------------------------------------------------

    csv_path = os.path.join(
        results_path,
        "random_search.csv",
    )

    rows = []

    for result in all_results:

        row = {
            "configuration":
                result["configuration"],
            "seed":
                result["seed"],
            "validation_mc":
                result["validation_mc"],
        }

        for key, value in (
            result["model_params"].items()
        ):
            row[key] = value

        rows.append(row)

    if rows:

        fieldnames = list(
            rows[0].keys()
        )

        with open(
            csv_path,
            "w",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)

    print(
        f"Random-search results saved in "
        f"{results_path}"
    )

def save_memcap_final_results(
    model_type,
    final_result,
    results_root="./results",
):
    """
    Save final MemCap test results.
    """

    results_path = get_memcap_results_path(
        model_type,
        results_root,
    )

    # --------------------------------------------------------
    # Complete result
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Individual guesses
    # --------------------------------------------------------

    guesses_path = os.path.join(
        results_path,
        "final_guesses.csv",
    )

    with open(
        guesses_path,
        "w",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "guess",
            "seed",
            "test_mc",
        ])

        for result in final_result["guesses"]:

            writer.writerow([
                result["guess"],
                result["seed"],
                result["test_mc"],
            ])

    # --------------------------------------------------------
    # MC as function of delay
    # --------------------------------------------------------

    curve_path = os.path.join(
        results_path,
        "mc_per_delay.csv",
    )

    with open(
        curve_path,
        "w",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "delay",
            "mean_mc",
            "std_mc",
        ])

        for delay, (
            mean_mc,
            std_mc,
        ) in enumerate(
            zip(
                final_result["mean_mc_per_delay"],
                final_result["std_mc_per_delay"],
            ),
            start=1,
        ):

            writer.writerow([
                delay,
                mean_mc,
                std_mc,
            ])

    print(
        f"Final evaluation saved in "
        f"{results_path}"
    )
    
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

    task = generate_memcap_task()

    # ========================================================
    # Dry / final configuration
    # ========================================================

    if DRY_RUN:

        TOTAL_CONFIGURATIONS = 20

        FINAL_SEEDS = [
            1001,
        ]

        print(
            "\n*** MEMCAP DRY RUN MODE ***"
        )

    else:

        # Same model-selection budget used for Hybrid.
        TOTAL_CONFIGURATIONS = 3380

        FINAL_SEEDS = [
            1001,
            1002,
            1003,
            1004,
            1005,
        ]

        print(
            "\n*** MEMCAP FINAL RUN MODE ***"
        )

    # ========================================================
    # New architectures
    # ========================================================

    for model_type in MODELS_TO_RUN:

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"MODEL: {model_type}"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Model selection
        # ----------------------------------------------------

        (
            best_result,
            search_results,
        ) = random_search_memcap_new_model(
            model_type=model_type,
            task=task,
            search_seed=SEARCH_SEED,
            total_configurations=
                TOTAL_CONFIGURATIONS,
            save_checkpoints=
                not DRY_RUN,
        )

        # ----------------------------------------------------
        # Best configuration
        # ----------------------------------------------------

        print(
            f"\n--- BEST "
            f"{model_type} "
            f"CONFIGURATION ---"
        )

        print(
            f"Validation MC: "
            f"{best_result['validation_mc']:.4f}"
        )

        print(
            best_result[
                "model_params"
            ]
        )

        # ----------------------------------------------------
        # Final multi-seed evaluation
        # ----------------------------------------------------

        final_result = (
            evaluate_best_memcap_configuration(
                best_result=
                    best_result,

                task=
                    task,

                final_seeds=
                    FINAL_SEEDS,
            )
        )

        # ----------------------------------------------------
        # Save only real experiments
        # ----------------------------------------------------

        if not DRY_RUN:

            save_memcap_random_search(
                model_type=
                    model_type,

                best_result=
                    best_result,

                all_results=
                    search_results,
            )

            save_memcap_final_results(
                model_type=
                    model_type,

                final_result=
                    final_result,
            )

    # ========================================================
    # End
    # ========================================================

    if DRY_RUN:

        print(
            "\n*** MEMCAP DRY RUN COMPLETED - "
            "NO FINAL RESULTS SAVED ***"
        )

    else:

        print(
            "\n*** MEMCAP FINAL RUN COMPLETED ***"
        )

        print(
            "Results saved in "
            "results/MemCap/"
        )


"""
if __name__ == "__main__":

    task = generate_memcap_task()

    SEARCH_SEED = 42

    FINAL_SEEDS = [
        1001,
        1002,
        1003,
        1004,
        1005,
    ]

    # ========================================================
    # HYBRID DRY RUN
    # ========================================================

    TOTAL_CONFIGURATIONS = 3380
    best_hybrid, hybrid_results = (
        random_search_memcap_hybrid(
            task=task,
            search_seed=SEARCH_SEED,
            total_configurations=TOTAL_CONFIGURATIONS,
        )
    )
    
    print(
        "\n--- BEST HYBRID CONFIGURATION ---"
    )

    print(
        f"Validation MC: "
        f"{best_hybrid['validation_mc']:.4f}"
    )

    print(
        best_hybrid["model_params"]
    )
    
    save_memcap_random_search(
        model_type="Hybrid",
        best_result=best_hybrid,
        all_results=hybrid_results,
    )

    final_hybrid = (
        evaluate_best_memcap_configuration(
            best_result=best_hybrid,
            task=task,
            final_seeds=FINAL_SEEDS,
        )
    )
    
    save_memcap_final_results(
        model_type="Hybrid",
        final_result=final_hybrid,
    )



if __name__ == "__main__":

    task = generate_memcap_task()

    SEARCH_SEED = 42
    
    FINAL_SEEDS = [
        1001,
        1002,
        1003,
        1004,
        1005,
    ]

    # ========================================================
    # DRY RUN
    # ========================================================

    MAX_CONFIGURATIONS = 3

    # ========================================================
    # ESN
    # ========================================================

    best_esn, esn_results = grid_search_memcap(
        model_type="ESN",
        task=task,
        search_seed=SEARCH_SEED,
        max_configurations=MAX_CONFIGURATIONS,
    )

    print(
        "\n--- BEST ESN CONFIGURATION ---"
    )

    print(
        f"Validation MC: "
        f"{best_esn['validation_mc']:.4f}"
    )

    print(
        best_esn["model_params"]
    )
    
    save_memcap_grid_search(
        model_type="ESN",
        best_result=best_esn,
        all_results=esn_results,
    )
    
    final_esn = evaluate_best_memcap_configuration(
        best_result=best_esn,
        task=task,
        final_seeds=FINAL_SEEDS,
    )
    
    save_memcap_final_results(
        model_type="ESN",
        final_result=final_esn,
    )

    # ========================================================
    # EuSN
    # ========================================================

    best_eusn, eusn_results = grid_search_memcap(
        model_type="EuSN",
        task=task,
        search_seed=SEARCH_SEED,
        max_configurations=MAX_CONFIGURATIONS,
    )

    print(
        "\n--- BEST EuSN CONFIGURATION ---"
    )

    print(
        f"Validation MC: "
        f"{best_eusn['validation_mc']:.4f}"
    )

    print(
        best_eusn["model_params"]
    )
    
    save_memcap_grid_search(
        model_type="EuSN",
        best_result=best_eusn,
        all_results=eusn_results,
    )

    final_eusn = evaluate_best_memcap_configuration(
        best_result=best_eusn,
        task=task,
        final_seeds=FINAL_SEEDS,
    )

    save_memcap_final_results(
        model_type="EuSN",
        final_result=final_eusn,
    )

"""
