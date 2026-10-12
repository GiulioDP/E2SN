import os
import csv
import pickle
import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score

from src.memory_tasks import generate_contextual_short_long_task
from src.reservoir import CoupledReservoir


# ============================================================
# GPU
# ============================================================

gpus = tf.config.list_physical_devices("GPU")

for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(
            gpu,
            True,
        )
    except RuntimeError:
        pass


# ============================================================
# CONFIGURATION
# ============================================================

DRY_RUN = True

SEARCH_SEED = 42

# Gli stessi 5 seed per OGNI configurazione.
MODEL_SELECTION_SEEDS = [
    42,
    43,
    44,
    45,
    46,
]

# Seed indipendenti per la valutazione finale.
FINAL_SEEDS = [
    1001,
    1002,
    1003,
    1004,
    1005,
]


if DRY_RUN:

    LONG_PADDING_LENGTHS = [
        700,
    ]

    NUM_TRIALS = 10

    FINAL_EVALUATION_SEEDS = [
        1001,
    ]

else:

    LONG_PADDING_LENGTHS = list(
        range(
            100,
            1001,
            100,
        )
    )

    NUM_TRIALS = 400

    FINAL_EVALUATION_SEEDS = (
        FINAL_SEEDS
    )


RESULTS_ROOT = os.path.join(
    "results",
    "diagnostics",
    "CoupledMultiSeed",
)


# ============================================================
# SEEDS
# ============================================================

def set_seed(seed):

    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# RESERVOIR
# ============================================================

def build_coupled(
    model_params,
):

    return CoupledReservoir(
        **model_params
    )


def extract_final_states(
    model,
    x,
):

    states = model(x)

    return np.asarray(
        states,
        dtype=np.float64,
    )


# ============================================================
# RIDGE READOUT
# ============================================================

def train_ridge_readout(
    x_train,
    y_train,
    num_classes,
    alpha,
):

    y_train_one_hot = np.eye(
        num_classes,
        dtype=np.float64,
    )[y_train]

    readout = Ridge(
        alpha=alpha,
        solver="cholesky",
    )

    readout.fit(
        x_train,
        y_train_one_hot,
    )

    return readout


def evaluate_readout(
    readout,
    states,
    labels,
):

    scores = readout.predict(
        states
    )

    predictions = np.argmax(
        scores,
        axis=1,
    )

    return float(
        accuracy_score(
            labels,
            predictions,
        )
    )


def select_ridge_alpha(
    h_train,
    y_train,
    h_val,
    y_val,
    num_classes,
):

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

    return (
        float(best_alpha),
        float(best_accuracy),
    )


# ============================================================
# DIAGNOSTIC PROBES
# ============================================================

def evaluate_binary_probe(
    h_train,
    y_train,
    h_val,
    y_val,
):

    _, best_accuracy = (
        select_ridge_alpha(
            h_train=h_train,
            y_train=y_train,
            h_val=h_val,
            y_val=y_val,
            num_classes=2,
        )
    )

    return float(
        best_accuracy
    )


def evaluate_binary_probe_test(
    h_train,
    y_train,
    h_val,
    y_val,
    h_test,
    y_test,
):

    best_alpha, _ = (
        select_ridge_alpha(
            h_train=h_train,
            y_train=y_train,
            h_val=h_val,
            y_val=y_val,
            num_classes=2,
        )
    )

    readout = train_ridge_readout(
        x_train=h_train,
        y_train=y_train,
        num_classes=2,
        alpha=best_alpha,
    )

    return evaluate_readout(
        readout,
        h_test,
        y_test,
    )


# ============================================================
# HYPERPARAMETER SAMPLING
# ============================================================

def sample_log_uniform(
    rng,
    low,
    high,
):

    return float(
        10 ** rng.uniform(
            np.log10(low),
            np.log10(high),
        )
    )


def sample_coupled_configuration(
    rng,
):

    """
    Same Coupled hyperparameter space used in the
    original Short+Long experiment.
    """

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

    (
        beta_esn_to_eusn,
        beta_eusn_to_esn,
    ) = beta_pairs[
        rng.integers(
            0,
            len(beta_pairs),
        )
    ]

    return {

        "units":
            100,

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
        "leaky":
            float(
                rng.uniform(
                    0.01,
                    1.0,
                )
            ),

        "spectral_radius":
            float(
                rng.uniform(
                    0.01,
                    1.5,
                )
            ),

        # EuSN
        "gamma":
            sample_log_uniform(
                rng,
                1e-5,
                1e-1,
            ),

        "epsilon":
            sample_log_uniform(
                rng,
                1e-5,
                1e-1,
            ),

        "recurrent_scaling":
            float(
                rng.uniform(
                    0.01,
                    1.5,
                )
            ),

        # Shared
        "input_scaling":
            float(
                rng.uniform(
                    0.01,
                    1.5,
                )
            ),

        "bias_scaling":
            float(
                rng.uniform(
                    0.01,
                    1.5,
                )
            ),

        "return_sequences":
            False,
    }


# ============================================================
# ONE VALIDATION RUN
# ============================================================

def run_short_long_trial(
    task,
    model_params,
    seed,
):

    tf.keras.backend.clear_session()

    set_seed(seed)

    model = build_coupled(
        model_params
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
    # Main 4-class task
    # --------------------------------------------------------

    best_alpha, validation_accuracy = (
        select_ridge_alpha(
            h_train=h_train,
            y_train=task["y_train"],
            h_val=h_val,
            y_val=task["y_val"],
            num_classes=task["num_classes"],
        )
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    context_accuracy = (
        evaluate_binary_probe(
            h_train=h_train,
            y_train=task["context_train"],
            h_val=h_val,
            y_val=task["context_val"],
        )
    )

    short_accuracy = (
        evaluate_binary_probe(
            h_train=h_train,
            y_train=task["short_train"],
            h_val=h_val,
            y_val=task["short_val"],
        )
    )

    return {

        "ridge_alpha":
            best_alpha,

        "validation_accuracy":
            validation_accuracy,

        "context_accuracy":
            context_accuracy,

        "short_accuracy":
            short_accuracy,

        "seed":
            seed,
    }


# ============================================================
# MULTI-SEED MODEL SELECTION
# ============================================================

def random_search_coupled_multiseed(
    task,
    num_trials,
    search_seed,
):

    rng = np.random.default_rng(
        search_seed
    )

    best_result = None
    results = []

    print()
    print("=" * 70)
    print("COUPLED MULTI-SEED MODEL SELECTION")
    print("=" * 70)

    print(
        "Padding:",
        task["long_padding_length"],
    )

    print(
        "Trials:",
        num_trials,
    )

    print(
        "Selection seeds:",
        MODEL_SELECTION_SEEDS,
    )

    for trial in range(
        1,
        num_trials + 1,
    ):

        # ----------------------------------------------------
        # ONE hyperparameter configuration
        # ----------------------------------------------------

        model_params = (
            sample_coupled_configuration(
                rng
            )
        )

        seed_results = []

        # ----------------------------------------------------
        # SAME configuration on FIVE reservoirs
        # ----------------------------------------------------

        for seed in MODEL_SELECTION_SEEDS:

            seed_result = (
                run_short_long_trial(
                    task=task,
                    model_params=model_params,
                    seed=seed,
                )
            )

            seed_results.append(
                seed_result
            )

        # ----------------------------------------------------
        # Aggregate
        # ----------------------------------------------------

        val_accuracies = [
            r["validation_accuracy"]
            for r in seed_results
        ]

        context_accuracies = [
            r["context_accuracy"]
            for r in seed_results
        ]

        short_accuracies = [
            r["short_accuracy"]
            for r in seed_results
        ]

        mean_val = float(
            np.mean(
                val_accuracies
            )
        )

        std_val = float(
            np.std(
                val_accuracies
            )
        )

        result = {

            "trial":
                trial,

            "model":
                "Coupled",

            "model_params":
                model_params,

            "selection_seeds":
                MODEL_SELECTION_SEEDS.copy(),

            # THIS is the model-selection criterion
            "validation_accuracy":
                mean_val,

            "validation_std":
                std_val,

            "validation_accuracies":
                val_accuracies,

            "mean_context_accuracy":
                float(
                    np.mean(
                        context_accuracies
                    )
                ),

            "std_context_accuracy":
                float(
                    np.std(
                        context_accuracies
                    )
                ),

            "mean_short_accuracy":
                float(
                    np.mean(
                        short_accuracies
                    )
                ),

            "std_short_accuracy":
                float(
                    np.std(
                        short_accuracies
                    )
                ),

            "ridge_alphas": [
                r["ridge_alpha"]
                for r in seed_results
            ],
        }

        results.append(
            result
        )

        is_new_best = (
            best_result is None
            or
            result["validation_accuracy"]
            >
            best_result["validation_accuracy"]
        )

        if is_new_best:

            best_result = result

            print()
            print(
                f"New best | "
                f"{trial}/{num_trials}"
            )

            print(
                f"Mean val: "
                f"{mean_val:.4f} "
                f"+/- {std_val:.4f}"
            )

            print(
                "Individual vals:",
                [
                    round(x, 4)
                    for x
                    in val_accuracies
                ],
            )

            print(
                "Params:",
                model_params,
            )

        elif (
            trial % 25 == 0
            or trial == num_trials
        ):

            print(
                f"Progress | "
                f"{trial}/{num_trials} | "
                f"Best mean val: "
                f"{best_result['validation_accuracy']:.4f} "
                f"+/- "
                f"{best_result['validation_std']:.4f}"
            )

    return (
        best_result,
        results,
    )


# ============================================================
# FINAL EVALUATION
# ============================================================

def final_short_long_evaluation(
    task,
    best_result,
    final_seeds,
):

    test_accuracies = []
    context_accuracies = []
    short_accuracies = []

    guesses = []

    print()
    print("=" * 70)
    print("FINAL SHORT+LONG EVALUATION")
    print("=" * 70)

    print(
        "Best hyperparameters:"
    )

    print(
        best_result[
            "model_params"
        ]
    )

    for guess, seed in enumerate(
        final_seeds,
        start=1,
    ):

        tf.keras.backend.clear_session()

        set_seed(seed)

        model = build_coupled(
            best_result[
                "model_params"
            ]
        )

        # ----------------------------------------------------
        # States
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Main readout
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Diagnostic probes
        # ----------------------------------------------------

        context_test_accuracy = (
            evaluate_binary_probe_test(
                h_train=h_train,
                y_train=task[
                    "context_train"
                ],
                h_val=h_val,
                y_val=task[
                    "context_val"
                ],
                h_test=h_test,
                y_test=task[
                    "context_test"
                ],
            )
        )

        short_test_accuracy = (
            evaluate_binary_probe_test(
                h_train=h_train,
                y_train=task[
                    "short_train"
                ],
                h_val=h_val,
                y_val=task[
                    "short_val"
                ],
                h_test=h_test,
                y_test=task[
                    "short_test"
                ],
            )
        )

        test_accuracies.append(
            test_accuracy
        )

        context_accuracies.append(
            context_test_accuracy
        )

        short_accuracies.append(
            short_test_accuracy
        )

        guesses.append({

            "guess":
                guess,

            "seed":
                seed,

            "ridge_alpha":
                best_alpha,

            "validation_accuracy":
                validation_accuracy,

            "test_accuracy":
                test_accuracy,

            "context_test_accuracy":
                context_test_accuracy,

            "short_test_accuracy":
                short_test_accuracy,
        })

        print(
            f"Guess "
            f"{guess}/{len(final_seeds)} | "
            f"Test: {test_accuracy:.4f} | "
            f"Val: {validation_accuracy:.4f} | "
            f"Context: {context_test_accuracy:.4f} | "
            f"Short: {short_test_accuracy:.4f} | "
            f"Seed: {seed}"
        )

    return {

        "mean_test_accuracy":
            float(
                np.mean(
                    test_accuracies
                )
            ),

        "std_test_accuracy":
            float(
                np.std(
                    test_accuracies
                )
            ),

        "mean_context_test_accuracy":
            float(
                np.mean(
                    context_accuracies
                )
            ),

        "std_context_test_accuracy":
            float(
                np.std(
                    context_accuracies
                )
            ),

        "mean_short_test_accuracy":
            float(
                np.mean(
                    short_accuracies
                )
            ),

        "std_short_test_accuracy":
            float(
                np.std(
                    short_accuracies
                )
            ),

        "guesses":
            guesses,
    }


# ============================================================
# SAVE
# ============================================================

def save_pickle(
    obj,
    path,
):

    with open(
        path,
        "wb",
    ) as f:

        pickle.dump(
            obj,
            f,
        )


def save_summary(
    rows,
):

    os.makedirs(
        RESULTS_ROOT,
        exist_ok=True,
    )

    filename = (
        "summary_dry.csv"
        if DRY_RUN
        else
        "summary.csv"
    )

    path = os.path.join(
        RESULTS_ROOT,
        filename,
    )

    fieldnames = [
        "padding",
        "mean_validation_accuracy",
        "std_validation_accuracy",
        "mean_test_accuracy",
        "std_test_accuracy",
        "gap_validation_test",
    ]

    with open(
        path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)

    if DRY_RUN:
        print(
            "*** COUPLED MULTI-SEED DRY RUN ***"
        )
    else:
        print(
            "*** COUPLED MULTI-SEED FULL RUN ***"
        )

    print("=" * 70)

    print(
        "Padding:",
        LONG_PADDING_LENGTHS,
    )

    print(
        "Trials per padding:",
        NUM_TRIALS,
    )

    print(
        "Model-selection seeds:",
        MODEL_SELECTION_SEEDS,
    )

    print(
        "Final seeds:",
        FINAL_EVALUATION_SEEDS,
    )

    os.makedirs(
        RESULTS_ROOT,
        exist_ok=True,
    )

    summary_rows = []

    # ========================================================
    # ALL PADDINGS
    # ========================================================

    for long_padding_length in (
        LONG_PADDING_LENGTHS
    ):

        print()
        print("=" * 70)

        print(
            f"LONG PADDING: "
            f"{long_padding_length}"
        )

        print("=" * 70)

        # Same task / same task seed as original experiment
        task = (
            generate_contextual_short_long_task(
                long_padding_length=
                    long_padding_length,

                seed=
                    SEARCH_SEED,
            )
        )

        print(
            "Train:",
            task["x_train"].shape,
        )

        print(
            "Validation:",
            task["x_val"].shape,
        )

        print(
            "Test:",
            task["x_test"].shape,
        )

        # ----------------------------------------------------
        # Model selection
        # ----------------------------------------------------

        (
            best_result,
            search_results,
        ) = random_search_coupled_multiseed(
            task=task,
            num_trials=NUM_TRIALS,
            search_seed=SEARCH_SEED,
        )

        print()
        print("=" * 70)
        print(
            "BEST CONFIGURATION"
        )
        print("=" * 70)

        print(
            f"Mean validation: "
            f"{best_result['validation_accuracy']:.4f} "
            f"+/- "
            f"{best_result['validation_std']:.4f}"
        )

        print(
            "Individual vals:",
            best_result[
                "validation_accuracies"
            ],
        )

        print(
            "Params:",
            best_result[
                "model_params"
            ],
        )

        # ----------------------------------------------------
        # Final evaluation
        # ----------------------------------------------------

        final_result = (
            final_short_long_evaluation(
                task=task,
                best_result=best_result,
                final_seeds=
                    FINAL_EVALUATION_SEEDS,
            )
        )

        mean_test = (
            final_result[
                "mean_test_accuracy"
            ]
        )

        std_test = (
            final_result[
                "std_test_accuracy"
            ]
        )

        gap = (
            best_result[
                "validation_accuracy"
            ]
            - mean_test
        )

        print()
        print("=" * 70)
        print(
            "FINAL COMPARISON"
        )
        print("=" * 70)

        print(
            f"Padding: "
            f"{long_padding_length}"
        )

        print(
            f"Validation: "
            f"{best_result['validation_accuracy']:.4f} "
            f"+/- "
            f"{best_result['validation_std']:.4f}"
        )

        print(
            f"Test: "
            f"{mean_test:.4f} "
            f"+/- "
            f"{std_test:.4f}"
        )

        print(
            f"Gap: "
            f"{gap:+.4f}"
        )

        # ----------------------------------------------------
        # Save separately from official results
        # ----------------------------------------------------

        output_dir = os.path.join(
            RESULTS_ROOT,
            f"long_{long_padding_length}",
        )

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        output = {

            "padding":
                long_padding_length,

            "num_trials":
                NUM_TRIALS,

            "model_selection_seeds":
                MODEL_SELECTION_SEEDS,

            "final_seeds":
                FINAL_EVALUATION_SEEDS,

            "best_result":
                best_result,

            "search_results":
                search_results,

            "final_result":
                final_result,
        }

        save_pickle(
            output,
            os.path.join(
                output_dir,
                (
                    "dry_run.pkl"
                    if DRY_RUN
                    else
                    "full_run.pkl"
                ),
            ),
        )

        summary_rows.append({

            "padding":
                long_padding_length,

            "mean_validation_accuracy":
                best_result[
                    "validation_accuracy"
                ],

            "std_validation_accuracy":
                best_result[
                    "validation_std"
                ],

            "mean_test_accuracy":
                mean_test,

            "std_test_accuracy":
                std_test,

            "gap_validation_test":
                gap,
        })

        # Save progressively
        save_summary(
            summary_rows
        )

    print()
    print("=" * 70)

    if DRY_RUN:
        print(
            "DRY RUN COMPLETED"
        )
    else:
        print(
            "FULL RUN COMPLETED"
        )

    print("=" * 70)
    print(
        "Results:",
        RESULTS_ROOT,
    )