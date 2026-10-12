import numpy as np
from sklearn.model_selection import train_test_split


# ============================================================
# Memory Capacity task
# ============================================================

def generate_memcap_task(
    sequence_length=7000,
    max_delay=200,
    warmup=100,
    train_end=5000,
    validation_end=6000,
    seed=42,
):
    """
    Generate the Memory Capacity task.

    Input:
        u(t) ~ Uniform[-0.8, 0.8]

    Target for every delay k:
        y_k(t) = u(t-k)

    A pre-history of max_delay samples is generated so that
    delayed targets are defined for every timestep.
    """

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Generate input + pre-history
    # --------------------------------------------------------

    raw_signal = rng.uniform(
        low=-0.8,
        high=0.8,
        size=(sequence_length + max_delay,),
    ).astype(np.float32)

    # Actual input presented to the reservoir
    u = raw_signal[
        max_delay:
    ].reshape(-1, 1)

    # --------------------------------------------------------
    # Build delayed targets
    # --------------------------------------------------------

    targets = np.zeros(
        shape=(sequence_length, max_delay),
        dtype=np.float32,
    )

    for delay in range(1, max_delay + 1):

        start = max_delay - delay
        end = start + sequence_length

        targets[:, delay - 1] = raw_signal[
            start:end
        ]

    # --------------------------------------------------------
    # Dataset ranges
    # --------------------------------------------------------

    return {
        "input": u,

        "targets": targets,

        "delays": np.arange(
            1,
            max_delay + 1
        ),

        "train_range": (
            warmup,
            train_end,
        ),

        "validation_range": (
            train_end,
            validation_end,
        ),

        "test_range": (
            validation_end,
            sequence_length,
        ),
    }

# ============================================================
# Synthetic Long-Term Memorization task
# ============================================================

def generate_synthetic_ltm_task(
    padding_length,
    num_samples=1000,
    seed=42,
):
    """
    Generate a 4-class synthetic Long-Term Memorization task.

    Each sequence starts with a class-dependent pattern of
    length 3, followed by random padding sampled uniformly
    from (0, 1).

    The task is to recover the class after processing the
    complete sequence.

    Dataset split:
        25% train
        25% validation
        50% test

    Parameters
    ----------
    padding_length : int
        Number of random timesteps after the informative pattern.

    num_samples : int
        Total number of sequences.

    seed : int
        Random seed.

    Returns
    -------
    dict
        Train, validation and test data plus task metadata.
    """

    if padding_length < 0:
        raise ValueError(
            "padding_length must be non-negative."
        )

    if num_samples % 4 != 0:
        raise ValueError(
            "num_samples must be divisible by 4 "
            "to create balanced classes."
        )

    rng = np.random.default_rng(seed)

    # ========================================================
    # Class patterns
    # ========================================================

    patterns = np.array([
        [ 1.0,  1.0,  1.0],
        [ 1.0, -1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0],
    ], dtype=np.float32)

    num_classes = len(patterns)
    pattern_length = patterns.shape[1]

    samples_per_class = (
        num_samples // num_classes
    )

    sequence_length = (
        pattern_length + padding_length
    )

    # ========================================================
    # Dataset allocation
    # ========================================================

    x = np.zeros(
        (
            num_samples,
            sequence_length,
            1,
        ),
        dtype=np.float32,
    )

    y = np.zeros(
        num_samples,
        dtype=np.int64,
    )

    # ========================================================
    # Generate sequences
    # ========================================================

    sample_index = 0

    for class_id, pattern in enumerate(
        patterns
    ):

        for _ in range(samples_per_class):

            # -----------------------------------------------
            # Informative pattern
            # -----------------------------------------------

            x[
                sample_index,
                :pattern_length,
                0,
            ] = pattern

            # -----------------------------------------------
            # Random padding
            # -----------------------------------------------

            if padding_length > 0:

                x[
                    sample_index,
                    pattern_length:,
                    0,
                ] = rng.uniform(
                    low=0.0,
                    high=1.0,
                    size=padding_length,
                )

            y[sample_index] = class_id

            sample_index += 1

    # ========================================================
    # Train / validation / test split
    # ========================================================

    # First:
    # 50% development set
    # 50% test set

    (
        x_dev,
        x_test,
        y_dev,
        y_test,
    ) = train_test_split(
        x,
        y,
        test_size=0.5,
        random_state=seed,
        stratify=y,
    )

    # Then divide development set:
    # 50% train
    # 50% validation
    #
    # Therefore overall:
    # 25% train
    # 25% validation
    # 50% test

    (
        x_train,
        x_val,
        y_train,
        y_val,
    ) = train_test_split(
        x_dev,
        y_dev,
        test_size=0.5,
        random_state=seed + 1,
        stratify=y_dev,
    )

    return {
        "x_train": x_train,
        "y_train": y_train,

        "x_val": x_val,
        "y_val": y_val,

        "x_test": x_test,
        "y_test": y_test,

        "patterns": patterns,

        "padding_length": padding_length,
        "pattern_length": pattern_length,
        "sequence_length": sequence_length,
        "num_classes": num_classes,
    }
    
def generate_contextual_short_long_task(
    long_padding_length,
    num_samples=1000,
    seed=42,
):
    """
    Contextual short-long memory task.

    A remote binary context must be combined with a
    short-term feature computed from recent continuous
    inputs.

    Final target:
    4-class joint label encoding both remote context
    and recent short-term feature.

    Recent feature uses values at delays:
        1, 3, 5, 10
    """

    if num_samples % 4 != 0:
        raise ValueError(
            "num_samples must be divisible by 4."
        )

    rng = np.random.default_rng(seed)

    context_length = 3
    recent_window_length = 10

    recent_delays = [
        1,
        3,
        5,
        10,
    ]

    # --------------------------------------------------------
    # Four balanced combinations:
    #
    # context   short
    #   +         +
    #   +         -
    #   -         +
    #   -         -
    # --------------------------------------------------------

    combinations = np.array(
        [
            [1, 1],
            [1, -1],
            [-1, 1],
            [-1, -1],
        ],
        dtype=np.int64,
    )

    repetitions = (
        num_samples // 4
    )

    combinations = np.repeat(
        combinations,
        repetitions,
        axis=0,
    )

    rng.shuffle(combinations)

    sequence_length = (
        context_length
        + long_padding_length
        + recent_window_length
        + 1
    )

    x = np.zeros(
        (
            num_samples,
            sequence_length,
            1,
        ),
        dtype=np.float32,
    )

    y = np.zeros(
        num_samples,
        dtype=np.int64,
    )

    context_labels = np.zeros(
        num_samples,
        dtype=np.int64,
    )

    short_labels = np.zeros(
        num_samples,
        dtype=np.int64,
    )

    joint_labels = np.zeros(
        num_samples,
        dtype=np.int64,
    )

    for i, (
        context_bit,
        desired_short_bit,
    ) in enumerate(combinations):

        # ----------------------------------------------------
        # Remote context
        # ----------------------------------------------------

        context = np.full(
            context_length,
            context_bit,
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Long distractor
        # ----------------------------------------------------

        long_noise = rng.uniform(
            -0.8,
            0.8,
            size=long_padding_length,
        ).astype(np.float32)

        # ----------------------------------------------------
        # Recent window
        #
        # Generate until its short-term feature has the
        # requested sign.
        # ----------------------------------------------------

        while True:

            recent = rng.uniform(
                -0.8,
                0.8,
                size=recent_window_length,
            ).astype(np.float32)

            short_score = (
                recent[-1]
                + recent[-3]
                - recent[-5]
                + recent[-10]
            )

            if short_score == 0:
                continue

            short_bit = (
                1
                if short_score > 0
                else -1
            )

            if (
                short_bit
                == desired_short_bit
            ):
                break

        # ----------------------------------------------------
        # One final irrelevant input.
        #
        # Therefore recent[-1], recent[-3], ...
        # correspond to delays 1, 3, 5, 10 from h_T.
        # ----------------------------------------------------

        final_distractor = rng.uniform(
            -0.8,
            0.8,
            size=1,
        ).astype(np.float32)

        sequence = np.concatenate(
            [
                context,
                long_noise,
                recent,
                final_distractor,
            ]
        )

        x[i, :, 0] = sequence

        # ----------------------------------------------------
        # Binary labels
        # ----------------------------------------------------

        context_label = (
            1
            if context_bit > 0
            else 0
        )

        short_label = (
            1
            if short_bit > 0
            else 0
        )

        # Joint 4-class target:
        #
        # context  short  -> class
        #    0       0    ->   0
        #    0       1    ->   1
        #    1       0    ->   2
        #    1       1    ->   3
        target = (
            2 * context_label
            + short_label
        )

        context_labels[i] = (
            context_label
        )

        short_labels[i] = (
            short_label
        )

        y[i] = target

        # Used only to guarantee balanced splitting
        joint_labels[i] = target
        
    # --------------------------------------------------------
    # Train / validation / test
    # --------------------------------------------------------

    (
        x_dev,
        x_test,
        y_dev,
        y_test,
        context_dev,
        context_test,
        short_dev,
        short_test,
        joint_dev,
        joint_test,
    ) = train_test_split(
        x,
        y,
        context_labels,
        short_labels,
        joint_labels,
        test_size=0.5,
        random_state=seed,
        stratify=joint_labels,
    )

    (
        x_train,
        x_val,
        y_train,
        y_val,
        context_train,
        context_val,
        short_train,
        short_val,
        joint_train,
        joint_val,
    ) = train_test_split(
        x_dev,
        y_dev,
        context_dev,
        short_dev,
        joint_dev,
        test_size=0.5,
        random_state=seed,
        stratify=joint_dev,
    )

    return {
        "x_train": x_train,
        "y_train": y_train,

        "x_val": x_val,
        "y_val": y_val,

        "x_test": x_test,
        "y_test": y_test,

        "context_train": context_train,
        "context_val": context_val,
        "context_test": context_test,

        "short_train": short_train,
        "short_val": short_val,
        "short_test": short_test,

        "num_classes": 4,

        "long_padding_length":
            long_padding_length,

        "recent_window_length":
            recent_window_length,

        "recent_delays":
            recent_delays,

        "sequence_length":
            sequence_length,
    }