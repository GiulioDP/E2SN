"""
Reservoir models used in the thesis experiments.

This module contains the baseline Echo State Network (ESN)
and Euler State Network (EuSN) reservoir implementations.

The implementations are adapted from the original
EulerStateNetworks repository, while keeping the reservoir
dynamics separated from the readout layer.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras


class EulerReservoirCell(keras.layers.Layer):
    """
    Recurrent cell used by the Euler State Network (EuSN).

    The state update is obtained through an Euler discretization
    of an ODE. The recurrent matrix is constructed from an
    antisymmetric component plus a controlled dissipation term.
    """

    def __init__(
        self,
        units,
        input_scaling=1.0,
        bias_scaling=1.0,
        recurrent_scaling=1.0,
        epsilon=0.01,
        gamma=0.001,
        activation=tf.nn.tanh,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.units = units
        self.state_size = units

        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.recurrent_scaling = recurrent_scaling

        self.epsilon = epsilon
        self.gamma = gamma
        self.activation = activation


    def build(self, input_shape):

        # Random matrix used to construct the antisymmetric component.
        W = tf.random.uniform(
            shape=(self.units, self.units),
            minval=-self.recurrent_scaling,
            maxval=self.recurrent_scaling,
        )

        I = tf.linalg.eye(self.units)

        # W - W^T is antisymmetric.
        # -gamma I introduces controlled dissipation.
        self.recurrent_kernel = (
            W - tf.transpose(W) - self.gamma * I
        )

        self.kernel = tf.random.uniform(
            shape=(input_shape[-1], self.units),
            minval=-self.input_scaling,
            maxval=self.input_scaling,
        )

        self.bias = tf.random.uniform(
            shape=(self.units,),
            minval=-self.bias_scaling,
            maxval=self.bias_scaling,
        )

        self.built = True


    def call(self, inputs, states):

        previous_state = states[0]

        input_part = tf.matmul(inputs, self.kernel)
        recurrent_part = tf.matmul(
            previous_state,
            self.recurrent_kernel,
        )

        pre_activation = (
            input_part
            + recurrent_part
            + self.bias
        )

        if self.activation is not None:
            delta = self.activation(pre_activation)
        else:
            delta = pre_activation

        output = previous_state + self.epsilon * delta

        return output, [output]


class ReservoirCell(keras.layers.Layer):
    """
    Standard Echo State Network recurrent cell.
    """

    def __init__(
        self,
        units,
        input_scaling=1.0,
        bias_scaling=1.0,
        spectral_radius=0.99,
        leaky=1.0,
        activation=tf.nn.tanh,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.units = units
        self.state_size = units

        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky

        self.activation = activation


    def build(self, input_shape):

        # Approximate spectral-radius initialization based
        # on the circular law used in the original repository.
        value = (
            self.spectral_radius
            / np.sqrt(self.units)
        ) * (6 / np.sqrt(12))

        self.recurrent_kernel = tf.random.uniform(
            shape=(self.units, self.units),
            minval=-value,
            maxval=value,
        )

        self.kernel = tf.random.uniform(
            shape=(input_shape[-1], self.units),
            minval=-self.input_scaling,
            maxval=self.input_scaling,
        )

        self.bias = tf.random.uniform(
            shape=(self.units,),
            minval=-self.bias_scaling,
            maxval=self.bias_scaling,
        )

        self.built = True


    def call(self, inputs, states):

        previous_state = states[0]

        input_part = tf.matmul(inputs, self.kernel)
        recurrent_part = tf.matmul(
            previous_state,
            self.recurrent_kernel,
        )

        pre_activation = (
            input_part
            + recurrent_part
            + self.bias
        )

        if self.activation is not None:
            candidate_state = self.activation(pre_activation)
        else:
            candidate_state = pre_activation

        output = (
            (1.0 - self.leaky) * previous_state
            + self.leaky * candidate_state
        )

        return output, [output]

class HybridReservoirCell(keras.layers.Layer):
    """
    Heterogeneous reservoir combining ESN and EuSN neurons.

    The state is partitioned into two independent recurrent
    populations with different dynamics.

    esn_fraction controls the proportion of ESN neurons.
    """

    def __init__(
        self,
        units,
        esn_fraction,
        input_scaling=1.0,
        bias_scaling=1.0,
        spectral_radius=0.99,
        leaky=1.0,
        recurrent_scaling=1.0,
        epsilon=0.01,
        gamma=0.001,
        activation=tf.nn.tanh,
        **kwargs
    ):
        super().__init__(**kwargs)

        if not 0.0 <= esn_fraction <= 1.0:
            raise ValueError(
                "esn_fraction must be between 0 and 1."
            )

        self.units = units
        self.state_size = units

        self.esn_fraction = esn_fraction

        self.num_esn_units = int(
            round(units * esn_fraction)
        )

        self.num_eusn_units = (
            units - self.num_esn_units
        )

        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling

        self.spectral_radius = spectral_radius
        self.leaky = leaky

        self.recurrent_scaling = recurrent_scaling
        self.epsilon = epsilon
        self.gamma = gamma

        self.activation = activation


    def build(self, input_shape):

        input_dim = input_shape[-1]

        # =====================================================
        # ESN block
        # =====================================================

        if self.num_esn_units > 0:

            value = (
                self.spectral_radius
                / np.sqrt(self.num_esn_units)
            ) * (6 / np.sqrt(12))

            self.esn_recurrent_kernel = tf.random.uniform(
                shape=(
                    self.num_esn_units,
                    self.num_esn_units,
                ),
                minval=-value,
                maxval=value,
            )

            self.esn_kernel = tf.random.uniform(
                shape=(
                    input_dim,
                    self.num_esn_units,
                ),
                minval=-self.input_scaling,
                maxval=self.input_scaling,
            )

            self.esn_bias = tf.random.uniform(
                shape=(self.num_esn_units,),
                minval=-self.bias_scaling,
                maxval=self.bias_scaling,
            )

        else:

            self.esn_recurrent_kernel = None
            self.esn_kernel = None
            self.esn_bias = None


        # =====================================================
        # EuSN block
        # =====================================================

        if self.num_eusn_units > 0:

            W = tf.random.uniform(
                shape=(
                    self.num_eusn_units,
                    self.num_eusn_units,
                ),
                minval=-self.recurrent_scaling,
                maxval=self.recurrent_scaling,
            )

            I = tf.linalg.eye(
                self.num_eusn_units
            )

            self.eusn_recurrent_kernel = (
                W
                - tf.transpose(W)
                - self.gamma * I
            )

            self.eusn_kernel = tf.random.uniform(
                shape=(
                    input_dim,
                    self.num_eusn_units,
                ),
                minval=-self.input_scaling,
                maxval=self.input_scaling,
            )

            self.eusn_bias = tf.random.uniform(
                shape=(self.num_eusn_units,),
                minval=-self.bias_scaling,
                maxval=self.bias_scaling,
            )

        else:

            self.eusn_recurrent_kernel = None
            self.eusn_kernel = None
            self.eusn_bias = None

        self.built = True


    def call(self, inputs, states):

        previous_state = states[0]

        esn_previous_state = (
            previous_state[
                :,
                :self.num_esn_units
            ]
        )

        eusn_previous_state = (
            previous_state[
                :,
                self.num_esn_units:
            ]
        )


        # =====================================================
        # ESN update
        # =====================================================

        if self.num_esn_units > 0:

            esn_input_part = tf.matmul(
                inputs,
                self.esn_kernel,
            )

            esn_recurrent_part = tf.matmul(
                esn_previous_state,
                self.esn_recurrent_kernel,
            )

            esn_pre_activation = (
                esn_input_part
                + esn_recurrent_part
                + self.esn_bias
            )

            if self.activation is not None:

                esn_candidate_state = (
                    self.activation(
                        esn_pre_activation
                    )
                )

            else:

                esn_candidate_state = (
                    esn_pre_activation
                )

            esn_output = (
                (1.0 - self.leaky)
                * esn_previous_state
                + self.leaky
                * esn_candidate_state
            )


        # =====================================================
        # EuSN update
        # =====================================================

        if self.num_eusn_units > 0:

            eusn_input_part = tf.matmul(
                inputs,
                self.eusn_kernel,
            )

            eusn_recurrent_part = tf.matmul(
                eusn_previous_state,
                self.eusn_recurrent_kernel,
            )

            eusn_pre_activation = (
                eusn_input_part
                + eusn_recurrent_part
                + self.eusn_bias
            )

            if self.activation is not None:

                eusn_delta = self.activation(
                    eusn_pre_activation
                )

            else:

                eusn_delta = (
                    eusn_pre_activation
                )

            eusn_output = (
                eusn_previous_state
                + self.epsilon * eusn_delta
            )


        # =====================================================
        # Merge states
        # =====================================================

        if self.num_esn_units == 0:

            output = eusn_output

        elif self.num_eusn_units == 0:

            output = esn_output

        else:

            output = tf.concat(
                [
                    esn_output,
                    eusn_output,
                ],
                axis=-1,
            )

        return output, [output]

class CoupledReservoirCell(keras.layers.Layer):
    """
    Coupled ESN-EuSN reservoir.

    The reservoir is divided into two equally-sized populations:
    one ESN and one EuSN.

    The populations interact through fixed orthogonal
    coupling matrices.
    """

    def __init__(
        self,
        units,
        beta_esn_to_eusn=0.0,
        beta_eusn_to_esn=0.0,
        input_scaling=1.0,
        bias_scaling=1.0,
        spectral_radius=0.99,
        leaky=1.0,
        recurrent_scaling=1.0,
        epsilon=0.01,
        gamma=0.001,
        activation=tf.nn.tanh,
        **kwargs
    ):
        super().__init__(**kwargs)

        if units % 2 != 0:
            raise ValueError(
                "CoupledReservoirCell requires an even number of units."
            )

        self.units = units
        self.state_size = units

        self.num_esn_units = units // 2
        self.num_eusn_units = units // 2

        self.beta_esn_to_eusn = beta_esn_to_eusn
        self.beta_eusn_to_esn = beta_eusn_to_esn

        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling

        self.spectral_radius = spectral_radius
        self.leaky = leaky

        self.recurrent_scaling = recurrent_scaling
        self.epsilon = epsilon
        self.gamma = gamma

        self.activation = activation


    def build(self, input_shape):

        input_dim = input_shape[-1]

        # =====================================================
        # ESN block
        # =====================================================

        value = (
            self.spectral_radius
            / np.sqrt(self.num_esn_units)
        ) * (6 / np.sqrt(12))

        self.esn_recurrent_kernel = tf.random.uniform(
            shape=(
                self.num_esn_units,
                self.num_esn_units,
            ),
            minval=-value,
            maxval=value,
        )

        self.esn_kernel = tf.random.uniform(
            shape=(
                input_dim,
                self.num_esn_units,
            ),
            minval=-self.input_scaling,
            maxval=self.input_scaling,
        )

        self.esn_bias = tf.random.uniform(
            shape=(self.num_esn_units,),
            minval=-self.bias_scaling,
            maxval=self.bias_scaling,
        )


        # =====================================================
        # EuSN block
        # =====================================================

        W = tf.random.uniform(
            shape=(
                self.num_eusn_units,
                self.num_eusn_units,
            ),
            minval=-self.recurrent_scaling,
            maxval=self.recurrent_scaling,
        )

        I = tf.linalg.eye(
            self.num_eusn_units
        )

        self.eusn_recurrent_kernel = (
            W
            - tf.transpose(W)
            - self.gamma * I
        )

        self.eusn_kernel = tf.random.uniform(
            shape=(
                input_dim,
                self.num_eusn_units,
            ),
            minval=-self.input_scaling,
            maxval=self.input_scaling,
        )

        self.eusn_bias = tf.random.uniform(
            shape=(self.num_eusn_units,),
            minval=-self.bias_scaling,
            maxval=self.bias_scaling,
        )


        # =====================================================
        # Coupling matrices
        # =====================================================

        raw_eusn_to_esn = tf.random.normal(
            shape=(
                self.num_eusn_units,
                self.num_esn_units,
            )
        )

        self.eusn_to_esn_kernel, _ = tf.linalg.qr(
            raw_eusn_to_esn
        )


        raw_esn_to_eusn = tf.random.normal(
            shape=(
                self.num_esn_units,
                self.num_eusn_units,
            )
        )

        self.esn_to_eusn_kernel, _ = tf.linalg.qr(
            raw_esn_to_eusn
        )

        self.built = True


    def call(self, inputs, states):

        previous_state = states[0]

        esn_previous_state = (
            previous_state[
                :,
                :self.num_esn_units
            ]
        )

        eusn_previous_state = (
            previous_state[
                :,
                self.num_esn_units:
            ]
        )


        # =====================================================
        # Cross-reservoir coupling
        # =====================================================

        eusn_to_esn_part = (
            self.beta_eusn_to_esn
            * tf.matmul(
                eusn_previous_state,
                self.eusn_to_esn_kernel,
            )
        )

        esn_to_eusn_part = (
            self.beta_esn_to_eusn
            * tf.matmul(
                esn_previous_state,
                self.esn_to_eusn_kernel,
            )
        )


        # =====================================================
        # ESN update
        # =====================================================

        esn_input_part = tf.matmul(
            inputs,
            self.esn_kernel,
        )

        esn_recurrent_part = tf.matmul(
            esn_previous_state,
            self.esn_recurrent_kernel,
        )

        esn_pre_activation = (
            esn_input_part
            + esn_recurrent_part
            + eusn_to_esn_part
            + self.esn_bias
        )

        if self.activation is not None:

            esn_candidate_state = (
                self.activation(
                    esn_pre_activation
                )
            )

        else:

            esn_candidate_state = (
                esn_pre_activation
            )

        esn_output = (
            (1.0 - self.leaky)
            * esn_previous_state
            + self.leaky
            * esn_candidate_state
        )


        # =====================================================
        # EuSN update
        # =====================================================

        eusn_input_part = tf.matmul(
            inputs,
            self.eusn_kernel,
        )

        eusn_recurrent_part = tf.matmul(
            eusn_previous_state,
            self.eusn_recurrent_kernel,
        )

        eusn_pre_activation = (
            eusn_input_part
            + eusn_recurrent_part
            + esn_to_eusn_part
            + self.eusn_bias
        )

        if self.activation is not None:

            eusn_delta = self.activation(
                eusn_pre_activation
            )

        else:

            eusn_delta = (
                eusn_pre_activation
            )

        eusn_output = (
            eusn_previous_state
            + self.epsilon * eusn_delta
        )


        # =====================================================
        # Merge states
        # =====================================================

        output = tf.concat(
            [
                esn_output,
                eusn_output,
            ],
            axis=-1,
        )

        return output, [output]

class MixedDynamicsReservoirCell(keras.layers.Layer):
    """
    Reservoir with a shared state updated by a mixture of
    ESN and EuSN dynamics.

    Both dynamics operate on the same previous state and produce
    two candidate states:

        z_esn
        z_eusn

    The new reservoir state is:

        h_t = mixing * z_esn + (1 - mixing) * z_eusn

    where mixing corresponds to lambda in the mathematical
    formulation.
    """

    def __init__(
        self,
        units,
        mixing=0.5,
        input_scaling=1.0,
        bias_scaling=1.0,
        spectral_radius=0.99,
        leaky=1.0,
        recurrent_scaling=1.0,
        epsilon=0.01,
        gamma=0.001,
        activation=tf.nn.tanh,
        **kwargs
    ):
        super().__init__(**kwargs)

        if not 0.0 <= mixing <= 1.0:
            raise ValueError(
                "mixing must be between 0 and 1."
            )

        self.units = units
        self.state_size = units

        self.mixing = mixing

        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling

        self.spectral_radius = spectral_radius
        self.leaky = leaky

        self.recurrent_scaling = recurrent_scaling
        self.epsilon = epsilon
        self.gamma = gamma

        self.activation = activation


    def build(self, input_shape):

        input_dim = input_shape[-1]


        # =====================================================
        # ESN dynamics
        # =====================================================

        value = (
            self.spectral_radius
            / np.sqrt(self.units)
        ) * (6 / np.sqrt(12))

        self.esn_recurrent_kernel = tf.random.uniform(
            shape=(
                self.units,
                self.units,
            ),
            minval=-value,
            maxval=value,
        )

        self.esn_kernel = tf.random.uniform(
            shape=(
                input_dim,
                self.units,
            ),
            minval=-self.input_scaling,
            maxval=self.input_scaling,
        )

        self.esn_bias = tf.random.uniform(
            shape=(self.units,),
            minval=-self.bias_scaling,
            maxval=self.bias_scaling,
        )


        # =====================================================
        # EuSN dynamics
        # =====================================================

        W = tf.random.uniform(
            shape=(
                self.units,
                self.units,
            ),
            minval=-self.recurrent_scaling,
            maxval=self.recurrent_scaling,
        )

        I = tf.linalg.eye(
            self.units
        )

        self.eusn_recurrent_kernel = (
            W
            - tf.transpose(W)
            - self.gamma * I
        )

        self.eusn_kernel = tf.random.uniform(
            shape=(
                input_dim,
                self.units,
            ),
            minval=-self.input_scaling,
            maxval=self.input_scaling,
        )

        self.eusn_bias = tf.random.uniform(
            shape=(self.units,),
            minval=-self.bias_scaling,
            maxval=self.bias_scaling,
        )

        self.built = True


    def call(self, inputs, states):

        previous_state = states[0]


        # =====================================================
        # ESN candidate state
        # =====================================================

        esn_input_part = tf.matmul(
            inputs,
            self.esn_kernel,
        )

        esn_recurrent_part = tf.matmul(
            previous_state,
            self.esn_recurrent_kernel,
        )

        esn_pre_activation = (
            esn_input_part
            + esn_recurrent_part
            + self.esn_bias
        )

        if self.activation is not None:

            esn_candidate = self.activation(
                esn_pre_activation
            )

        else:

            esn_candidate = (
                esn_pre_activation
            )

        esn_output = (
            (1.0 - self.leaky)
            * previous_state
            + self.leaky
            * esn_candidate
        )


        # =====================================================
        # EuSN candidate state
        # =====================================================

        eusn_input_part = tf.matmul(
            inputs,
            self.eusn_kernel,
        )

        eusn_recurrent_part = tf.matmul(
            previous_state,
            self.eusn_recurrent_kernel,
        )

        eusn_pre_activation = (
            eusn_input_part
            + eusn_recurrent_part
            + self.eusn_bias
        )

        if self.activation is not None:

            eusn_delta = self.activation(
                eusn_pre_activation
            )

        else:

            eusn_delta = (
                eusn_pre_activation
            )

        eusn_output = (
            previous_state
            + self.epsilon * eusn_delta
        )


        # =====================================================
        # Mix ESN and EuSN dynamics
        # =====================================================

        output = (
            self.mixing * esn_output
            + (1.0 - self.mixing) * eusn_output
        )

        return output, [output]


class ESN(keras.Model):
    """
    Echo State Network reservoir.

    The model returns the final reservoir state h_T.
    The readout is intentionally handled externally.
    """

    def __init__(
        self,
        units,
        input_scaling=1.0,
        bias_scaling=1.0,
        spectral_radius=0.9,
        leaky=1.0,
        activation=tf.nn.tanh,
        return_sequences=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.units = units

        self.reservoir = keras.layers.RNN(
            ReservoirCell(
                units=units,
                input_scaling=input_scaling,
                bias_scaling=bias_scaling,
                spectral_radius=spectral_radius,
                leaky=leaky,
                activation=activation,
            ),
            return_sequences=return_sequences,
        )


    def call(self, inputs):
        return self.reservoir(inputs)


class EuSN(keras.Model):
    """
    Euler State Network reservoir.

    The model returns the final reservoir state h_T.
    The readout is intentionally handled externally.
    """

    def __init__(
        self,
        units,
        input_scaling=1.0,
        bias_scaling=1.0,
        recurrent_scaling=1.0,
        epsilon=0.01,
        gamma=0.001,
        activation=tf.nn.tanh,
        return_sequences=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.units = units

        self.reservoir = keras.layers.RNN(
            EulerReservoirCell(
                units=units,
                input_scaling=input_scaling,
                bias_scaling=bias_scaling,
                recurrent_scaling=recurrent_scaling,
                epsilon=epsilon,
                gamma=gamma,
                activation=activation,
            ),
            return_sequences=return_sequences,
        )


    def call(self, inputs):
        return self.reservoir(inputs)

class HybridReservoir(keras.Model):
    """
    Heterogeneous ESN-EuSN reservoir.

    The readout is handled externally.
    """

    def __init__(
        self,
        units,
        esn_fraction,
        input_scaling=1.0,
        bias_scaling=1.0,
        spectral_radius=0.99,
        leaky=1.0,
        recurrent_scaling=1.0,
        epsilon=0.01,
        gamma=0.001,
        activation=tf.nn.tanh,
        return_sequences=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.units = units
        self.esn_fraction = esn_fraction

        self.reservoir = keras.layers.RNN(
            HybridReservoirCell(
                units=units,
                esn_fraction=esn_fraction,
                input_scaling=input_scaling,
                bias_scaling=bias_scaling,
                spectral_radius=spectral_radius,
                leaky=leaky,
                recurrent_scaling=recurrent_scaling,
                epsilon=epsilon,
                gamma=gamma,
                activation=activation,
            ),
            return_sequences=return_sequences,
        )


    def call(self, inputs):
        return self.reservoir(inputs)
    
class CoupledReservoir(keras.Model):
    """
    Coupled ESN-EuSN reservoir.

    The reservoir contains two equally-sized recurrent populations,
    one ESN and one EuSN, interacting through bidirectional coupling.

    The readout is handled externally.
    """

    def __init__(
        self,
        units,
        beta_esn_to_eusn=0.0,
        beta_eusn_to_esn=0.0,
        input_scaling=1.0,
        bias_scaling=1.0,
        spectral_radius=0.99,
        leaky=1.0,
        recurrent_scaling=1.0,
        epsilon=0.01,
        gamma=0.001,
        activation=tf.nn.tanh,
        return_sequences=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.units = units

        self.reservoir = keras.layers.RNN(
            CoupledReservoirCell(
                units=units,
                beta_esn_to_eusn=beta_esn_to_eusn,
                beta_eusn_to_esn=beta_eusn_to_esn,
                input_scaling=input_scaling,
                bias_scaling=bias_scaling,
                spectral_radius=spectral_radius,
                leaky=leaky,
                recurrent_scaling=recurrent_scaling,
                epsilon=epsilon,
                gamma=gamma,
                activation=activation,
            ),
            return_sequences=return_sequences,
        )

    def call(self, inputs):
        return self.reservoir(inputs)

class CoupledSymmetricReservoir(CoupledReservoir):
    """
    Coupled ESN-EuSN reservoir with symmetric coupling strength.

    The same scalar beta is used in both directions:

        ESN -> EuSN : beta
        EuSN -> ESN : beta

    The two coupling matrices remain independent and fixed.
    """

    def __init__(
        self,
        units,
        coupling_strength=0.1,
        input_scaling=1.0,
        bias_scaling=0.0,
        spectral_radius=0.9,
        leaky=1.0,
        recurrent_scaling=1.0,
        epsilon=0.01,
        gamma=0.001,
        return_sequences=False,
        **kwargs
    ):
        self.coupling_strength = coupling_strength

        super().__init__(
            units=units,
            beta_esn_to_eusn=coupling_strength,
            beta_eusn_to_esn=coupling_strength,
            input_scaling=input_scaling,
            bias_scaling=bias_scaling,
            spectral_radius=spectral_radius,
            leaky=leaky,
            recurrent_scaling=recurrent_scaling,
            epsilon=epsilon,
            gamma=gamma,
            return_sequences=return_sequences,
            **kwargs
        )

class MixedDynamicsReservoir(keras.Model):
    """
    Reservoir combining ESN and EuSN candidate dynamics
    over a shared recurrent state.

    The readout is handled externally.
    """

    def __init__(
        self,
        units,
        mixing=0.5,
        input_scaling=1.0,
        bias_scaling=1.0,
        spectral_radius=0.99,
        leaky=1.0,
        recurrent_scaling=1.0,
        epsilon=0.01,
        gamma=0.001,
        activation=tf.nn.tanh,
        return_sequences=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.units = units
        self.mixing = mixing

        self.reservoir = keras.layers.RNN(
            MixedDynamicsReservoirCell(
                units=units,
                mixing=mixing,
                input_scaling=input_scaling,
                bias_scaling=bias_scaling,
                spectral_radius=spectral_radius,
                leaky=leaky,
                recurrent_scaling=recurrent_scaling,
                epsilon=epsilon,
                gamma=gamma,
                activation=activation,
            ),
            return_sequences=return_sequences,
        )


    def call(self, inputs):
        return self.reservoir(inputs)