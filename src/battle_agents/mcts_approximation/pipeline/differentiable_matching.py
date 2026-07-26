import keras
import numpy as np

@keras.saving.register_keras_serializable()
class DifferentiableSpeciesMatching(keras.layers.Layer):
    """Differentiable matching between predictions and real species."""

    def __init__(
        self,
        real_stats_matrix,
        real_types_matrix,
        valid_species_mask,
        temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if temperature <= 0:
            raise ValueError(
                "The temperature must be greater than zero."
            )

        self.temperature = float(temperature)

        self._real_stats_values = np.asarray(
            real_stats_matrix,
            dtype=np.float32,
        )

        self._real_types_values = np.asarray(
            real_types_matrix,
            dtype=np.float32,
        )

        self._valid_species_values = np.asarray(
            valid_species_mask,
            dtype=np.float32,
        )

        self._validate_reference_data()

    def _validate_reference_data(self) -> None:
        if self._real_stats_values.ndim != 2:
            raise ValueError(
                "real_stats_matrix must have shape "
                "(num_species, num_stats)."
            )

        if self._real_types_values.ndim != 2:
            raise ValueError(
                "real_types_matrix must have shape "
                "(num_species, num_types)."
            )

        if self._valid_species_values.ndim != 1:
            raise ValueError(
                "valid_species_mask must have shape "
                "(num_species,)."
            )

        num_species = self._real_stats_values.shape[0]

        if self._real_types_values.shape[0] != num_species:
            raise ValueError("Stats and types contain different numbers of species.")

        if self._valid_species_values.shape[0] != num_species:
            raise ValueError("The mask does not contain the same number of species as the reference matrices.")

    def build(self, input_shape) -> None:
        # Catalog data. These values will not be modified during training.
        self.real_stats = self.add_weight(
            name="real_stats",
            shape=self._real_stats_values.shape,
            initializer=keras.initializers.Constant(
                self._real_stats_values
            ),
            trainable=False,
        )

        self.real_types = self.add_weight(
            name="real_types",
            shape=self._real_types_values.shape,
            initializer=keras.initializers.Constant(
                self._real_types_values
            ),
            trainable=False,
        )

        self.valid_species_mask = self.add_weight(
            name="valid_species_mask",
            shape=self._valid_species_values.shape,
            initializer=keras.initializers.Constant(
                self._valid_species_values
            ),
            trainable=False,
        )

        super().build(input_shape)

    def call(self, inputs):
        if len(inputs) != 6:
            raise ValueError("DifferentiableSpeciesMatching expects six input tensors.")
        (
            pred_species,
            pred_stats,
            pred_types,
            weight_species,
            weight_stats,
            weight_type,
        ) = inputs

        # 1. Direct species prediction distance

        d_species = 1.0 - pred_species

        # 2. Stats distance

        pred_stats_expanded = keras.ops.expand_dims(pred_stats, axis=1)
        real_stats_expanded = keras.ops.expand_dims(self.real_stats, axis=0)

        stats_difference = pred_stats_expanded - real_stats_expanded
        d_stats = keras.ops.sum(keras.ops.square(stats_difference), axis=-1)

        # 3. Types distance

        pred_types_expanded = keras.ops.expand_dims(pred_types, axis=1)
        real_types_expanded = keras.ops.expand_dims(self.real_types, axis=0)
        types_difference = pred_types_expanded - real_types_expanded
        d_types = keras.ops.sum(keras.ops.square(types_difference), axis=-1,)

        # 4. Weighted sum of the three distances

        total_distance = (
            weight_species * d_species
            + weight_stats * d_stats
            + weight_type * d_types
        )

        # 5. Prevent unknown/padding from winning

        valid_mask = keras.ops.expand_dims(self.valid_species_mask > 0.0, axis=0)

        large_distance = (
            keras.ops.ones_like(total_distance)
            * keras.ops.cast(
                1e9,
                total_distance.dtype,
            )
        )

        # Invalid species receive a very large distance
        masked_distance = keras.ops.where(
            valid_mask,
            total_distance,
            large_distance,
        )

        # 6. Convert distances into probabilities

        matching_probabilities = keras.ops.softmax(-masked_distance / self.temperature, axis=-1)

        return matching_probabilities

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()

        config.update({
            "real_stats_matrix":
                self._real_stats_values.tolist(),
            "real_types_matrix":
                self._real_types_values.tolist(),
            "valid_species_mask":
                self._valid_species_values.tolist(),
            "temperature":
                self.temperature,
        })

        return config