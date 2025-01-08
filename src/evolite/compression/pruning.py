from typing import Literal
import torch
import tensorflow as tf


class Pruning:
    def __init__(self, target: Literal["weights", "neurons"], rate: float) -> None:
        """
        Initialize the pruning module.

        Args:
            target: What to prune ("weights" or "neurons").
            rate: Fraction of weights/neurons to prune (e.g., 0.5 for 50%).
        """
        self.target = target
        self.rate = rate

    def apply(
        self, model: torch.nn.Module | tf.keras.Model
    ) -> torch.nn.Module | tf.keras.Model:
        """
        Apply pruning to the model.

        Args:
            model: The model to prune.

        Returns:
            The pruned model.
        """
        if isinstance(model, torch.nn.Module):
            if self.target == "neurons":
                return self._prune_pytorch_neurons(model)
            elif self.target == "weights":
                return self._prune_pytorch_weights(model)
            else:
                raise ValueError("Unsupported target.")
        elif isinstance(model, tf.keras.Model):
            if self.target == "neurons":
                return self._prune_tensorflow_neurons(model)
            elif self.target == "weights":
                return self._prune_tensorflow_weights(model)
            else:
                raise ValueError("Unsupported target.")
        else:
            raise ValueError("Unsupported model type.")

    def _prune_pytorch_weights(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Prune a PyTorch model.

        Args:
            model: The PyTorch model to prune.

        Returns:
            The pruned PyTorch model.
        """

        # Global pruning using global threshold
        # Collect all weights
        all_weights = []
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                all_weights.append(module.weight.data.abs().flatten())

        # Calculate global threshold
        all_weights = torch.cat(all_weights)
        threshold = torch.quantile(all_weights, self.rate)

        # Apply pruning using global threshold
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                mask = module.weight.data.abs() > threshold
                module.weight.data *= mask.float()

        """
        # Per layer pruning using local threshold
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                weight = module.weight.data
                threshold = torch.quantile(torch.abs(weight), self.rate)
                mask = torch.abs(weight) > threshold
                module.weight.data *= mask.float()
        """
        return model

    def _prune_pytorch_neurons(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Prune neurons in a PyTorch model.

        Args:
            model: The PyTorch model to prune.

        Returns:
            The pruned PyTorch model.
        """
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                weight = module.weight.data
                bias = module.bias.data if module.bias is not None else None

                # Prune neurons by removing columns from the weight matrix
                neuron_importance = torch.norm(weight, dim=0)
                threshold = torch.quantile(neuron_importance, self.rate)
                mask = neuron_importance > threshold

                # Apply mask to weights and bias
                module.weight.data = weight[:, mask]
                if bias is not None:
                    module.bias.data = bias[mask]
        return model

    def _prune_tensorflow_weights(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        Prune a TensorFlow model.

        Args:
            model: The TensorFlow model to prune.

        Returns:
            The pruned TensorFlow model.
        """

        # Global pruning using global threshold
        # Collect all weights
        all_weights = []
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                all_weights.append(tf.abs(layer.get_weights()[0]))

        # Calculate global threshold using concatenated weights
        all_weights_concat = tf.concat(
            [tf.reshape(w, [-1]) for w in all_weights], axis=0
        )
        threshold = tf.keras.ops.quantile(all_weights_concat, self.rate)

        # Apply the global threshold to each layer
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                weights = layer.get_weights()
                weight = weights[0]
                mask = tf.abs(weight) > threshold
                weights[0] *= mask.numpy().astype(float)
                layer.set_weights(weights)

        """
        # Per layer pruning using local threshold
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                weights = layer.get_weights()
                weight = weights[0]
                threshold = tfp.stats.percentile(tf.abs(weight), self.rate * 100)
                mask = tf.abs(weight) > threshold
                weights[0] *= mask.numpy().astype(float)
                layer.set_weights(weights)
        """

        return model

    def _prune_tensorflow_neurons(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        Prune neurons in a TensorFlow model by recreating it.

        Args:
            model: The TensorFlow model to prune.

        Returns:
            The pruned TensorFlow model.
        """
        new_layers = []
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                weights = layer.get_weights()
                weight = weights[0]
                bias = weights[1] if len(weights) > 1 else None

                # Prune neurons by removing columns from the weight matrix
                neuron_importance = np.linalg.norm(weight, axis=0)
                threshold = np.percentile(neuron_importance, self.rate * 100)
                mask = neuron_importance > threshold

                # Apply mask to weights and bias
                pruned_weight = weight[:, mask]
                if bias is not None:
                    pruned_bias = bias[mask]

                # Create a new layer with pruned neurons
                new_layer = tf.keras.layers.Dense(
                    units=np.sum(mask),
                    activation=layer.activation,
                    use_bias=layer.use_bias,
                )
                new_layer.build(layer.input_shape)
                if bias is not None:
                    new_layer.set_weights([pruned_weight, pruned_bias])
                else:
                    new_layer.set_weights([pruned_weight])
                new_layers.append(new_layer)
            else:
                new_layers.append(layer)

        # Recreate the model
        new_model = tf.keras.Sequential(new_layers)
        return new_model
