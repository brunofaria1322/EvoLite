import torch
import tensorflow as tf
from src.evolite.utils.validation import validate_model, validate_dataset


def test_validate_model():
    pytorch_model = torch.nn.Linear(10, 2)
    validate_model(pytorch_model)  # Should not raise an error

    tf_model = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(10,)), tf.keras.layers.Dense(2)]
    )

    validate_model(tf_model)  # Should not raise an error


def test_validate_dataset():
    class ValidDataset:
        def __getitem__(self, index):
            return index

        def __len__(self):
            return 10

    class InvalidDataset:
        pass

    validate_dataset(ValidDataset())  # Should not raise an error

    dataset = [(torch.rand(10), torch.rand(2)) for _ in range(10)]
    validate_dataset(dataset)  # Should not raise an error

    try:
        validate_dataset(InvalidDataset())
        assert False, "Expected ValueError"
    except ValueError:
        pass
