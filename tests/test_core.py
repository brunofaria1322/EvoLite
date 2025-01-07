import torch
import tensorflow as tf
from src.evolite import EvoLite
import os


def test_evolite_save_load_pytorch(tmp_path):
    dataset = [(torch.rand(10), torch.rand(2)) for _ in range(10)]

    pytorch_model = torch.nn.Linear(10, 2)
    evo_pytorch = EvoLite(pytorch_model, dataset)
    path = os.path.join(tmp_path, "model.pth")
    evo_pytorch.save(path)
    new_model = torch.nn.Linear(10, 2)
    evo_pytorch.model = new_model
    evo_pytorch.load(path)
    assert all(
        torch.equal(p1, p2)
        for p1, p2 in zip(pytorch_model.parameters(), evo_pytorch.model.parameters())
    )


def test_evolite_save_load_tensorflow(tmp_path):
    dataset = [(torch.rand(10), torch.rand(2)) for _ in range(10)]

    tf_model = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(10,)), tf.keras.layers.Dense(2)]
    )
    evo_tf = EvoLite(tf_model, dataset)
    path = os.path.join(tmp_path, "model.weights.h5")
    evo_tf.save(path)
    new_model = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(10,)), tf.keras.layers.Dense(2)]
    )
    evo_tf.model = new_model
    evo_tf.load(path)
    assert all(
        tf.reduce_all(tf.equal(w1, w2))
        for w1, w2 in zip(tf_model.weights, evo_tf.model.weights)
    )
