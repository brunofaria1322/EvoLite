import torch
import tensorflow as tf
import os
from src.evolite.utils.model_utils import save_model, load_model


def test_save_load_pytorch_model(tmp_path):
    model = torch.nn.Linear(10, 2)
    path = os.path.join(tmp_path, "model.pth")
    save_model(model, path)
    new_model = torch.nn.Linear(10, 2)
    load_model(new_model, path)
    assert all(
        torch.equal(p1, p2)
        for p1, p2 in zip(model.parameters(), new_model.parameters())
    )


def test_save_load_tensorflow_model(tmp_path):
    model = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(10,)), tf.keras.layers.Dense(2)]
    )
    path = os.path.join(tmp_path, "model.weights.h5")
    save_model(model, path)
    new_model = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(10,)), tf.keras.layers.Dense(2)]
    )
    load_model(new_model, path)
    assert all(
        tf.reduce_all(tf.equal(w1, w2))
        for w1, w2 in zip(model.weights, new_model.weights)
    )
