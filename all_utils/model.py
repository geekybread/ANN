import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

def model():
    LAYERS = [
            tf.keras.layers.Flatten(input_shape=[28,28], name="input_layer"),
            tf.keras.layers.Dense(300, activation="relu", name='hidden_layer_1'),
            tf.keras.layers.Dense(100, activation="relu", name='hidden_layer_2'),
            tf.keras.layers.Dense(10, activation="softmax", name='output_layer')

    ]

    model_clf = tf.keras.models.Sequential(LAYERS)

    model_clf.summary()

    weights, biases = model_clf.layers[1].get_weights()

    LOSS_FUNCTION = "sparse_categorical_crossentropy"
    OPTIMIZER = "SGD"
    METRICS = ["accuracy"]


    model_clf.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)

    return model_clf, weights, biases
