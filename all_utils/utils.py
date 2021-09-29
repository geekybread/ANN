import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pandas as pd

def prepare_data(dataset="mnist"):
    data = tf.keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test) = data.load_data()
    X_valid, X_train = X_train_full[:5000]/255., X_train_full[5000:]/255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    X_test = X_test/255.

    return X_valid, X_train, y_valid, y_train, X_test 



def save_plot(history, file_name):

    df = pd.DataFrame(history.history)
    df.plot(figsize=(10,8))
    plt.grid(True)

    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True) #only create if model_dir doesn't exists
    plot_Path = os.path.join(plot_dir, file_name)

    plt.savefig(plot_Path)