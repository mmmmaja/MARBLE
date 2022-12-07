import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from data_analysis import SampleDataAnalysis
from sklearn.metrics import accuracy_score
from noise_filter import *


DATA_PATH_MAJA = "C:/Users/majag/Desktop/marble/NewData"
DATA_PATH_LUKAS = "C:/University/Marble/Data"

DATA_PATH = DATA_PATH_MAJA

# LABELS = ['incorrect_orthosis', 'correct_orthosis', 'no_orthosis']
LABELS = [
    'incorrect_orthosis_down',
    'incorrect_orthosis_up',
    'incorrect_orthosis_rotated_+',
    'incorrect_orthosis_rotated_-'
]


def get_dataset(LIN=False, filter=True):
    X, y = [], []

    for folder_name in os.listdir(DATA_PATH):
        folder_path = os.path.join(DATA_PATH, folder_name)

        label = get_label(folder_name)
        if label:
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                if (LIN and file_name[-8:-4] == 'LIN.') or (not LIN and file_name[-8:-4] == 'RAW.'):
                    analysis = SampleDataAnalysis(
                        folder_name,
                        file_path=file_path,
                    )
                    # filtering
                    if filter:
                        filtered_sensors = normal_filter_sample_sensors_global(analysis)
                        analysis = replace_filtered_sensors_normal_global(analysis, filtered_sensors)

                    min_pressure, max_pressure = analysis.extrema_pressure_time_stamp()
                    max_pressure = max_pressure.flatten()

                    y.append(label)
                    X.append(max_pressure)
    return X, y


def get_label(folder_name):
    label = len(LABELS) * [0]
    for j in range(len(LABELS)):
        if LABELS[j] in folder_name:
            label[j] = 1
            return label
    return None


def show_loss_plot(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def show_error_plot(history):
    print(history.history.keys())
    plt.plot(history.history['mae'], color='magenta')
    plt.plot(history.history['val_mae'], color='green')
    plt.title('model error')
    plt.ylabel('Mean absolute error [mae]')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def get_data():

    # Getting the dataset
    X, y = get_dataset(filter=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
    X_train, X_test = tf.convert_to_tensor(X_train), tf.convert_to_tensor(X_test)
    y_train, y_test = tf.convert_to_tensor(y_train), tf.convert_to_tensor(y_test)
    return X_train, X_test, y_train, y_test


def build_model():
    # Build NN
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=128, activation='leaky_relu'),  # 256
        tf.keras.layers.Dense(units=64, activation='relu'),  # 192
        tf.keras.layers.Dense(units=32, activation='relu'),  # 128
        tf.keras.layers.Dense(units=len(LABELS), activation='softmax')
    ])

    # Compile the model
    model.compile(loss=tf.keras.losses.mae,  # mae is short for mean absolute error
                  optimizer=tf.keras.optimizers.SGD(lr=0.01),  # SGD is short for stochastic gradient descent
                  metrics=["mae"])
    return model


class TFModel:

    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = get_data()
        self.model = build_model()

    def train(self):
        print(len(self.X_train))
        history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=100,
            batch_size=min(30, len(self.X_train)),
            verbose=1,
            validation_split=0.2
        )
        return history


def get_index(tf_array):
    for j in range(len(tf_array)):
        if tf_array[j] == 1:
            return j
    return None


tf_model = TFModel()
_history = tf_model.train()

show_error_plot(_history)

print("\n\n")
prediction = tf_model.model.predict(tf_model.X_test)
y_test_numpy = tf_model.y_test.numpy()
for i in range(len(prediction)):

    print("pred: ", np.argmax(prediction[i]), " label: ", np.argmax(y_test_numpy[i]))


print("Accuracy score: ", accuracy_score(np.argmax(tf_model.y_test.numpy(), axis = 1),np.argmax(prediction,axis= 1)))

