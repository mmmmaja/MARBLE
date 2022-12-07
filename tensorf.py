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

DATA_PATH = DATA_PATH_LUKAS

LABELS = ['incorrect_orthosis', 'correct_orthosis', 'no_orthosis']

# TODO try with the filtered data- sometimes the filtering causes the network to perform way better, but sometimes way worse

# TODO for some reason 1_LIN incorrect orthosis down 1 cm has everywhere 0 values
# check different data, what labels?
# Check recurrent nn

# Lukas pick time steps, size down input for each sample


def build_FF_model():
    # Build NN
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=60, activation='leaky_relu'),
        tf.keras.layers.Dense(units=30, activation='relu'),
        tf.keras.layers.Dense(units=30, activation='relu'),
        tf.keras.layers.Dense(units=3, activation='softmax')
    ])

    # model.add(tf.keras.layers.Dense(32, activation='relu'))

    # Compile the model
    model.compile(loss=tf.keras.losses.mae,  # mae is short for mean absolute error
                  optimizer=tf.keras.optimizers.SGD(),  # SGD is short for stochastic gradient descent
                  metrics=["mae"])

    print(model.summary())
    return model

def build_RNN_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(3, 80)),
        tf.keras.layers.SimpleRNN(units=40),
        tf.keras.layers.Dense(units=3, activation='softmax'),
    ])

    # model.add(tf.keras.layers.Dense(32, activation='relu'))

    # Compile the model
    model.compile(loss=tf.keras.losses.mae,  # mae is short for mean absolute error
                  optimizer=tf.keras.optimizers.SGD(),  # SGD is short for stochastic gradient descent
                  metrics=["mae"],
                  )

    print(model.summary())
    return model


class TFModel:

    def __init__(self, model, dataset_path, filter_noise, rnn = False):

        self.X, self.y = self.load_dataset(dataset_path,filter_noise,LIN = False,RNN = rnn)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=42, test_size=0.3)
        self.model = model
        self.history = None

    def load_dataset(self,path,filter_noise,LIN=False,RNN = False):
        X, y = [], []

        for folder_name in os.listdir(path):
            folder_path = os.path.join(path, folder_name)

            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                if (LIN and file_name[-8:-4] == 'LIN.') or (not LIN and file_name[-8:-4] == 'RAW.'):
                    analysis = SampleDataAnalysis(
                        folder_name,
                        file_path=file_path,
                    )

                    if filter_noise:
                        filtered_sensors = normal_filter_sample_sensors_global(analysis)
                        analysis = replace_filtered_sensors_normal_global(analysis, filtered_sensors)


                    x_point = None
                    if RNN:
                        analysis = analysis.shrink_sample_time_domain(3)
                        x_point = analysis.p_sensors
                    else:
                        min_pressure, max_pressure = analysis.extrema_pressure_time_stamp()
                        x_point = max_pressure.flatten()


                    y.append(self.get_label(folder_name))
                    X.append(x_point)
        return X, y

    def get_label(self,folder_name):
        label = len(LABELS) * [0]
        for j in range(len(LABELS)):
            if LABELS[j] in folder_name:
                label[j] = 1
                return label
        return None

    def train(self):

        X_train, y_train = tf.convert_to_tensor(self.X_train), tf.convert_to_tensor(self.y_train)

        history = self.model.fit(
            X_train,
            y_train,
            epochs=200,
            batch_size=30,
            verbose=1,
            validation_split=0.2
        )
        self.history = history
        return history

    def show_loss_plot(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    def show_error_plot(self):
        print(self.history.history.keys())
        plt.plot(self.history.history['mae'], color='magenta')
        plt.plot(self.history.history['val_mae'], color='green')
        plt.title('model error')
        plt.ylabel('Mean absolute error [mae]')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    def show_accuracy(self,):
        prediction = self.model.predict(tf.convert_to_tensor(self.X_test))

        for i in range(len(prediction)):
            if np.argmax(prediction[i]) != np.argmax(self.y_test[i]):
                print("Incorrect classification; pred: ", np.argmax(prediction[i]), " label: ", np.argmax(self.y_test[i]))

        print("Accuracy score: ",
              accuracy_score(np.argmax(self.y_test, axis=1), np.argmax(prediction, axis=1)))



tf_model = TFModel(model=build_RNN_model(),dataset_path=DATA_PATH,filter_noise=False,rnn=True)

tf_model.train()
tf_model.show_error_plot()
tf_model.show_accuracy()

