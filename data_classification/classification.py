from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import accuracy_score
from data_classification.noise_filter import *

# Constants for file paths
DATA_PATH_MAJA = "C:/Users/majag/Desktop/marble/NewData"
DATA_PATH_LUKAS = "C:/University/Marble/Data"

# Labels for the dataset
LABELS = ['incorrect_orthosis', 'correct_orthosis', 'no_orthosis']

# Choose the data path
DATA_PATH = DATA_PATH_MAJA


def build_FF_model():
    """
    Function to build a feedforward neural network
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=60, activation='leaky_relu'),
        tf.keras.layers.Dense(units=30, activation='relu'),
        tf.keras.layers.Dense(units=30, activation='relu'),
        tf.keras.layers.Dense(units=3, activation='softmax')
    ])

    # Compile the model
    model.compile(loss=tf.keras.losses.mae,  # mae is short for mean absolute error
                  optimizer=tf.keras.optimizers.SGD(),  # SGD is short for stochastic gradient descent
                  metrics=["mae"])

    return model


def build_RNN_model():
    """
    Function to build a recurrent neural network
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(3, 80)),
        tf.keras.layers.SimpleRNN(units=40),
        tf.keras.layers.Dense(units=len(LABELS), activation='softmax'),
    ])

    # Compile the model
    model.compile(loss=tf.keras.losses.mae, # mae is short for mean absolute error
                  optimizer=tf.keras.optimizers.SGD(), # SGD is short for stochastic gradient descent
                  metrics=["mae"],
                  )

    return model


def get_label(folder_name):
    """
    Function to get the label for a given folder
    :param folder_name: Folder with .xls data files from recording
    :return: string label for the given folder
    """

    label = len(LABELS) * [0]
    for j in range(len(LABELS)):
        if LABELS[j] in folder_name:
            label[j] = 1
            return label
    return None


def load_dataset(path, filter_noise, LIN=False, RNN=False):
    """
    Function to load the dataset
    :param path: path to the dataset
    :param filter_noise: True if we want to get rid of the data noise
    :param LIN: True if we want to consider linearized data
    :param RNN: True if we want to use RNN for classification, False if we will use FFN
    :return: [X, y] labeled dataset from given path
    """
    X, y = [], []

    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)

        label = get_label(folder_name)
        if label:
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

                    if RNN:
                        analysis = analysis.shrink_sample_time_domain(3)
                        x_point = analysis.p_sensors
                    else:
                        min_pressure, max_pressure = analysis.extrema_pressure_time_stamp()
                        x_point = max_pressure.flatten()

                    y.append(label)
                    X.append(x_point)
    return X, y


class TFModel:

    def __init__(self, dataset_path, filter_noise, rnn=False):

        self.X, self.y = load_dataset(dataset_path, filter_noise, LIN=False, RNN=rnn)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            random_state=42,
            test_size=0.3
        )
        if rnn:
            self.model = build_RNN_model()
        else:
            self.model = build_FF_model()
        self.history = None

    def train(self):
        # Train given models and return training history
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
        plt.plot(self.history.history['mae'], color='magenta')
        plt.plot(self.history.history['val_mae'], color='green')
        plt.title('model error')
        plt.ylabel('Mean absolute error [mae]')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    def show_accuracy(self):
        prediction = self.model.predict(tf.convert_to_tensor(self.X_test))

        for i in range(len(prediction)):
            if np.argmax(prediction[i]) != np.argmax(self.y_test[i]):
                print("Incorrect classification; pred: ", np.argmax(prediction[i]), " label: ",
                      np.argmax(self.y_test[i]))

        print("Accuracy score: ",
              accuracy_score(np.argmax(self.y_test, axis=1), np.argmax(prediction, axis=1)))


# Create prediction model
tf_model = TFModel(dataset_path=DATA_PATH, filter_noise=False, rnn=False)

# Train model and show plots
tf_model.train()
tf_model.show_error_plot()
tf_model.show_accuracy()
