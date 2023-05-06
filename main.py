import requests
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras.callbacks import EarlyStopping


class DataLoader:
    def __init__(self, split, cols):
        self.df = None
        self.split = split
        self.cols = cols
        self.data_train = None
        self.data_test = None
        self.len_train = None
        self.len_test = None

    def split_data(self):
        i_split = int(len(self.df) * self.split)
        self.data_train = self.df.get(self.cols).values[:i_split]
        self.data_test = self.df.get(self.cols).values[i_split:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)
        print("len(dataframe) == ", len(self.df))
        print("len(self.data_train) == ", len(self.data_train))
        print("len(self.data_test) == ", len(self.data_test))

    def load_data_from_csv(self, filename):
        self.df = pd.read_csv(filename, index_col=0, decimal='.', delimiter=',')
        self.split_data()

    def de_normalize_predict(self, price_1st, _data):
        return (_data + 1) * price_1st

    def get_last_data(self, seq_len, normalise):
        last_data = self.data_test[seq_len:]
        data_windows = np.array(last_data).astype(float)
        data_windows = self.normalize_windows(data_windows, single_window=True) if normalise else data_windows
        return data_windows

    def get_test_data(self, seq_len, normalise):
        data_windows = []
        for i in range(self.len_test - seq_len + 1):
            data_windows.append(self.data_test[i:i + seq_len])
        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalize_windows(data_windows, single_window=False) if normalise else data_windows
        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x, y

    def get_train_data(self, seq_len, normalize):
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len + 1):
            x, y = self._next_window(i, seq_len, normalize)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def _next_window(self, i, seq_len, normalize):
        window = self.data_train[i:i + seq_len]
        window = self.normalize_windows(window, single_window=True)[0] if normalize else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    def normalize_windows(self, window_data, single_window=False):
        normalized_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalize_window = []
            for col_i in range(window.shape[1]):
                max_in_col = max(window[:, col_i])
                normalized_col = [(float(p) / float(max_in_col)) for p in window[:, col_i]]
                normalize_window.append(normalized_col)
            normalize_window = np.array(normalize_window).T
            normalized_data.append(normalize_window)
        return np.array(normalized_data)


class Model:
    def __init__(self):
        self.model = tf.keras.Sequential()

    def build_model(self):
        sequence_len = 50
        input_dim = 2

        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(100, input_shape=(sequence_len - 1, input_dim), return_sequences=True),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.LSTM(100, return_sequences=True),
            tf.keras.layers.LSTM(100, return_sequences=False),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.Dense(1, activation='linear')
        ])

        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model.summary()
        print('[Model] Model Compiled')

    def train(self, x, y, epochs, batch_size):
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        callbacks = [EarlyStopping(monitor='mae', patience=2)]
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        print('[Model] Training Completed.')

    def eval_test(self, x_test, y_test, verbose):
        return self.model.evaluate(x_test, y_test, verbose=verbose)

    def predict_point_by_point(self, data):
        print('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_next_hour(self, len1, len2):
        print('[Model] Predicting Next Hour...')
        predicted = self.model.predict(len1, len2)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted


def api_connect():
    key = "https://api.binance.com/api/v3/trades?symbol=BTCUSDT"
    data = requests.get(key)
    data = data.json()
    df = pd.DataFrame.from_dict(data)
    # df['time'] = pd.to_datetime(df['time'])
    df.to_csv('data/tmp.csv')
    print(df)


def plot_results(predicted_data, true_data, y):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(1, 1, 1)

    x = []
    for i in range(len(predicted_data)):
        x.append(len(y) - 1 + i)

    ax.plot(x, true_data, label='True Data')
    plt.plot(x, predicted_data, label='Prediction')
    plt.plot(y, label='Train data and test data')
    plt.legend()
    plt.grid()
    plt.show()


def main():
    api_connect()

    data = DataLoader(split=0.85, cols=['Close', 'Volume'])
    # data = DataLoader(split=0.85, cols=['price', 'qty'])
    data.load_data_from_csv('data/SBER_000101_211220.csv')

    model = Model()
    model.build_model()

    sequence_len = 50
    # epochs = 10
    epochs = 2
    batch_size = 32
    x, y = data.get_train_data(sequence_len, normalize=True)
    model.train(x, y, epochs=epochs, batch_size=batch_size)
    x_test, y_test = data.get_test_data(seq_len=sequence_len, normalise=True)
    print('Test data shapes:', x_test.shape, y_test.shape)

    model.eval_test(x_test, y_test, verbose=2)
    predictions = model.predict_point_by_point(x_test)
    # next_hour = model.predict_next_hour(len(predictions), len(predictions))
    plot_results(predictions, y_test, y)


if __name__ == '__main__':
    main()
