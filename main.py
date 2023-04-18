import requests
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
from keras.callbacks import EarlyStopping


def api_connect():
    key = "https://api.binance.com/api/v3/trades?symbol=BTCUSDT"
    data = requests.get(key)
    data = data.json()
    df = pd.DataFrame.from_dict(data)
    # df['time'] = pd.to_datetime(df['time'])
    df.to_csv('tmp.csv')
    print(df)


def normalize_windows(window_data, single_window=False):
    normalized_data = []
    window_data = [window_data] if single_window else window_data
    for window in window_data:
        normalize_window = []
        for col_i in range(window.shape[1]):
            normalized_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
            normalize_window.append(normalized_col)
        normalize_window = np.array(normalize_window).T
        normalized_data.append(normalize_window)
    return np.array(normalized_data)


def _next_window(data_train, i, seq_len, normalize):
    window = data_train[i:i + seq_len]
    window = normalize_windows(window, single_window=True)[0] if normalize else window
    x = window[:-1]
    y = window[-1, [0]]
    return x, y


def get_train_data(data_train, len_train, seq_len, normalize):
    data_x = []
    data_y = []
    for i in range(len_train - seq_len + 1):
        x, y = _next_window(data_train, i, seq_len, normalize)
        data_x.append(x)
        data_y.append(y)
    return np.array(data_x), np.array(data_y)


def get_test_data(data_test, len_test, seq_len, normalise):
    data_windows = []
    for i in range(len_test - seq_len + 1):
        data_windows.append(data_test[i:i + seq_len])
    data_windows = np.array(data_windows).astype(float)
    data_windows = normalize_windows(data_windows, single_window=False) if normalise else data_windows
    x = data_windows[:, :-1]
    y = data_windows[:, -1, [0]]
    return x, y


def get_last_data(data_test, seq_len, normalise):
    last_data = data_test[seq_len:]
    data_windows = np.array(last_data).astype(float)
    # data_windows = np.array([data_windows])
    # data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows
    data_windows = normalize_windows(data_windows, single_window=True) if normalise else data_windows
    return data_windows


def de_normalize_predict(price_1st, _data):
    return (_data + 1) * price_1st


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


def main():
    api_connect()
    df = pd.read_csv('tmp.csv', index_col=0, decimal='.', delimiter=',')
    print(df)
    # разбиение данных
    split = 0.85
    i_split = int(len(df) * split)
    cols = ['price', 'time']
    data_train = df.get(cols).values[:i_split]
    data_test = df.get(cols).values[i_split:]
    len_train = len(data_train)
    len_test = len(data_test)
    print(len(df), len_train, len_test)
    print(data_train.shape, data_test.shape)

    sequence_len = 50
    input_dim = 2
    batch_size = 32
    epochs = 2

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(100, input_shape=(sequence_len - 1, input_dim), return_sequences=True),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.LSTM(100, return_sequences=True),
        tf.keras.layers.LSTM(100, return_sequences=False),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.summary()
    # tf.keras.utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    x, y = get_train_data(data_train, len_train, sequence_len, normalize=True)
    print(x, y, x.shape, y.shape)

    steps_per_epoch = math.ceil((len_train - sequence_len) / batch_size)
    print(steps_per_epoch)
    callbacks = [EarlyStopping(monitor='mae', patience=2)]
    model.fit(x, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    x_test, y_test = get_test_data(data_test, len_test, seq_len=sequence_len, normalise=True)
    print('Test data shapes:', x_test.shape, y_test.shape)
    print(y_test)

    model.evaluate(x_test, y_test, verbose=2)

    last_data_2_predict_prices = get_last_data(data_test, -(sequence_len - 1), False) # ненормализованные
    last_data_2_predict_prices_1st_price = last_data_2_predict_prices[0][0]
    last_data_2_predict = get_last_data(data_test, -(sequence_len - 1), True) # нормализованные
    print('***', -(sequence_len - 1), last_data_2_predict.size, '***')

    print(last_data_2_predict_prices_1st_price)

    predictions2 = model.predict(last_data_2_predict)
    print(predictions2, predictions2[0][0])

    predicted_price = de_normalize_predict(last_data_2_predict_prices_1st_price, predictions2)
    print(predicted_price)

    print('[Model] Predicting Point-by-Point...')
    predicted = model.predict(x_test)
    predicted = np.reshape(predicted, (predicted.size,))

    plot_results(predicted, y_test)


if __name__ == '__main__':
    main()
