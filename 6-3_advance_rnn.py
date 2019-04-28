'''
リカレントニューラルネットワークの高度な使い方
'''

import os
import numpy as np
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.datasets import imdb
from keras.preprocessing import sequence

# 気象データセットのデータの調査
def search_data():
    data_dir = '/Users/masa/Downloads/jena_climate'
    fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

    f = open(fname)
    data = f.read()
    f.close()

    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]

    print(header)
    print(len(lines))

    # データの解析
    float_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        float_data[i, :] = values

    # 気温の時系列データのプロット
    temp = float_data[:, 1]
    plt.plot(range(len(temp)), temp)
    plt.show()

    # 最初の10日間の気温データをプロット
    plt.plot(range(1440), temp[:1440])
    plt.show()

# 時系列サンプルとそれらの目的値を生成するジェネレータ
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback

    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index,
                                     size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

# メイン
def main():
    data_dir = '/Users/masa/Downloads/jena_climate'
    fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

    f = open(fname)
    data = f.read()
    f.close()

    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]

    # データの解析
    float_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        float_data[i, :] = values

    mean = float_data[:200000].mean(axis=0)
    float_data -= mean
    std = float_data[:200000].std(axis=0)
    float_data /= std

    # 訓練、検証、テストに使用するジェネレータの準備
    lookback = 1440
    step = 6
    delay = 144
    batch_size = 128

    # 訓練ジェネレータ
    train_gen = generator(float_data,
                          lookback=lookback,
                          delay=delay,
                          min_index=0,
                          max_index=200000,
                          shuffle=True,
                          step=step,
                          batch_size=batch_size)

    # 検証ジェネレータ
    val_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=200001,
                        max_index=300000,
                        step=step,
                        batch_size=batch_size)

    # テストジェネレータ
    test_gen = generator(float_data,
                         lookback=lookback,
                         delay=delay,
                         min_index=300001,
                         max_index=None,
                         step=step,
                         batch_size=batch_size)

    # 検証データセット全体を調べるためにval_genから抽出する時間刻みの数
    val_steps = (300000 - 200001 - lookback) // batch_size

    # テストデータセット全体を調べるためにtest_genから抽出する時間刻みの数
    test_steps = (len(float_data) - 300001 - lookback) // batch_size

    evaluate_naive_method(val_steps, val_gen)

    stack_gru_model_with_dropout(train_gen, val_gen, test_gen, lookback, step, float_data,
                                 val_steps)

# 常識的なベースラインのMAEを計算
def evaluate_naive_method(val_steps, val_gen):
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))

# 全結合モデルの訓練と評価
def simple_dense_model(train_gen, val_gen, test_gen, lookback, step, float_data,
                       val_steps):

    model = Sequential()
    model.add(layers.Flatten(input_shape=(lookback // step,
                                          float_data.shape[-1])))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=20,
                                  validation_data=val_gen,
                                  validation_steps=val_steps)

    show_result(history)

# 結果をプロット
def show_result(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    # 損失値をプロット
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.savefig('./fig/6-3_training_and_validation_loss_stack_gru_dropout.png')

# GRUベースのモデルの訓練と評価
def gru_model(train_gen, val_gen, test_gen, lookback, step, float_data,
              val_steps):

    model = Sequential()
    model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
    model.add(layers.Dense(1))

    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=20,
                                  validation_data=val_gen,
                                  validation_steps=val_steps)

    show_result(history)

# ドロップアウトで正則化したGRUベースのモデルの訓練と評価
def gru_model_with_dropout(train_gen, val_gen, test_gen, lookback, step, float_data,
                           val_steps):

    model = Sequential()
    model.add(layers.GRU(32,
                         dropout=0.2,
                         recurrent_dropout=0.2,
                         input_shape=(None, float_data.shape[-1])))
    model.add(layers.Dense(1))

    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=40,
                                  validation_data=val_gen,
                                  validation_steps=val_steps)

    show_result(history)

# ドロップアウトで正則化されたスタッキングGRUモデルでの訓練と評価
def stack_gru_model_with_dropout(train_gen, val_gen, test_gen, lookback, step, float_data,
                                 val_steps):
    model = Sequential()
    model.add(layers.GRU(32,
                         dropout=0.1,
                         recurrent_dropout=0.5,
                         return_sequences=True,
                         input_shape=(None, float_data.shape[-1])))
    model.add(layers.GRU(64,
                         dropout=0.1,
                         recurrent_dropout=0.5))
    model.add(layers.Dense(1))

    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=40,
                                  validation_data=val_gen,
                                  validation_steps=val_steps)

    show_result(history)

def main_imdb():
    # 特徴量として考慮する単語の数
    max_features = 10000

    # max_features個の最も出現頻度の高い単語のうち、
    # この数の単語を残してテキストをカット
    max_len = 50

    # データを読み込む
    (x_train, y_train), (x_test, y_test) = \
        imdb.load_data(num_words=max_features)

    # シーケンスを逆向きにする
    x_train = [x[::-1] for x in x_train]
    y_test = [x[::-1] for x in x_test]

    #シーケンスをパディングする
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)

    model = build_bidi_lstm_model(max_features)

    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=128,
                        validation_split=0.2)

def build_lstm_model(max_features):
    model = Sequential()
    model.add(layers.Embedding(max_features, 128))
    model.add(layers.LSTM(32))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])

    return model

def build_bidi_lstm_model(max_features):
    model = Sequential()
    model.add(layers.Embedding(max_features, 128))
    model.add(layers.Bidirectional(layers.LSTM(32)))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])

    return model


if __name__ == '__main__':
    main_imdb()

