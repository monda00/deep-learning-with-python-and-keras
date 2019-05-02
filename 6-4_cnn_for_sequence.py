'''
畳み込みニューラルネットワークでのシーケンス処理
'''

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
import os
import numpy as np

def build_model_cnn(max_features, max_len):
    model = Sequential()
    model.add(layers.Embedding(max_features, 128, input_length=max_len))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(1))

    model.summary()

    return model

def main():
    max_features = 10000    # 特徴量として考慮する単語の数
    max_len = 500           # この数の単語を残してテキストをカット

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) =\
        imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    model = build_model_cnn(max_features, max_len)

    model.compile(optimizer=RMSprop(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=128,
                        validation_split=0.2)

def build_model_cnn_for_climate(float_data):
    model = Sequential()
    model.add(layers.Conv1D(32, 5, activation='relu',
                            input_shape=(None, float_data.shape[-1])))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(32, 5, activation='relu'))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(32, 5, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(1))

    return model

def build_model_cnn_gru(float_data):
    model = Sequential()
    model.add(layers.Conv1D(32, 5, activation='relu',
                            input_shape=(None, float_data.shape[-1])))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(32, 5, activation='relu'))
    model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))
    model.add(layers.Dense(1))

    model.summary()

    return model

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

def main_climate():
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
    lookback = 720
    step = 3
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

    model = build_model_cnn_gru(float_data)

    model.compile(optimizer=RMSprop(), loss='mae')

    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=20,
                                  validation_data=val_gen,
                                  validation_steps=val_steps)


if __name__ == '__main__':
    main_climate()

