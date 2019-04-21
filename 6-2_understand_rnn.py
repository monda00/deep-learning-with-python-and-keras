'''
リカレントニューラルネットワークを理解する
'''

import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt

# 単純なRNNのNumPy実装
def simple_rnn():
    timesteps = 100         # 入力シーケンスの時間刻みの数
    input_features = 32     # 入力特徴空間の次元の数
    output_features = 64    # 出力特徴空間の次元の数

    # 入力データ：ランダムにノイズを挿入
    inputs = np.random.random((timesteps, input_features))

    # 初期状態：全て0のベクトル
    state_t = np.zeros((output_features, ))

    # ランダムな重み行列を作成
    W = np.random.random((output_features, input_features))
    U = np.random.random((output_features, output_features))
    b = np.random.random((output_features, ))

    successive_outputs = []

    # input_tは形状が(input_features, )のベクトル
    for input_t in inputs:
        # 入力と現在の状態（１つ前の出力）を結合して現在の出力を取得
        output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
        # この出力をリストに格納
        successive_outputs.append(output_t)
        # 次の時間刻みのためにRNNの状態を更新
        state_t = output_t

    # 最終的な出力は形状が(timesteps, output_features)の２次元テンソル
    final_output_sequence = np.stack(successive_outputs, axis=0)
    print(final_output_sequence)

# 結果をプロット
def show_result(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    # 正解率をプロット
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and Validation accuracy')
    plt.legend()
    plt.savefig('./fig/6-2_training_and_validation_accuracy_lstm.png')

    plt.figure()

    # 損失値をプロット
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.savefig('./fig/6-2_training_and_validation_loss_lstm.png')

# Embedding層とSimpleRNN層を使ったモデル
def build_embedding_and_simplernn_model(max_features):
    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(SimpleRNN(32))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])

    return model

# LSTMを使ったモデル
def build_lstm_model(max_features):
    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])

    return model

# IMDbデータに対するRNN
def rnn_for_imdb():
    '''
    IMDbデータの前処理
    '''
    max_features = 10000    # 特徴量として考慮する単語の数
    max_len = 500           # この数の単語を残してテキストをカット
    batch_size = 32

    print('Loading data...')
    (input_train, y_train), (input_test, y_test) = \
        imdb.load_data(num_words=max_features)
    print(len(input_train), 'train sequence')
    print(len(input_test), 'test sequence')

    print('Pad sequences (samples x time)')
    input_train = sequence.pad_sequences(input_train, maxlen=max_len)
    input_test = sequence.pad_sequences(input_test, maxlen=max_len)
    print('input_train shape:', input_train.shape)
    print('input_test shape:', input_test.shape)

    model = build_embedding_and_simplernn_model(max_features)
    model.summary()

    history = model.fit(input_train, y_train,
                        epochs=10, batch_size=128, validation_split=0.2)

    show_result(history)


if __name__ == '__main__':
    rnn_for_imdb()

