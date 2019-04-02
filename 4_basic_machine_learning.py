'''
機械学習の基礎
'''

from keras.datasets import imdb
from keras import models
from keras import layers

import numpy as np
import matplotlib.pyplot as plt

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

def show_loss_history(history):
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("fig/bigger_model.png")

def build_original_model():
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

def build_bigger_model():
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

def build_smaller_model():
    model = models.Sequential()
    model.add(layers.Dense(4, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    # データの読み込み
    (train_data, train_labels), (test_data, test_labels) = \
        imdb.load_data(num_words=10000)

    # 訓練データのベクトル化
    x_train = vectorize_sequences(train_data)
    y_train = np.asarray(train_labels).astype('float32')
    # テストデータのベクトル化
    x_test = vectorize_sequences(test_data)
    y_test = np.asarray(test_labels).astype('float32')

    # 検証データの設定
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]

    model = build_bigger_model()

    history = model.fit(partial_x_train, partial_y_train,
                        epochs=20, batch_size=512,
                        validation_data=(x_val, y_val))

    show_loss_history(history)
    results = model.evaluate(x_test, y_test)
    print(results)
    print(model.predict(x_test))


