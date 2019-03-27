'''
多クラス分類の例：ニュース配信の分類
'''

from keras.datasets import reuters
from keras import models
from keras import layers

import numpy as np
import matplotlib.pyplot as plt

def print_data(data, labels):
    print(len(data))
    print(data[8])

def to_words(train_data):
    word_index = reuters.get_word_index()
    reverse_word_index = \
        dict([(value, key) for (key, value) in word_index.items()])

    decoded_newswire = ' '.join(
        [reverse_word_index.get(i - 3, '?') for i in train_data[0]])
    print(decoded_newswire)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

def show_loss_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Trainig and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def show_acc_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Trainig and validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def define_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    # データの読み込み
    (train_data, train_labels), (test_data, test_labels) = \
        reuters.load_data(num_words=10000)

    # データのベクトル化
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    # ラベルのベクトル化
    one_hot_train_labels = to_one_hot(train_labels)
    one_hot_test_labels = to_one_hot(test_labels)

    # 検証データの設定
    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]
    y_val = one_hot_train_labels[:1000]
    partial_y_train = one_hot_train_labels[1000:]

    model = define_model()

    history = model.fit(partial_x_train, partial_y_train,
                        epochs=20, batch_size=128,
                        validation_data=(x_val, y_val))

    show_loss_history(history)
    show_acc_history(history)
    results = model.evaluate(x_test, one_hot_test_labels)
    print(results)
    print(model.predict(x_test))


