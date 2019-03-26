'''
3.4 二値分類の例：映画レビューの分類
'''

from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics

import numpy as np
import matplotlib.pyplot as plt

def confirm_data(train_data, train_labels):
    print(train_data[0])
    print(train_labels[0])

def num_to_words(train_data, train_labels):
    # 単語を整数のインデックスにマッピングする
    word_index = imdb.get_word_index()
    # 整数のインデックスを単語にマッピング
    reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()])
    # レビューをデコード
    decoded_rebiew = ' '.join(
        [reverse_word_index.get(i - 3, '?') for i in train_data[0]])
    # デコードした内容を表示
    print(decoded_rebiew)

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
    plt.show()

def show_acc_history(history):
    history_dict = history.history
    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']

    epochs = range(1, len(acc) + 1)
    print(epochs)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def define_model():
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # model.compile(optimizer='rmsprop',
    #               loss='binary_crossentropy',
    #               metrics=['accuracy'])
    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss=losses.binary_crossentropy,
                  metrics=[metrics.binary_accuracy])

    return model


# =============================================================================
# main関数
# ここから色々試す
# =============================================================================
def main():
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

    model = define_model()

    history = model.fit(partial_x_train, partial_y_train,
                        epochs=20, batch_size=512,
                        validation_data=(x_val, y_val))

    show_loss_history(history)
    show_acc_history(history)


if __name__ == '__main__':
    main()

