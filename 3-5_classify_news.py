'''
多クラス分類の例：ニュース配信の分類
'''

from keras.datasets import reuters

import numpy as np

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


