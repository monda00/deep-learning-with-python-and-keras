'''
多クラス分類の例：ニュース配信の分類
'''

from keras.datasets import reuters

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


if __name__ == '__main__':
    # データの読み込み
    (train_data, train_labels), (test_data, test_labels) = \
        reuters.load_data(num_words=10000)

    to_words(train_data)

