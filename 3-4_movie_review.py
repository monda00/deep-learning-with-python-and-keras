'''
3.4 二値分類の例：映画レビューの分類
'''

from keras.datasets import imdb

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


def main():
    (train_data, train_labels), (test_data, test_labels) = \
        imdb.load_data(num_words=10000)

    num_to_words(train_data, train_labels)


if __name__ == '__main__':
    main()

