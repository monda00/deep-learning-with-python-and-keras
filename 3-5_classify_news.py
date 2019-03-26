'''
多クラス分類の例：ニュース配信の分類
'''

from keras.datasets import reuters

def print_data(data, labels):
    print(len(data))
    print(data[8])


if __name__ == '__main__':
    # データの読み込み
    (train_data, train_labels), (test_data, test_labels) = \
        reuters.load_data(num_words=10000)

    print_data(train_data, train_labels)

