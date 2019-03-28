'''
回帰の例：住宅価格の予測
'''

from keras.datasets import boston_housing

if __name__ == '__main__':
    # データの読み込み
    (train_data, train_labels), (test_data, test_labels) = \
        boston_housing.load_data()

