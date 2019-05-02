'''
畳み込みニューラルネットワークでのシーケンス処理
'''

from keras.datasets import imdb
from keras.preprocessing import sequence

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

if __name__ == '__main__':
    main()

