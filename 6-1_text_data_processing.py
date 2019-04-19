'''
テキストデータの操作
'''

import numpy as np
import string
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

# 単語レベルでの単純なone-hotエンコーディング
def simple_one_hot_encoding_by_word():
    # 初期データ：サンプルごとにエントリが１つ含まれている
    # （この単純な例では、サンプルは単なる１つの文章だが、文書全体でも良い）
    samples = ['The cat sat on the mat.', 'The dog ate my homework.']

    # データに含まれている全てのトークンのインデックスを構築
    token_index = {}
    for sample in samples:
        # ここでは単にsplitメソッドを使ってサンプルをトークン化する
        # 実際には、サンプルから句読点と特殊な文字を取り除くことになる
        for word in sample.split():
            if word not in token_index:
                # 一意な単語にそれぞれ一意なインデックスを割り当てする
                # インデックス0をどの単語にも割り当てないことに注意
                token_index[word] = len(token_index) + 1

    # 次に、サンプルをベクトル化する：サンプルごとに最初のmax_length個の単語だけを考慮
    max_length = 10

    # 結果の格納場所
    results = np.zeros((len(samples),
                        max_length,
                        max(token_index.values()) + 1))
    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = token_index.get(word)
            results[i, j, index] = 1.

# 文字レベルでの単純なont-hotエンコーディング
def simple_one_hot_encoding_by_char():
    samples = ['The cat sat on the mat.', 'The dog ate my homework.']
    characters = string.printable # 全て印字可能なASCII文字
    token_index = dict(zip(characters, range(1, len(characters) + 1)))

    max_length = 30
    results = np.zeros((len(samples),
                        max_length,
                        max(token_index.values()) + 1))
    for i, sample in enumerate(samples):
        for j, character in list(enumerate(sample.split()))[:max_length]:
            index = token_index.get(character)
            results[i, j, index] = 1.

# Kerasを使った単語レベルでのont-hotエンコーディング
def simple_one_hot_encoding_by_word_keras():
    samples = ['The cat sat on the mat.', 'The dog ate my homework.']

    # 出現頻度が最も高い1000個の単語だけを処理するように設定された
    # トークナイザを生成
    tokenizer = Tokenizer(num_words=1000)

    # 単語のインデックスを構築
    tokenizer.fit_on_texts(samples)

    # 文字列を整数のインデックスのリストに変換
    sequences = tokenizer.texts_to_sequences(samples)

    # 二値のone-hotエンコーディング表現を直接取得することも可能
    # one-hotエンコーディング以外のベクトル化モードもサポートされている
    one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

    # 計算された単語のインデックスを復元する方法
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

# ハッシュトリックを用いた単語レベルの単純なone-hotエンコーディング
def simple_one_hot_encoding_by_hash():
    samples = ['The cat sat on the mat.', 'The dog ate my homework.']

    # 単語をサイズが1000ベクトルとして格納
    # 単語の数が1000個に近い（またはそれ以上である）場合は、
    # ハッシュ衝突が頻発し、このエンコーディング手法の精度が低下することに注意
    dimensionality = 1000
    max_length = 10

    results = np.zeros((len(samples), max_length, dimensionality))
    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            # 単語をハッシュ化し、0~1000のランダムな整数に変換
            index = abs(hash(word)) % dimensionality
            results[i, j, index] = 1.

# 埋め込み層を使った単語埋め込みの学習
def word_embedding_by_embedding_layer():
    # Embedding層の引数は少なくとも２つ：
    #   有効なトークンの数：この場合は1000
    #   埋め込みの次元数：この場合は64
    embedding_layer = Embedding(1000, 64)

    # 特徴量として考慮する単語の数
    max_features = 10000

    # max_features個の最も出現頻度の高い単語のうち、
    # この数の単語を残してテキストをカット
    max_len = 20

    # データを複数の整数リストとして読み込む
    (x_train, y_train), (x_test, y_test) = \
        imdb.load_data(num_words=max_features)

    # 整数のリストを形状が(samples, max_len)の２次元整数テンソルに変換
    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

    model = Sequential()

    # 後から埋め込み入力を平坦化できるよう、
    # Embedding層に入力の長さとしてmax_lenを指定
    # Embedding層のあと、活性化の形状は(samples, max_len, 8)になる
    model.add(Embedding(10000, 8, input_length=max_len))

    # 埋め込みの３次元テンソルを形状が(samples, max_len * 8)の２次元テンソルに変換
    model.add(Flatten())

    # 最後に分類器を追加（分類器のみで学習しているだけ）
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    model.summary()

    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=32,
                        validation_split=0.2)


if __name__ == '__main__':
    word_embedding_by_embedding_layer()

