'''
テキストデータの操作
'''

import numpy as np
import string
from keras.preprocessing.text import Tokenizer

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


if __name__ == '__main__':
    simple_one_hot_encoding_by_hash()

