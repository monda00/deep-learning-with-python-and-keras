'''
テキストデータの操作
'''

import numpy as np
import os, shutil
import string
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

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

# IMDbデータセットの処理
def preprocessing_imdb(max_len, training_samples, validation_samples, max_words):
    # IMDbデータセットが置かれているディレクトリ
    imdb_dir = './data/aclImdb'

    train_dir = os.path.join(imdb_dir, 'train')
    labels = []
    texts = []

    # 元のIMDbデータセットのラベルを処理
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(train_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname))
                texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=max_len)

    labels = np.asarray(labels)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # データを訓練データセットと検証データセットに分割：
    # ただし、サンプルが順番に並んでいる（否定的なレビューの後に肯定的なレビューが
    # 配置されている）状態のデータを使用するため、最初にデータをシャッフル
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    lebels = labels[indices]

    x_train = data[:training_samples]
    y_train = labels[:training_samples]
    x_val = data[training_samples: training_samples + validation_samples]
    y_val = labels[training_samples: training_samples + validation_samples]

    return x_train, y_train, x_val, y_val, word_index

def preprocessing_glove(max_words, embedding_dim, word_index):
    # GloVeの単語埋め込みファイルを解析
    # GloVeの埋め込みファイルが置かれているディレクトリ
    glove_dir = './data/glove.6B'

    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if i < max_words:
            if embedding_vector is not None:
                # 埋め込みインデックスで見つからない単語は0で埋める
                embedding_matrix[i] = embedding_vector

    return embedding_matrix

# モデル定義
def build_model(max_words, embedding_dim, max_len, embedding_matrix):
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=max_len))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    # GloVeの埋め込みをモデルに読み込む
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])

    return model

# 結果をプロット
def show_result(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # 正解率をプロット
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and Validation accuracy')
    plt.legend()
    plt.savefig('./fig/6-1_training_and_validation_accuracy.png')

    plt.figure()

    # 損失値をプロット
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.savefig('./fig/6-1_training_and_validation_loss.png')

# テキストのトークン化から単語埋め込みまで
def token_to_word_embedding():
    # IMDbデータのテキストをトークン化
    max_len = 100               # 映画レビューを100ワードでカット
    training_samples = 200      # 200個のサンプルで訓練
    validation_samples = 10000  # 10000個のサンプルで検証
    max_words = 10000           # データセットの最初から10000ワードのみを考慮

    x_train, y_train, x_val, y_val, word_index =\
        preprocessing_imdb(max_len, training_samples, validation_samples, max_words)

    embedding_dim = 100

    embedding_matrix = preprocessing_glove(max_words, embedding_dim, word_index)

    model = build_model(max_words, embedding_dim, max_len, embedding_matrix)

    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=32,
                        validation_data=(x_val, y_val))

    model.save_weights('pre_trained_glove_model.h5')

    show_result(history)


if __name__ == '__main__':
    token_to_word_embedding()

