'''
LSTMによるテキスト生成
'''

import keras
import numpy as np
from keras import layers
import random
import sys

# モデルの予測に基づいて次の文字をサンプリング
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def main():
    # ---
    # 最初のテキストファイルのダウンロードと解析
    # ---
    path = keras.utils.get_file(
        'nietzsche.txt',
        origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
    text = open(path).read().lower()
    print('Corpus length:', len(text))

    # ---
    # 文字シーケンスのベクトル化
    # ---
    maxlen = 60         # 60文字のシーケンスを抽出
    step = 3            # 3文字おきに新しいシーケンスをサンプリング
    sentences = []      # 抽出されたシーケンスを保持
    next_chars = []     # 目的値（次にくる文字）を保持

    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])

    print('Number of sequences:', len(sentences))

    # コーパスの一意な文字のリスト
    chars = sorted(list(set(text)))
    print('Unique characters:', len(chars))

    # これらの文字をリストcharsのインデックスにマッピンングするディクショナリ
    char_indices = dict((char, chars.index(char)) for char in chars)

    print('Vectorization...')

    # one-hotエンコーディング
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    # ---
    # 単層LSTMモデル
    # ---
    model = keras.models.Sequential()
    model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(layers.Dense(len(chars), activation='softmax'))

    # ---
    # モデルのコンパイル
    # ---
    optimizer = keras.optimizers.RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    # ---
    # テキスト生成ループ
    # ---

    # モデルを60エポックで訓練
    for epoch in range(1, 60):
        print('epoch', epoch)

        # 1エポックでデータを学習
        model.fit(x, y, batch_size=128, epochs=1)

        # テキストシードをランダムに選択
        start_index = random.randint(0, len(text) - maxlen - 1)
        generated_text = text[start_index: start_index + maxlen]
        print('--- Generating with seed: "' + generated_text + '"')

        # ある範囲内の異なるサンプリング温度を試してみる
        for temperature in [0.2, 0.5, 1.0, 1.2]:
            print('------ temperature:', temperature)
            sys.stdout.write(generated_text)

            # 400文字を生成
            for i in range(400):
                # これまでに生成された文字にone-hotエンコーディングを適用
                sampled = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(generated_text):
                    sampled[0, t, char_indices[char]] = 1

                # 次の文字をサンプリング
                preds = model.predict(sampled, verbose=0)[0]
                next_index = sample(preds, temperature)
                next_char = chars[next_index]

                generated_text += next_char
                generated_text = generated_text[1:]

                sys.stdout.write(next_char)
                sys.stdout.flush()


if __name__ == '__main__':
    main()

