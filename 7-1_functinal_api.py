'''
Keras Functional API
'''

from keras.models import Model
from keras import layers
from keras import Input
import numpy as np

def dubble_input_model():
    text_vocabulary_size = 10000
    question_vocabulary_size = 10000
    answer_vocabulary_size = 500

    # テキスト入力は整数の可変長のシーケンス
    # なお、必要であれば、入力に名前をつけることもできる
    text_input = Input(shape=(None, ), dtype='int32', name='text')

    # 入力をサイズが64のベクトルシーケンスに埋め込む
    embedded_text = layers.Embedding(
        text_vocabulary_size, 64)(text_input)

    # LSTMを通じてこれらのベクトルを単一のベクトルにエンコード
    encoded_text = layers.LSTM(32)(embedded_text)

    # 質問入力でも（異なる層のインスタンスを使って）同じプロセスを繰り返す
    question_input = Input(shape=(None, ), dtype='int32', name='question')
    embedded_question = layers.Embedding(
        question_vocabulary_size, 32)(question_input)
    encoded_question = layers.LSTM(16)(embedded_question)

    # エンコードされたテキストと質問を連結
    concatenated = layers.concatenate([encoded_text, encoded_question],
                                      axis=-1)

    # ソフトマックス分類器を追加
    answer = layers.Dense(
        answer_vocabulary_size, activation='softmax')(concatenated)

    # モデルをインスタンス化するときには、２つの入力と１つの出力を指定
    model = Model([text_input, question_input], answer)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    model.summary()

    # 多入力モデルへのデータ供給
    num_samples = 1000
    max_length = 100

    # ダミーのNumPyデータを生成
    text = np.random.randint(1, text_vocabulary_size,
                             size=(num_samples, max_length))
    question = np.random.randint(1, question_vocabulary_size,
                                 size=(num_samples, max_length))

    # 答えに（整数ではなく）one-hotエンコーディングを適用
    answers = np.zeros(shape=(num_samples, answer_vocabulary_size))
    indices = np.random.randint(0, answer_vocabulary_size, size=num_samples)
    for i, x in enumerate(answers):
        x[indices[i]] = 1

    # 入力リストを使った場合
    model.fit([text, question], answers, epochs=10, batch_size=128)

    # 入力ディクショナリを使った場合（入力に名前をつける場合のみ）
    model.fit({'text': text, 'question': question}, answers,
              epochs=10, batch_size=128)


if __name__ == '__main__':
    dubble_input_model()

