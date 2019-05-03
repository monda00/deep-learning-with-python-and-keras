'''
Keras Functional API
'''

from keras.models import Model
from keras import layers
from keras import Input
import numpy as np

# 多入力モデル
def multi_input_model():
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

# 多出力モデル
def multi_output_model():
    vocabulary_size = 50000
    num_income_groups = 10

    posts_input = Input(shape=(None, ), dtype='int32', name='posts')
    embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)
    x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
    x = layers.MaxPooling1D(5)(x)
    x = layers.Conv1D(256, 5, activation='relu')(x)
    x = layers.Conv1D(256, 5, activation='relu')(x)
    x = layers.MaxPooling1D(5)(x)
    x = layers.Conv1D(256, 5, activation='relu')(x)
    x = layers.Conv1D(256, 5, activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)

    # 出力層に名前がついていることに注意
    age_prediction = layers.Dense(1, name='age')(x)
    income_prediction = layers.Dense(num_income_groups,
                                     activation='softmax',
                                     name='income')(x)
    gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)
    model = Model(posts_input,
                  [age_prediction, income_prediction, gender_prediction])

    model.compile(optimizer='rmsprop',
                  loss=['mse',
                        'categorical_crossentropy',
                        'binary_crossentropy'],
                  loss_weights=[0.25, 1., 10.])

    # 上記と同じ（出力層に名前をつけている場合にのみ可能）
    model.compile(optimizer='rmsprop',
                  loss={'age': 'mse',
                        'income': 'categorical_crossentropy',
                        'gender': 'binary_crossentropy'},
                  loss_weights={'age': 0.25, 'income': 1., 'gender': 10.})

    # age_targets, income_targets, gender_targetsはNumPy配列と仮定
    model.fit(posts, [age_targets, income_targets, gender_targets],
              epochs=10, batch_size=64)

    # 上記と同じ（出力層に名前をつけている場合にのみ可能）
    model.fit(posts, {'age': age_targets,
                      'income': income_targets,
                      'gender': gender_targets},
              epochs=10, batch_size=64)


if __name__ == '__main__':
    multi_output_model()

