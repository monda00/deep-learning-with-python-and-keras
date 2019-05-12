'''
変分オートエンコーダによる画像の生成
'''

import keras
from keras import layers
from keras import backend as K
from keras.models import Model
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from scipy.stats import norm

img_shape = (28, 28, 1)
batch_size = 16
latent_dim = 2  # 潜在空間の次元数：２次元

# VAEの損失関数を計算するためのカスタム層
class CustomVariationalLayer(keras.layers.Layer):
    def __init__(self, z_log_var):
        self.z_log_var = z_log_var

    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(
            1 + self.z_log_var - K.square(z_mean) - K.exp(self.z_log_var), axis=1)
        return K.mean(xent_loss + kl_loss)

    # カスタム層の実装ではcallメソッドを定義する
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        # この出力は使用しないが、層は何かを返さなければいけない
        return x

# 潜在空間サンプリング関数
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var) * epsilon

def main():
    # ===
    # VAEエンコーダネットワーク
    # ===
    input_img = keras.Input(shape=img_shape)

    x = layers.Conv2D(32, 3,
                      padding='same', activation='relu')(input_img)
    x = layers.Conv2D(64, 3,
                      padding='same', activation='relu',
                      strides=(2, 2))(x)
    x = layers.Conv2D(64, 3,
                      padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3,
                      padding='same', activation='relu')(x)
    shape_before_flattening = K.int_shape(x)

    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)

    # 入力画像はこれら２つのパラメータにエンコードされる
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)

    z = layers.Lambda(sampling)([z_mean, z_log_var])

    # ===
    # 潜在空間の点を画像にマッピングするVAEデコーダネットワーク
    # ===

    # この入力でzを供給
    decoder_input = layers.Input(K.int_shape(z)[1:])

    # 入力を正しい数のユニットにアップサンプリング
    x = layers.Dense(np.prod(shape_before_flattening[1:]),
                     activation='relu')(decoder_input)

    # 最後のFlatten層の直前の特徴マップと同じ形状の特徴マップに変換
    x = layers.Reshape(shape_before_flattening[1:])(x)

    # Conv2DTranspose層とConv2D層を使って
    # 元の入力画像と同じサイズの特徴マップに変換
    x = layers.Conv2DTranspose(32, 3,
                               padding='same', activation='relu',
                               strides=(2, 2))(x)
    x = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)

    # decore_inputをデコードされた画像に変換するデコーダモデルをインスタンス化
    decoder = Model(decoder_input, x)

    # このモデルをzに適用してデコードされたzを復元
    z_decoded = decoder(z)

    # カスタム層を呼び出し、最終的なモデル出力を取得するための入力と
    # デコードされた出力を渡す
    y = CustomVariationalLayer(z_log_var)([input_img, z_decoded])

    # ===
    # VAEの訓練
    # ===
    vae = Model(input_img, y)
    vae.compile(optimizer='rmsprop', loss=None)
    vae.summary()

    # MNISTの手書きの数字でVAEを訓練
    (x_train, _), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape(x_test.shape + (1,))

    vae.fit(x=x_train, y=None,
            shuffle=True,
            epochs=10,
            batch_size=batch_size,
            validation_data=(x_test, None))

    # ===
    # ２次元の潜在空間から点のグリッドを抽出し、画像にデコード
    # ===

    # 15×15の数字のグリッドを表示（数字は合計で255個）
    n = 15
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))

    # Scipyのppf関数を使って線型空間座標を変換し、潜在変数zの値を生成
    # （潜在空間の前はガウス分布であるため）
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            # 完全なバッチを形成するためにzを複数回繰り返す
            z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
            # バッチを数字の画像にデコード
            x_decoded = decoder.predict(z_sample, batch_size=batch_size)
            # バッチの最初の数字を28×28×1から28×28に変形
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()


if __name__ == '__main__':
    main()

