'''
敵対的生成ネットワーク
'''

import keras
from keras import layers
import numpy as np
import os
from keras.preprocessing import image

latent_dim = 32
height = 32
width = 32
channels = 3

def main():
    # ===
    # GANの生成者ネットワーク
    # ===
    generator_input = keras.Input(shape=(latent_dim,))

    # 入力を16×16、128チャネルの特徴マップに変換
    x = layers.Dense(128 * 16 * 16)(generator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((16, 16, 128))(x)

    # 畳み込み層を追加
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LealyReLU()(x)

    # 32×32にアップサンプリング
    x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)

    # さらに畳み込み層を追加
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 5, padding='same')(x)
    x = layers.LeakyReLU()(x)

    # 32×32、1チャネル（CIFAR10の画像の形状）の特徴マップを生成
    x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)

    # generatorモデルをインスタンス化
    # 形状が(latent_dim,)の入力を形状が(32, 32, 3)の画像にマッピング
    generator = keras.models.Model(generator_input, x)
    generator.summary()

    # ===
    # GANの判別者ネットワーク
    # ===
    discriminator_input = layers.Input(shape=(height, width, channels))
    x = layers.Conv2D(128, 3)(discriminator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Flatten()(x)

    # ドロップアウト層を１つ追加：重要なトリック！
    x = layers.Dropout(0.4)(x)

    # 分類器
    x = layers.Dense(1, activation='sigmoid')(x)

    # discriminatorモデルをインスタンス化：
    # 形状が(32, 32, 3)の入力で二値分類を実行
    discriminator = keras.models.Model(discriminator_input, x)
    discriminator.summary()

    # オプティマイザで勾配刈り込みを使用し(clipvalue),
    # 訓練を安定させるために学習率減衰を使用(decay)
    discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008,
                                                       clipvalue=1.0,
                                                       decay=1e-8)
    discriminator.compile(optimizer=discriminator_optimizer,
                          loss='binary_crossentropy')

    # ===
    # 敵対者ネットワーク
    # ===

    # discriminatorの重みを訓練不可能に設定（これはganモデルにのみ適用される）
    discriminator.trainable = False

    gan_input = keras.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = keras.models.Model(gan_input, gan_output)

    gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0,
                                             decay=1e-8)
    gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

    # ===
    # GANの訓練の実装
    # ===

    # CIFAR10のデータを読み込む
    (x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()

    # カエルの画像（クラス６）を選択
    x_train = x_train[y_train.flatten() == 6]

    # データを正規化
    x_train = x_train.reshape(
        (x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.

    iterations = 10000
    batch_size = 20

    # 生成された画像の保存先を指定
    save_dir = './fig/'

    start = 0
    for step in range(iterations):  # 訓練ループを開始
        # 潜在空間から点をランダムに抽出
        random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

        # 偽物の画像にデコーディング
        generated_images = generator.predict(random_latent_vectors)

        # 本物の画像と組み合わせる
        stop = start + batch_size
        real_images = x_train[start: stop]
        combined_images = np.concatenate([generated_images, real_images])

        # 本物の画像と偽物の画像を区別するラベルを組み立てる
        labels = np.concatenate([np.ones((batch_size, 1)),
                                 np.zeros((batch_size, 1))])

        # ラベルにランダムノイズを追加：重要なトリック！
        labels += 0.05 * np.random.random(labels.shape)

        # discriminatorを訓練
        d_loss = discriminator.train_on_batch(combined_images, labels)

        # 潜在空間から点をランダムに抽出
        random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

        # 「これらは全て本物の画像」であることを示すラベルを組み立てる
        misleading_targets = np.zeros((batch_size, 1))

        # ganモデルを通じてgeneratorを訓練
        # （ganモデルではdiscriminatorの重みが凍結される）
        a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

        start += batch_size
        if start > len(x_train) - batch_size:
            start = 0

        if step % 100 == 0:             # 100ステップおきに保存とプロット
            gan.save_weights('gan.h5')  # モデルの重みの保存

            # 成果指標を出力
            print('discriminator loss at step %s: %s' % (step, d_loss))
            print('adversarial loss at step %s: %s' % (step, a_loss))

            # 生成された画像を１つ保存
            img = image.array_to_img(generated_images[0] * 255., scale=False)
            img.save(os.path.join(save_dir,
                                  'generated_frog' + str(step) + '.png'))

            # 比較のために本物の画像を１つ保存
            img = image.array_to_img(real_images[0] * 255., scale=False)
            img.save(os.path.join(save_dir, 'real_frog' + str(step) + '.png'))


if __name__ == '__main__':
    main()

