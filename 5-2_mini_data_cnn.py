'''
小さなデータセットでCNNを一から訓練する
'''

import os, shutil
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications import VGG16
from keras.models import load_model
from keras import backend as K

def make_dataset():
    # 元のデータセットを展開したディレクトリへのパス
    original_dataset_dir = './data/train'
    # より小さなデータセットを格納するディレクトリへのパス
    base_dir = './data/cats_and_dogs_small'
    # os.mkdir(base_dir)

    # 訓練データセット、検証データセット、テストデータセットを配置するディレクトリ
    train_dir = os.path.join(base_dir, 'train')
    # os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir, 'validation')
    # os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir, 'test')
    # os.mkdir(test_dir)

    # 訓練用の猫の画像を配置するディレクトリ
    train_cats_dir = os.path.join(train_dir, 'cats')
    # os.mkdir(train_cats_dir)

    # 訓練用の犬の画像を配置するディレクトリ
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    # os.mkdir(train_dogs_dir)

    # 検証用の猫の画像を配置するディレクトリ
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    # os.mkdir(validation_cats_dir)

    # 検証用の犬の画像を配置するディレクトリ
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    # os.mkdir(validation_dogs_dir)

    # テスト用の猫の画像を配置するディレクトリ
    test_cats_dir = os.path.join(test_dir, 'cats')
    # os.mkdir(test_cats_dir)

    # テスト用の犬の画像を配置するディレクトリ
    test_dogs_dir = os.path.join(test_dir, 'dogs')
    # os.mkdir(test_dogs_dir)

    # 最初の1000個の猫画像をtrain_cats_dirにコピー
    # fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    # for fname in fnames:
    #     src = os.path.join(original_dataset_dir, fname)
    #     dst = os.path.join(train_cats_dir, fname)
    #     shutil.copyfile(src, dst)

    # 次の500個の猫画像をvalidation_cats_dirにコピー
    # fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    # for fname in fnames:
    #     src = os.path.join(original_dataset_dir, fname)
    #     dst = os.path.join(validation_cats_dir, fname)
    #     shutil.copyfile(src, dst)

    # 次の500個の猫画像をtest_cats_dirにコピー
    # fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    # for fname in fnames:
    #     src = os.path.join(original_dataset_dir, fname)
    #     dst = os.path.join(test_cats_dir, fname)
    #     shutil.copyfile(src, dst)

    # 最初の500個の犬画像をtrain_dogs_dirにコピー
    # fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    # for fname in fnames:
    #     src = os.path.join(original_dataset_dir, fname)
    #     dst = os.path.join(train_dogs_dir, fname)
    #     shutil.copyfile(src, dst)

    # 次の500個の犬画像をvalidation_dogs_dirにコピー
    # fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    # for fname in fnames:
    #     src = os.path.join(original_dataset_dir, fname)
    #     dst = os.path.join(validation_dogs_dir, fname)
    #     shutil.copyfile(src, dst)

    # 次の500個の犬画像をtest_dogs_dirにコピー
    # fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    # for fname in fnames:
    #     src = os.path.join(original_dataset_dir, fname)
    #     dst = os.path.join(test_dogs_dir, fname)
    #     shutil.copyfile(src, dst)

    return train_dir, validation_dir, train_cats_dir

def build_cnn_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    return model

def build_cnn_dropout_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    return model

def make_generator(train_dir, validation_dir):
    # 全ての画像を1/255でスケーリング
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    return train_generator, validation_generator

def make_extended_data_generator(train_dir, validation_dir):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,)

    # 検証データは水増しするべきではないことに注意
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    return train_generator, validation_generator


def show_loss_and_acc(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # 正解率をプロット
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.savefig('./fig/acc.png')

    # 損失値をプロット
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.savefig('./fig/loss.png')

def extended_data():
    datagen = ImageDataGenerator(rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest')

    return datagen

def show_extended_data(datagen, train_cats_dir):
    fnames = [os.path.join(train_cats_dir, fname)
              for fname in os.listdir(train_cats_dir)]

    # 水増しする画像を選択
    img_path = fname[3]

    # 画像を読み込み、サイズを変更
    img = image.load_img(img_path, target_size=(150, 150))

    # 形状が(150, 150, 3)のNumPy配列に変換
    x = image.img_to_array(img)

    # (1, 150, 150, 3)に変形
    x = x.reshape((1, ) + x.shape)

    # ランダムに変換した画像のパッチを生成する
    # 無限ループとなるため、何らかのタイミングでbreakする必要がある
    i = 0
    for batch in datagen.flow(x, batch_size=1):
        plt.figure(i)
        imgplot = plt.imshow(image.array_to_img(batch[0]))
        i += 1
        if i % 4 == 0:
            break

    plt.show()

def training_model_from_scratch():
    train_dir, validation_dir, train_cats_dir = make_dataset()

    model = build_cnn_dropout_model()
    model.summary()

    train_generator, validation_generator =\
        make_extended_data_generator(train_dir, validation_dir)

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=100,
                                  epochs=100,
                                  validation_data=validation_generator,
                                  validation_steps=50)

    model.save('cats_and_dogs_small_2.h5')

    show_loss_and_acc(history)

def extract_features(directory, sample_count, batch_size, conv_base, datagen):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(directory,
                                            target_size=(150, 150),
                                            batch_size=batch_size,
                                            class_mode='binary')

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # ジェネレータはデータを無限ループで生成するため
            # 画像を一通り処理したらbreakしなければならない
            break

    return features, labels

def build_dense_model():
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    return model

def show_loss_and_acc_trained(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    # 正解率をプロット
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('./fig/acc_trained.png')

    plt.figure()

    # 損失値をプロット
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('./fig/loss_trained.png')

def build_extend_conv_base_model(conv_base):
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    conv_base.trainable = False

def training_model_from_trained_model():
    base_dir = './data/cats_and_dogs_small'

    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    datagen = ImageDataGenerator(rescale=1./255)
    batch_size=20

    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(150, 150, 3))
    conv_base.summary()

    train_features, train_labels = extract_features(train_dir, 2000, batch_size,
                                                    conv_base, datagen)
    validation_features, validation_labels = extract_features(validation_dir,
                                                              1000, batch_size,
                                                              conv_base,
                                                              datagen)
    test_features, test_labels = extract_features(test_dir, 1000, batch_size,
                                                  conv_base, datagen)

    train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
    validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
    test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

    model = build_dense_model()

    history = model.fit(train_features, train_labels,
                        epochs=30,
                        batch_size=20,
                        validation_data=(validation_features, validation_labels))

def training_model_from_trained_model_with_extend_data():
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(150, 150, 3))

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,)

    # 検証データは水増しするべきではないことに注意
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    model = build_extend_conv_base_model(conv_base)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=2e-5),
                  metrics=['acc'])

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=100,
                                  epochs=30,
                                  validation_data=validation_generator,
                                  validation_steps=50,
                                  verbose=2)

def training_model_fine_tuning():
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(150, 150, 3))

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,)

    # 検証データは水増しするべきではないことに注意
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    conv_base.trainable = True
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    model = conv_base

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-5),
                  metrics=['acc'])

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=100,
                                  epochs=100,
                                  validation_data=validation_generator,
                                  validation_steps=50)

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def showing_middle_layer_output():
    model = load_model('cats_and_dogs_small_2.h5')
    model.summary()

    img_path = \
        './data/cats_and_dogs_small/test/cats/cat.1700.jpg'

    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)

    img_tensor /= 255.

    #print(img_tensor.shape)

    #plt.imshow(img_tensor[0])
    #plt.show()

    # 出力側の８つの層から出力を抽出
    layer_outputs = [layer.output for layer in model.layers[:8]]
    # 特定の入力をもとに、これらの出力を返すモデルを作成
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    # ５つのNumPy配列（層の活性化ごとに１つ）のリストを返す
    activations = activation_model.predict(img_tensor)

    first_layer_activation = activations[0]
    plt.matshow(first_layer_activation[0, :, :, 3], cmap='viridis')
    plt.show()

def showing_middle_layer_output_all():
    model = load_model('cats_and_dogs_small_2.h5')
    model.summary()

    img_path = \
        './data/cats_and_dogs_small/test/cats/cat.1700.jpg'

    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)

    img_tensor /= 255.

    # 出力側の８つの層から出力を抽出
    layer_outputs = [layer.output for layer in model.layers[:8]]
    # 特定の入力をもとに、これらの出力を返すモデルを作成
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    # ５つのNumPy配列（層の活性化ごとに１つ）のリストを返す
    activations = activation_model.predict(img_tensor)

    # プロットの一部として使用する層の名前
    layer_names = []
    for layer in model.layers[:8]:
        layer_names.append(layer.name)

    images_per_row = 16

    # 特徴マップを表示
    for layer_name, layer_activation in zip(layer_names, activations):
        # 特徴マップに含まれている特徴量の数
        n_features = layer_activation.shape[-1]

        # 特徴マップの形状(1, size, size, n_features)
        size = layer_activation.shape[1]

        # この行列で活性化のチャネルをタイル表示
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        # 各フィルタを１つの大きな水平グリッドでタイル表示
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :,
                                                 col * images_per_row + row]
                # 特徴量の見た目をよくするための後処理
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                             row * size : (row + 1) * size] = channel_image

        # グリッドを表示
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

    # plt.savefig("./fig/activations.png")
    plt.show()

def deprocess_image(x):
    # テンソルを正規化：中心を0、標準偏差を0.1にする
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # [0, 1]にクリッピング
    x += 0.5
    x = np.clip(x, 0, 1)

    # RGB配列に変換
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def generate_pattern(layer_name, filter_index, model, size=150):
    # ターゲット層のn番目のフィルタの活性化を最大化する損失関数を構築
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # この損失関数を使って入力画像の勾配を計算
    grads = K.gradients(loss, model.input)[0]

    # 正規化トリック：勾配を正則化
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # 入力画像に基づいて損失値と勾配値を返す関数
    iterate = K.function([model.input], [loss, grads])

    # 最初はノイズが含まれたグレースケール画像を使用
    input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.

    # 勾配上昇上昇法を40ステップ実行
    step = 1.
    for i in range(40):
        # 損失値と勾配値を計算
        loss_value, grads_value = iterate([input_img_data])
        # 損失が最大になる方向に入力画像を調節

    img = input_img_data[0]
    return deprocess_image(img)

def showing_filter():
    model = VGG16(weights='imagenet', include_top=False)

    layer_name = 'block3_conv1'
    filter_index = 0

    plt.imshow(generate_pattern('block3_conv1', 0, model))
    plt.show()

if __name__ == '__main__':
    showing_filter()

