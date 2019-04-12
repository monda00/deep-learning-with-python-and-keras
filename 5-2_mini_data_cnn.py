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
                                                              1000, conv_base,
                                                              datagen)
    test_features, test_labels = extract_features(test_dir, 1000, conv_base,
                                                  datagen)

    train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
    validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
    test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

    model = build_dense_model()

    history = model.fit(train_features, train_labels,
                        epochs=30,
                        batch_size=20,
                        validation_data=(validation_features, validation_labels))


if __name__ == '__main__':
    training_model_from_trained_model()

