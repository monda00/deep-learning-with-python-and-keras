'''
小さなデータセットでCNNを一から訓練する
'''

import os, shutil

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
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)

    # 次の500個の猫画像をvalidation_cats_dirにコピー
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)

    # 次の500個の猫画像をtest_cats_dirにコピー
    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)

    # 最初の500個の犬画像をtrain_dogs_dirにコピー
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)

    # 次の500個の犬画像をvalidation_dogs_dirにコピー
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)

    # 次の500個の犬画像をtest_dogs_dirにコピー
    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)


if __name__ == '__main__':
    make_dataset()

