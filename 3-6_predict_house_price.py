'''
回帰の例：住宅価格の予測
'''

from keras.datasets import boston_housing
from keras import models
from keras import layers

import numpy as np
import matplotlib.pyplot as plt

def standardize_data(train_data, test_data):
    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std

    test_data -= mean
    test_data /= std

    return train_data, test_data

def build_model(train_data):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',\
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def k_fold_cross_validation(train_data, trian_targets, k=4):
    num_val_samples = len(train_data) // k
    num_epochs = 80
    # all_scores = []
    all_mae_histories = []

    for i in range(k):
        print('processing fold #', i)

        # 検証データの準備：フォールドiのデータ
        val_data = \
            train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = \
            train_targets[i * num_val_samples: (i + 1) * num_val_samples]

        # 訓練データの準備：フォールドiのデータ
        partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
            axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
            axis=0)

        # Kerasモデルを構築
        model = build_model(train_data)

        # モデルをサイレントモード(verbose=0)で適合
        history = model.fit(partial_train_data, partial_train_targets,
                            validation_data=(val_data, val_targets),
                            epochs=num_epochs, batch_size=16, verbose=0)

        # モデルを検証データで評価
        # val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
        # all_scores.append(val_mae)
        mae_history = history.history['val_mean_absolute_error']
        all_mae_histories.append(mae_history)

    average_mae_history = [
        np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

    return average_mae_history

def show_scores_by_k_fold(average_mae_history):
    plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.show()

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


if __name__ == '__main__':
    # データの読み込み
    (train_data, train_targets), (test_data, test_targets) = \
        boston_housing.load_data()

    train_data, test_data = standardize_data(train_data, test_data)

    average_mae_history = k_fold_cross_validation(train_data, train_targets)

    show_scores_by_k_fold(smooth_curve(average_mae_history[10:]))

