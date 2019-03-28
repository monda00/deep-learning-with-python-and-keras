'''
回帰の例：住宅価格の予測
'''

from keras.datasets import boston_housing
from keras import models
from keras import layers

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


if __name__ == '__main__':
    # データの読み込み
    (train_data, train_targets), (test_data, test_targets) = \
        boston_housing.load_data()

    train_data, test_data = standardize_data(train_data, test_data)

    model = build_model(train_data)
    print(model.summary())

