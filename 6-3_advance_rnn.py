'''
リカレントニューラルネットワークの高度な使い方
'''

import os
import numpy as np
from matplotlib import pyplot as plt

# 気象データセットのデータの調査
def search_data():
    data_dir = '/Users/masa/Downloads/jena_climate'
    fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

    f = open(fname)
    data = f.read()
    f.close()

    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]

    print(header)
    print(len(lines))

    # データの解析
    float_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        float_data[i, :] = values

    # 気温の時系列データのプロット
    temp = float_data[:, 1]
    plt.plot(range(len(temp)), temp)
    plt.show()

    # 最初の10日間の気温データをプロット
    plt.plot(range(1440), temp[:1440])
    plt.show()

# 時系列サンプルとそれらの目的値を生成するジェネレータ
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback

    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index,
                                     size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

# データの準備
def ready_data():
    data_dir = '/Users/masa/Downloads/jena_climate'
    fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

    f = open(fname)
    data = f.read()
    f.close()

    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]

    # データの解析
    float_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        float_data[i, :] = values

    mean = float_data[:200000].mean(axis=0)
    float_data -= mean
    std = float_data[:200000].std(axis=0)
    float_data /= std

    # 訓練、検証、テストに使用するジェネレータの準備
    lookback = 1440
    step = 6
    delay = 144
    batch_size = 128

    # 訓練ジェネレータ
    train_gen = generator(float_data,
                          lookback=lookback,
                          delay=delay,
                          min_index=0,
                          max_index=200000,
                          shuffle=True,
                          step=step,
                          batch_size=batch_size)

    # 検証ジェネレータ
    val_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=200001,
                        max_index=300000,
                        step=step,
                        batch_size=batch_size)

    # テストジェネレータ
    test_gen = generator(float_data,
                         lookback=lookback,
                         delay=delay,
                         min_index=300001,
                         max_index=None,
                         step=step,
                         batch_size=batch_size)

    # 検証データセット全体を調べるためにval_genから抽出する時間刻みの数
    val_steps = (300000 - 200001 - lookback) // batch_size

    # テストデータセット全体を調べるためにtest_genから抽出する時間刻みの数
    test_steps = (len(float_data) - 300001 - lookback) // batch_size

    evaluate_naive_method(val_steps, val_gen)

# 常識的なベースラインのMAEを計算
def evaluate_naive_method(val_steps, val_gen):
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))


if __name__ == '__main__':
    ready_data()
