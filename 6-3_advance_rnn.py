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


if __name__ == '__main__':
    search_data()

