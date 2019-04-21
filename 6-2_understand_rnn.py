'''
リカレントニューラルネットワークを理解する
'''

import numpy as np

# 単純なRNNのNumPy実装
def simple_rnn():
    timesteps = 100         # 入力シーケンスの時間刻みの数
    input_features = 32     # 入力特徴空間の次元の数
    output_features = 64    # 出力特徴空間の次元の数

    # 入力データ：ランダムにノイズを挿入
    inputs = np.random.random((timesteps, input_features))

    # 初期状態：全て0のベクトル
    state_t = np.zeros((output_features, ))

    # ランダムな重み行列を作成
    W = np.random.random((output_features, input_features))
    U = np.random.random((output_features, output_features))
    b = np.random.random((output_features, ))

    successive_outputs = []

    # input_tは形状が(input_features, )のベクトル
    for input_t in inputs:
        # 入力と現在の状態（１つ前の出力）を結合して現在の出力を取得
        output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
        # この出力をリストに格納
        successive_outputs.append(output_t)
        # 次の時間刻みのためにRNNの状態を更新
        state_t = output_t

    # 最終的な出力は形状が(timesteps, output_features)の２次元テンソル
    final_output_sequence = np.stack(successive_outputs, axis=0)
    print(final_output_sequence)


if __name__ == '__main__':
    simple_rnn()

