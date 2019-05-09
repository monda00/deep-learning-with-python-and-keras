'''
Deep Dream
'''

from keras.applications import inception_v3
from keras import backend as K

def eval_loss_and_grads(x, fetch_loss_and_grads):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values

# 勾配上昇法を指定された回数に渡って実行する関数
def gradient_ascent(x, iterations, step, fetch_loss_and_grads, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x, fetch_loss_and_grads)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x

def main():
    # ===
    # 学習ずみのInception V3モデルを読み込む
    # ===

    # ここではモデルを訓練しないため、訓練関連の演算はずべて無効にする
    K.set_learning_phase(0)

    # InceptionV3ネットワークを畳み込みベースなしで構築する
    # このモデルは学習ずみのImageNetの重み付きで読み込まれる
    model = inception_v3.InceptionV3(weights='imagenet', include_top=False)

    # ===
    # DeepDreamの構成
    # ===

    # 層の名前を係数にマッピングするディクショナリ。この経緯数は最大化の対象と
    # なる損失値にその層の活性化がどれくらい貢献するのか表す。これらの層の
    # 名前は組み込みのInception V3アプリケーションにハードコーディングされて
    # いることに注意。全ての層の名前はmodel.summary()を使って確認できる。
    layer_contributions = {
        'mixed2': 0.2,
        'mixed3': 3.,
        'mixed4': 2.,
        'mixed5': 1.5,
    }

    # ===
    # 最大化の対象となる損失値を定義
    # ===

    # 層の名前を層のインスタンスにマッピングするディクショナリを作成
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    # 損失値を定義
    loss = K.variable(0.)
    for layer_name in layer_contributions:
        coeff = layer_contributions[layer_name]

        # 層の出力を出力
        activation = layer_dict[layer_name].output

        scaling = K.prod(K.cast(K.shape(activation), 'float32'))

        # 層の特徴量のL2ノルムをlossに加算
        # 非境界ピクセルのみをlossに適用することで、周辺効果を回避
        loss += coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling

    # ===
    # 勾配上昇法のプロセス
    # ===

    # 生成された画像（ドリーム）を保持するテンソル
    dream = model.input

    # ドリームの損失関数の勾配を計算
    grads = K.gradients(loss, dream)[0]

    # 勾配を正規化（重要）
    grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

    # 入力画像に基づいて損失と勾配の値を取得するKeras関数を設定
    outputs = [loss, grads]
    fetch_loss_and_grads = K.function([dream], outputs)


if __name__ == '__main__':
    main()

