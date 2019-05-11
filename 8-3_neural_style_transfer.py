'''
ニューラルネットワークによるスタイル変換
'''

from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.applications import vgg19
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time

# ===
# 変数の定義
# ===

# 変換したい画像へのパス
target_image_path = 'fig/portrait.jpg'

# スタイル画像へのパス
style_reference_image_path = 'fig/transfer_style_reference.jpg'

# 生成する画像のサイズ
width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / height)

# ===
# 勾配降下法のプロセスを定義
# ===

# このクラスは、損失関数の値と勾配の値を２つのメソッド呼び出しを通じて取得できる
# ようにfetch_loss_and_gradsをラッピングする。この２つのメソッド呼び出しは、
# ここで使用するSciPyのオプティマイザによって要求される
class Evaluator(object):
    def __init__(self, fetch_loss_and_grads):
        self.loss_value = None
        self.grads_value = None
        self.fetch_loss_and_grads = fetch_loss_and_grads

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = self.fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    # ImageNetから平均ピクセル値を取り除くことにより、中心を0に設定
    # これにより、vgg19.preprocess_inputによって実行される変換が逆になる
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # 画像を'BGR'から'RGB'に変換
    # これもvgg19.preprocess_inputの変換を逆にするための措置
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x

# コンテンツの損失関数
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

# スタイルの損失関数
def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

# 全変動損失関数
def total_variation_loss(x):
    a = K.square(x[:, :img_height - 1, :img_width - 1, :] -
                 x[:, 1:, :img_width - 1, :])
    b = K.square(x[:, :img_height - 1, :img_width - 1, :] -
                 x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

def main():
    # ===
    # 学習済みのVGG19ネットワークを読み込み、３つの画像に適用
    # ===

    target_image = K.constant(preprocess_image(target_image_path))
    style_reference_image = K.constant(preprocess_image(
        style_reference_image_path))

    # 生成された画像を保持するプレースホルダ
    combination_image = K.placeholder((1, img_height, img_width, 3))

    # ３つの画像を１つのバッチにまとめる
    input_tensor = K.concatenate([target_image, style_reference_image,
                                  combination_image], axis=0)

    # ３つの画像からなるバッチを入力として使用するVGG19モデルを構築
    # このモデルには、学習済みのImageNetの重みが読み込まれる
    model = vgg19.VGG19(input_tensor=input_tensor,
                        weights='imagenet',
                        include_top=False)
    print('Model loaded.')

    # ===
    # 最小化の対象となる最終的な損失関数を定義
    # ===

    # 層の名前を活性化テンソルにマッピングするディクショナリ
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    content_layer = 'block5_conv2'  # コンテンツの損失関数に使用する層の名前

    style_layers = ['block1_conv1', # スタイルの損失関数に使用する層の名前
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    # 損失関数の荷重平均の重み
    total_variation_weight = 1e-4
    style_weight = 1.
    content_weight = 0.025

    # 全てのコンポーネントをこのスカラー変数に追加することで、損失関数を定義
    loss = K.variable(0.)

    # コンテンツの損失関数を追加
    layer_features = outputs_dict[content_layer]
    target_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss += content_weight * content_loss(target_image_features,
                                          combination_features)

    # 各ターゲット層のスタイルの損失関数を追加
    for layer_name in style_layers:
        layer_features = outputs_dict[layer_name]
        style_reference_features =layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layers)) * sl

    # 全変動損失関数を追加
    loss += total_variation_weight * total_variation_loss(combination_image)

    # 損失関数を元に、生成された画像の勾配を取得
    grads = K.gradients(loss, combination_image)[0]

    # 現在の損失関数の値と勾配の値を取得する関数
    fetch_loss_and_grads = K.function([combination_image], [loss, grads])

    evaluator = Evaluator(fetch_loss_and_grads)

    # ===
    # スタイル変換ループ
    # ===

    result_prefix = './fig/style_transfer_result'
    iterations = 20

    # 初期状態：ターゲット画像
    x = preprocess_image(target_image_path)

    # 画像を平坦化：scipy.optimize.fmin_l_bfgs_bは１次元ベクトルしか処理しない
    x = x.flatten()
    for i in range(iterations):
        print('Start of iteraion', i)
        start_time = time.time()
        # ニューラルスタイル変換の損失関数を最小化するために
        # 生成された画像のピクセルにわたってL-BFGS最適化を実行
        # 損失関数を計算する関数と勾配を計算する関数を２つの別々の引数として
        # 渡さなければならないことに注意
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x,
                                         fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        # この時点の生成された画像を保存
        img = x.copy().reshape((img_height, img_width, 3))
        img = deprocess_image(img)
        fname = result_prefix + '_at_iteration_%d.png' % i
        imsave(fname, img)
        end_time = time.time()
        print('Image save as', fname)
        print('Iteration %d completed in %ds' % (i, end_time - start_time))


if __name__ == '__main__':
    main()

