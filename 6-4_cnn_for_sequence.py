'''
畳み込みニューラルネットワークでのシーケンス処理
'''

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

def build_model_cnn(max_features, max_len):
    model = Sequential()
    model.add(layers.Embedding(max_features, 128, input_length=max_len))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(1))

    model.summary()

    return model

def main():
    max_features = 10000    # 特徴量として考慮する単語の数
    max_len = 500           # この数の単語を残してテキストをカット

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) =\
        imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    model = build_model_cnn(max_features, max_len)

    model.compile(optimizer=RMSprop(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=128,
                        validation_split=0.2)

if __name__ == '__main__':
    main()

