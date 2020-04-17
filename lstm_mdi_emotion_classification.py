import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
import sys
from keras.callbacks import LambdaCallback

# sortmax 结果转 onehot
def props_to_onehot(props):
    if isinstance(props, list):
        props = np.array(props)
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]), dtype=np.uint8)
    b[np.arange(len(a)), a] = 1
    return b


# accuracy of softmax predicts
def acc(y_preds, y):
    correct_prediction = np.equal(np.argmax(y_preds, 1), np.argmax(y, 1))
    return np.mean(correct_prediction.astype(int))

def load_data():
    data_path = '/home/zhao/Desktop/datasets/'
    with np.load(data_path + 'vec.npz') as data_train:
        x = data_train['x']
        y = data_train['y']
        x, y = np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)
        permutation = np.random.permutation(x.shape[0])
        x = x[permutation, :, :]
        y= y[permutation, :]

        train_size = int(x.shape[0] * 0.8)
        x_train = x[:train_size, ::]
        x_test = x[train_size:, ::]
        y_train = y[:train_size, :]
        y_test = y[train_size:, :]
        print('x_train shape:', x_train.shape, 'y_train shape:', y_train.shape)
        print('x_test shape:', x_test.shape, 'y_test shape:', y_test.shape)
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data()
_, Tx, n_freq = x_train.shape
_, y_len = y_train.shape


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(256, input_shape=(Tx, n_freq)))
model.add(Dense(y_len, activation='softmax'))

model.summary()

opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

model.fit(x_train, y_train,
          epochs=20,
          batch_size=64,
          validation_data=(x_test, y_test))

print('train set accuracy:', acc(model.predict(x_train), y_train))
print('test set accuracy:', acc(model.predict(x_test), y_test))

emotion_dict = {'excited': 0, 'angry': 1, 'sad': 2, 'relaxed': 3}
predict_t = np.zeros((4, 4))
y_test = np.argmax(y_test, axis=1)
y_peds = np.argmax(model.predict(x_test), axis=1)
for pre, real in zip(y_peds, y_test):
    predict_t[pre, real] += 1
print(emotion_dict.keys())
print(predict_t)