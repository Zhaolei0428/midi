import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input, TimeDistributed, LSTM, Conv1D, Bidirectional, concatenate
from keras.layers import GRU, BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
import tensorflow as tf
from matplotlib import pyplot

#
# =====================================================================
# 不加这几句，则CONV 报错
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# =========================================================================

# softmax 结果转 onehot
def props_to_onehot(props):
    if isinstance(props, list):
        props = np.array(props)
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]), dtype=np.int32)
    b[np.arange(len(a)), a] = 1
    return b


# accuracy of softmax predicts
def acc(y_true, y_pred):
    correct_prediction = np.equal(np.argmax(y_pred, 1), np.argmax(y_true, 1))
    return np.mean(correct_prediction.astype(int))


def load_data():
    data_path = './datasets/'
    with np.load(data_path + 'vec.npz') as data:
        x_train = data['x_train']
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']

        print('x_train shape:', x_train.shape, 'y_train shape:', y_train.shape)
        print('x_test shape:', x_test.shape, 'y_test shape:', y_test.shape)
    return (x_train, y_train), (x_test, y_test)


def load_testdata():
    data_path = './datasets/test/'
    with np.load(data_path + 'vec.npz') as data:
        x = data['x']
        y = data['y']
    return x, y

def model(input_shape, output_len):
    """
    Function creating the model's graph in Keras.

    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)
    concat_shape -- shape of the midi basic features
    Returns:
    model -- Keras model instance
    """

    input = Input(shape=input_shape, name='main_input')
    ### START CODE HERE ###
    # nn layer
    # X = Dense(64, activation='sigmoid')(input)
    X = Dense(128, activation='sigmoid')(input)
    X = Dense(256, activation='sigmoid')(X)
    X = Dense(256, activation='sigmoid')(X)
    X = Dense(512, activation='sigmoid')(X)
    # X = Dense(512, activation='sigmoid')(X)
    # X = Dropout(0.2)(X)
    X = Dense(256, activation='sigmoid')(X)
    # X = Dense(256, activation='sigmoid')(X)
    # X = Dropout(0.2)(X)
    X = Dense(128, activation='sigmoid')(X)
    # X = Dropout(0.2)(X)
    X = Dense(64, activation='sigmoid')(X)
    X = Dense(output_len, activation="softmax")(X)


    model = Model(inputs=[input], outputs=X)

    return model


(x_train, y_train), (x_test, y_test) = load_data()
x_len = x_train.shape[1]
y_len = y_train.shape[1]

# build the model: a single LSTM
print('Build model...')
model = model((x_len,), y_len)

model.summary()
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])  # what is this accuracy?

plot_model(model, to_file='nn_model.png', show_shapes=True, show_layer_names=True)

history = model.fit([x_train], y_train,
          epochs=2000,
          batch_size=64,
          validation_data=(x_test, y_test))


print('train set accuracy:', acc(y_train, model.predict(x_train)))
print('validation set accuracy:', acc(y_test, model.predict(x_test)))

# validation set
emotion_dict = {'excited': 0, 'angry': 1, 'sad': 2, 'relaxed': 3}
predict_t = np.zeros((4, 4))
y_test = np.argmax(y_test, axis=1)
y_peds = np.argmax(model.predict(x_test), axis=1)
for pre, real in zip(y_peds, y_test):
    predict_t[pre, real] += 1
print(emotion_dict.keys())
print(predict_t)

# test sets
x, y = load_testdata()
print('test set accuracy:', acc(y, model.predict(x)))
predict_t = np.zeros((4, 4))
y_test = np.argmax(y, axis=1)
y_peds = np.argmax(model.predict(x), axis=1)
for pre, real in zip(y_peds, y_test):
    predict_t[pre, real] += 1
print(emotion_dict.keys())
print(predict_t)

pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()