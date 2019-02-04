import os
import numpy as np
from keras.activations import elu, linear
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Input, Lambda, BatchNormalization, GaussianNoise, ZeroPadding2D, Conv2D, AveragePooling2D, Conv1D, Dropout, Softmax, Flatten, Dense
from keras.regularizers import l1_l2
from keras.models import Model
from keras.optimizers import Adam
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import tensorflow as tf


RANDOM_STATE = 42
TARGET_DISCRETIZATION_DEPTH = 4
BATCH_SIZE = 16
NOISE = 0.1
FILTER_SIZE = 10
MAX_GRAD_NORM = 5.0
LEARNING_RATE = 0.0005
CLIPNORM = 2.0


class Discretizer:
    def __init__(self, depth: int, y: np.ndarray):
        self.tree = DecisionTreeRegressor(max_depth=depth)
        self.tree.fit(y.reshape([-1, 1]), y)
        self.onehot = OneHotEncoder(sparse=False).fit(self._tree_encode(y))

    def _tree_encode(self, y):
        return self.tree.predict(y.reshape([-1, 1])).reshape([-1, 1]).astype(np.str)

    def transform(self, y):
        return self.onehot.transform(self._tree_encode(y)).astype(np.int32)

    @property
    def possible_values(self):
        return self.onehot.categories_[0].astype(np.float32)


def categorical_mae(possible_values):
    values = K.variable(possible_values)

    def mae(y_true, y_pred):
        y_true_cont = K.gather(values, K.argmax(y_true, axis=-1))
        y_pred_cont = K.gather(values, K.argmax(y_pred, axis=-1))
        return K.mean(K.abs(y_true_cont - y_pred_cont), axis=-1)

    return mae


def categorical_mae2(possible_values):
    values = K.variable(possible_values.reshape([-1, 1]))

    def mae2(y_true, y_pred):
        y_true_cont = K.dot(y_true, values)
        y_pred_cont = K.dot(y_pred, values)
        return K.mean(K.abs(y_true_cont - y_pred_cont), axis=-1)

    return mae2


def get_model(input_shape, possible_values):
    sample_length, frequencies_count = input_shape

    input = Input(shape=input_shape)
    # Normalization
    norm_flatten = Lambda(lambda x: K.reshape(x, [-1, frequencies_count]))(input)
    norm_bn = BatchNormalization()(norm_flatten)
    normed = Lambda(lambda x: K.reshape(x, [-1, sample_length, frequencies_count]))(norm_bn)
    # Noise
    noised = GaussianNoise(NOISE)(normed)
    # Pre-liminary filter
    reshaped = Lambda(lambda x: K.reshape(x, [-1, sample_length, frequencies_count, 1]))(noised)

    conv = Conv2D(32, (6, 3), activation=elu, kernel_regularizer=l1_l2())(reshaped)
    drop = Dropout(0.5)(conv)
    pool = AveragePooling2D((5, 2))(drop)

    conv = Conv2D(32, (6, 3), activation=elu, kernel_regularizer=l1_l2())(pool)
    drop = Dropout(0.5)(conv)
    pool = AveragePooling2D((5, 2))(drop)

    conv = Conv2D(32, (3, 3), activation=elu, kernel_regularizer=l1_l2())(pool)
    drop = Dropout(0.5)(conv)
    pool = AveragePooling2D((2, 2))(drop)

    conv = Conv2D(32, (3, 3), activation=elu, kernel_regularizer=l1_l2())(pool)
    drop = Dropout(0.5)(conv)
    pool = AveragePooling2D((2, 2))(drop)

    flat = Flatten()(pool)
    fc = Dense(len(possible_values), activation='softmax', kernel_regularizer=l1_l2())(flat)

    model = Model(input, fc)
    optimizer = Adam(
        lr=LEARNING_RATE,
        clipnorm=CLIPNORM
    )
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[
        categorical_mae(possible_values),
        categorical_mae2(possible_values),
    ])
    return model


if __name__ == '__main__':
    np.random.seed(RANDOM_STATE)
    tf.set_random_seed(RANDOM_STATE)

    X_train = np.load('../data/X_train.npy').astype(np.float32)
    X_val = np.load('../data/X_val.npy').astype(np.float32)
    X_test = np.load('../data/X_test.npy').astype(np.float32)
    y_train = np.load('../data/y_train.npy').astype(np.float32)
    y_val = np.load('../data/y_val.npy').astype(np.float32)

    test_ids = [fname.replace('.csv', '')
                for fname in os.listdir('../input/test')]

    discretizer = Discretizer(TARGET_DISCRETIZATION_DEPTH, y_train)
    y_train_cat = discretizer.transform(y_train)
    y_val_cat = discretizer.transform(y_val)

    model = get_model(X_train[0].shape, discretizer.possible_values)
    model.summary()

    model.fit(X_train, y_train_cat,
              validation_data=(X_val, y_val_cat),
              batch_size=BATCH_SIZE,
              verbose=True,
              epochs=100,
              callbacks=[
                  ModelCheckpoint('model.h5', save_best_only=True),
                  ReduceLROnPlateau(patience=2),
                  EarlyStopping(patience=10),
              ])
    model.load_weights('model.h5')

    predictions = model.predict(X_val)
    y_val_pred1 = discretizer.possible_values[predictions.argmax(axis=-1)]
    y_val_pred2 = np.matmul(predictions, discretizer.possible_values)
    print('MAE1: {0}'.format(mean_absolute_error(y_val, y_val_pred1)))
    print('MAE2: {0}'.format(mean_absolute_error(y_val, y_val_pred2)))

    predictions = model.predict(X_test)
    df1 = pd.DataFrame({
        'seg_id': test_ids,
        'time_to_failure': discretizer.possible_values[predictions.argmax(axis=-1)]
    })
    df1.to_csv('submission1.csv', index=None)
    df2 = pd.DataFrame({
        'seg_id': test_ids,
        'time_to_failure': np.matmul(predictions, discretizer.possible_values)
    })
    df2.to_csv('submission2.csv', index=None)
