from nemlar import nemlar
from utils import *
import random, math
from model import model

from tensorflow.keras.callbacks import ModelCheckpoint


X, y = prepare_text(nemlar.text, 500)

data = list(zip(X, y))
random.seed(1717)
random.shuffle(data)
X, y = zip(*data)
X, y = np.asarray(X), np.asarray(y)

X_train, X_val, y_train, y_val = X[:math.floor(len(X)*0.7)], X[math.floor(len(X)*0.7):], y[:math.floor(len(X)*0.7)], y[math.floor(len(X)*0.7):]


input_shape = X_train[0].shape[1]
output_shape = y_train[0].shape[1]


CHECKPOINT_PATH = 'checkpoints/epoch{epoch:02d}.ckpt'
checkpoint_callback = ModelCheckpoint(filepath = CHECKPOINT_PATH, save_weights_only = True, verbose = 1)

model.fit(x = X_train, y = y_train, batch_size = 64, epochs = 30, validation_data = (X_val, y_val), callbacks = [checkpoint_callback])