from utils import tashkeel

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, TimeDistributed
from tensorflow.keras.initializers import glorot_normal

output_shape = len(tashkeel)

model = tf.keras.Sequential()
model.add(Bidirectional(LSTM(units = 256, return_sequences = True, kernel_initializer = glorot_normal(seed = 1717))))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(units = 256, return_sequences = True, kernel_initializer = glorot_normal(seed = 1717))))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(units = 512, activation = 'relu', kernel_initializer = glorot_normal(seed = 1717))))
model.add(TimeDistributed(Dense(units = 512, activation = 'relu', kernel_initializer = glorot_normal(seed = 1717))))
model.add(TimeDistributed(Dense(units = output_shape, activation = 'softmax', kernel_initializer = glorot_normal(seed = 1717))))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])





