import datetime
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.regularizers import l2

import load_json
import text_glove
import glove_wrapper

'''
    Train a LSTM Language Model on ~100k web articles

    GPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_lstm.py
'''

GLOVE_DIM = glove_wrapper.NUM_DIM
VOCAB_SIZE = glove_wrapper.NUM_TOKENS

BATCH_SIZE = 128
N_TOTAL_EPOCHS = 100
N_EPOCHS_PER_SAVE = 4
DROPOUT = 0.5
REG = 0.001

MAX_SEQ_LENGTH = 256

print 'LOADING DATASET'
texts = load_json.get_all_text()
num_train = len(texts)
print '%d TRAINING SENTENCES' % num_train

vecs = [text_glove.text_to_vec(text_seq, MAX_SEQ_LENGTH) for text_seq in texts]
one_hots = [text_glove.text_to_index(text_seq, MAX_SEQ_LENGTH) for text_seq in texts]
assert len(vecs) == num_train
assert len(one_hots) == num_train

X = np.zeros((num_train, MAX_SEQ_LENGTH, GLOVE_DIM))
Y = np.zeros((num_train, MAX_SEQ_LENGTH, VOCAB_SIZE))
for i in range(num_train):
    seq = texts[i]
    seq_length = len(seq)
    for j in range(min(MAX_SEQ_LENGTH, seq_length)):
        X[i,j] = vecs[i][j]
        Y[i,j] = one_hots[i][j]

print 'BUILDING MODEL'
model = Sequential()
model.add(LSTM(GLOVE_DIM, GLOVE_DIM, return_sequences=True))
model.add(Dropout(DROPOUT))
model.add(Dense(GLOVE_DIM, VOCAB_SIZE, W_regularizer=l2(REG), init='glorot_normal'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

print 'TRAINING'

train_history = {}
for i in range(N_TOTAL_EPOCHS/N_EPOCHS_PER_SAVE):
    try:
        fit_callback = model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=N_EPOCHS_PER_SAVE, show_accuracy=True)
        this_history = fit_callback.history
        for key in this_history:
            if key in train_history:
                train_history[key].extend(this_history[key])
            else:
                train_history[key] = this_history[key]
    except KeyboardInterrupt:
        print '\n\n===== TRAINING INTERRUPTED ====='

    model_save_file = '../models/model_epoch_' + str((i+1)*N_EPOCHS_PER_SAVE) + '.hdf5'
    model.save_weights(model_save_file)
    generate_text.generate(model)

datetime_str = str(datetime.datetime.now()).replace(' ', '_')
epochs_finished = len(train_history['loss'])

# save loss by epoch fig
inputs, handles = [], []
train_loss_line, = plt.plot(epochs_finished, train_history['loss'], color='g')
inputs.append(train_loss_line)
handles.append('train loss')
plt.legend(inputs, handles)
title = 'loss_' + datetime_str
plt.title(title)
plt.savefig('../graphs/' + title + '.png')
plt.close()

# save acc by epoch fig
inputs, handles = [], []
train_acc_line, = plt.plot(epochs_finished, train_history['acc'], color='g')
inputs.append(train_acc_line)
handles.append('train acc')
plt.legend(inputs, handles)
title = 'acc_' + datetime_str
plt.title(title)
plt.savefig('../graphs/' + title + '.png')
plt.close()
