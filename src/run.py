import datetime
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random

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

BATCH_SIZE = 64
N_TOTAL_EPOCHS = 100
N_EPOCHS_PER_SAVE = 4
DROPOUT = 0.5
REG = 0.001

MAX_SEQ_LENGTH = 64

print 'LOADING DATASET'
texts = load_json.get_all_text()
num_train = len(texts)
print '%d TRAINING SENTENCES' % num_train

# TOO MEMORY INTENSIVE -- MUST USE BATCHING WHILE TRAINING
#vecs = [text_glove.text_to_vec(text_seq, MAX_SEQ_LENGTH) for text_seq in texts]
#one_hots = [text_glove.text_to_index(text_seq, MAX_SEQ_LENGTH) for text_seq in texts]
#assert len(vecs) == num_train
#assert len(one_hots) == num_train
#
#X = np.zeros((num_train, MAX_SEQ_LENGTH, GLOVE_DIM))
#Y = np.zeros((num_train, MAX_SEQ_LENGTH, VOCAB_SIZE))
#for i in range(num_train):
#    seq = texts[i]
#    seq_length = min(MAX_SEQ_LENGTH, len(seq))
#    for j in range(seq_len):
#        X[i,j] = vecs[i][j]
#        # y_t = one hot for index of x_(t-1), except for last
#        if j < seq_len-1:
#            Y[i,j] = one_hots[i][j+1]
#        else:
#            Y[i,j] = one_hots[i][j]

# LSTM with hidden size of GLOVE_DIM
print 'BUILDING MODEL'
model = Sequential()
model.add(LSTM(GLOVE_DIM, GLOVE_DIM, return_sequences=True))
model.add(Dropout(DROPOUT))
model.add(Dense(GLOVE_DIM, VOCAB_SIZE, W_regularizer=l2(REG), init='glorot_normal'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

print 'TRAINING'

train_history = {}
batches_per_epoch = num_train/BATCH_SIZE
for i in range(N_TOTAL_EPOCHS/N_EPOCHS_PER_SAVE):
    epoch_loss, epoch_acc = 0.0, 0.0
    for j in range(batches_per_epoch):
        # sample for batch
        batch_inds = random.sample(range(num_train), BATCH_SIZE)
        batch_texts = [texts[ind] for ind in batch_inds]
        vecs = [text_glove.text_to_vec(text_seq, MAX_SEQ_LENGTH) for text_seq in batch_texts]
        one_hots = [text_glove.text_to_index(text_seq, MAX_SEQ_LENGTH) for text_seq in batch_texts]
        assert len(vecs) == BATCH_SIZE
        assert len(one_hots) == BATCH_SIZE
        # needs to be numpy tensor
        X = np.zeros((BATCH_SIZE, MAX_SEQ_LENGTH, GLOVE_DIM))
        Y = np.zeros((BATCH_SIZE, MAX_SEQ_LENGTH, VOCAB_SIZE))
        for k in range(BATCH_SIZE):
            seq = vecs[k]
            seq_len = min(MAX_SEQ_LENGTH, len(seq))
            for l in range(seq_len):
                X[k,l] = vecs[k][l]
                # y_t = one hot for index of x_(t-1), except for last
                if l < seq_len-1:
                    Y[k,l] = one_hots[k][l+1]
                else:
                    Y[k,l] = one_hots[k][l]
        batch_loss, batch_accuracy = model.train_on_batch(X, Y, accuracy=True)
        epoch_loss += batch_loss
        batch_accuracy += batch_accuracy
        if j % 25 == 0:
            print '\t\tBATCH %d of %d' % (j, batches_per_epoch)
            print '\t\t\tloss: %f, acc: %f' % (batch_loss, batch_accuracy)
    print '\tEPOCH %d' % (i+1)
    print '\tavg loss: %f, avg acc: %f' % (epoch_loss/batches_per_epoch, epoch_acc/batches_per_epoch)



    #try:
    #    fit_callback = model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=N_EPOCHS_PER_SAVE, show_accuracy=True)
    #    this_history = fit_callback.history
    #    for key in this_history:
    #        if key in train_history:
    #            train_history[key].extend(this_history[key])
    #        else:
    #            train_history[key] = this_history[key]
    #except KeyboardInterrupt:
    #    print '\n\n===== TRAINING INTERRUPTED ====='

    # save model every N_EPOCHS_PER_SAVE epochs
    model_save_file = '../models/model_epoch_' + str((i+1)*N_EPOCHS_PER_SAVE) + '.hdf5'
    model.save_weights(model_save_file)
    generate_text.generate(model)

#datetime_str = str(datetime.datetime.now()).replace(' ', '_')
#epochs_finished = len(train_history['loss'])

# save loss by epoch fig
#inputs, handles = [], []
#train_loss_line, = plt.plot(epochs_finished, train_history['loss'], color='g')
#inputs.append(train_loss_line)
#handles.append('train loss')
#plt.legend(inputs, handles)
#title = 'loss_' + datetime_str
#plt.title(title)
#plt.savefig('../graphs/' + title + '.png')
#plt.close()

# save acc by epoch fig
#inputs, handles = [], []
#train_acc_line, = plt.plot(epochs_finished, train_history['acc'], color='g')
#inputs.append(train_acc_line)
#handles.append('train acc')
#plt.legend(inputs, handles)
#title = 'acc_' + datetime_str
#plt.title(title)
#plt.savefig('../graphs/' + title + '.png')
#plt.close()
