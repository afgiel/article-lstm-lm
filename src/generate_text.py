import numpy as np
import random, sys
import glove_wrapper

NUM_PRED_WORDS = 30
glove_matrix = glove_wrapper.GloveWrapper()

def sample(a, temperature=1.0):
    """Samples an index from a probability array"""
    a = np.log(a)/temperature
    a = np.exp(a)/np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1,a,1))

def generate(fit_model):
    """Generates list of predicted words given an already fit model"""
    pred_indices = []
    pred_words = []
    # Replace start_index with actual start token
    start_index = random.randint(0, len(text) - maxlen - 1)
    current_vec = glove_matrix.get_vec(start_index)

    for iteration in range(NUM_PRED_WORDS):
        preds = fit_model.predict(current_vec, verbose=0)
        pred_index = sample(preds)
        pred_indices = pred_indices + [next_index]
        pred_words = pred_words + [glove_matrix.get_word(pred_index)]
        current_vec = glove_matrix.get_vec(pred_index)

    assert NUM_PRED_WORDS == len(pred_words)
    return pred_words
