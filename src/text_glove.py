from glove_wrapper import GloveWrapper, NUM_TOKENS
import numpy as np

glove_matrix = GloveWrapper()

def text_to_index(text_list, max_len):
    """Takes in a list of strings, returns a list of corresponding glove indices"""
    seq_len = min(len(text_list), max_len)
    text_indices = [0]*seq_len
    for index in range(seq_len):
        word = text_list[index]
        text_indices[index] = glove_matrix.get_index(word)
    assert len(text_indices) == seq_len
    text_onehots = np.zeros((seq_len, NUM_TOKENS))
    for index in range(seq_len):
        text_index = text_indices[index]
        text_onehots[index, text_index] = 1.
    return text_onehots

def text_to_vec(text_list, max_len):
    """Takes in a list of strings, returns a list of glove vectors"""
    seq_len = min(len(text_list), max_len)
    glove_vectors = [0]*seq_len
    for index in range(seq_len):
        word = text_list[index]
        text_index = glove_matrix.get_index(word)
        glove_vectors[index] = glove_matrix.get_vec(text_index)
    assert len(glove_vectors) == seq_len
    return glove_vectors
