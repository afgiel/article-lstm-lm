from glove_wrapper import GloveWrapper, NUM_TOKENS
import numpy as np

glove_matrix = GloveWrapper()

def text_to_index(text_list):
    """Takes in a list of strings, returns a list of corresponding glove indices"""
    text_indices = [0]*len(text_list)
    for index,word in enumerate(text_list):
        text_indices[index] = glove_matrix.get_index(word)
    assert len(text_indices) == len(text_list)
    text_onehots = np.zeros((len(text_list), NUM_TOKENS))
    for index,text_index in enumerate(text_indices):
        text_onehots[index, text_index] = 1
    print text_onehots.shape
    return text_onehots

def text_to_vec(text_list):
    """Takes in a list of strings, returns a list of glove vectors"""
    glove_vectors = [0]*len(text_list)
    for index,word in enumerate(text_list):
        text_index = glove_matrix.get_index(word)
        glove_vectors[index] = glove_matrix.get_vec(text_index)
    assert len(glove_vectors) == len(text_list)
    return glove_vectors