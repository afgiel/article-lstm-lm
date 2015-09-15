from glove_wrapper import GloveWrapper, NUM_TOKENS
import numpy as np

glove_matrix = GloveWrapper()

def text_to_glove(text):
    """Takes in a string, returns a list of corresponding glove indices"""
    text_list = text.split()
    text_indices = []
    for word in text_list:
        index = glove_matrix.get_index(word)
        text_indices = text_indices + [index]
    print text_indices
    assert len(text_indices) == len(text_list)
    text_vectors = np.eye(NUM_TOKENS)[text_indices]
    return text_vectors