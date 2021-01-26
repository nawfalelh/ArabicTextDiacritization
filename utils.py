from text_processing.constants import *
from text_processing import diacritize, diacritic_char_split, remove_harakat
import numpy as np

chars = list(LETTERS) + ['<NON_AR>']
oh_chars = np.eye(len(chars))

tashkeel = list(TASHKEEL + (SHADDA, ''))
oh_tashkeel = np.eye(len(tashkeel))


def prepare_text(text, vector_max_length):
    letters, diacritics = diacritic_char_split(text)
    X, y = [], []
    i = 0
    while True:
        j = i + vector_max_length
        if j < len(letters):
            while letters[j-1] not in PUNCTUATION + (' ',):
                j -= 1
            inputs = np.array([char_to_oh_vector(c) for c in letters[i:j] + ['<NON_AR>']*(vector_max_length - (j - i))])
            targets = np.array([diacritic_to_oh_vector(d) for d in diacritics[i:j] + ['']*(vector_max_length - (j - i))])
            X.append(inputs)
            y.append(targets)
            i = j
        else:
            inputs = np.array([char_to_oh_vector(c) for c in letters[i:] + ['<NON_AR>'] * (vector_max_length - (len(letters) - i))])
            targets = np.array([diacritic_to_oh_vector(d) for d in diacritics[i:] + [''] * (vector_max_length - (len(letters) - i))])
            X.append(inputs)
            y.append(targets)
            break
    return X, y


def char_to_oh_vector(char):
    return oh_chars[chars.index(char)] if char in chars else oh_chars[-1]

def oh_vector_to_char(oh_vector):
    return chars[np.where(np.all(oh_vector == oh_chars, axis = 1))[0][0]]

def diacritic_to_oh_vector(diacritic):
    return oh_tashkeel[tashkeel.index(diacritic)]

def oh_vector_to_diacritic(oh_vector):
    return tashkeel[np.where(np.all(oh_vector == oh_tashkeel, axis = 1))[0][0]]

def predict_and_diacritize(model, text):
    text = remove_harakat(text)
    text_oh_vectors = np.array([[char_to_oh_vector(char) for char in text]])

    probs = model.predict(text_oh_vectors)
    idxs = probs.argmax(axis = 2)

    diacritics_oh_vectors = np.zeros((len(text), len(tashkeel)))
    diacritics_oh_vectors[np.arange(idxs.size), idxs] = 1

    diacritics = [oh_vector_to_diacritic(oh_vector) for oh_vector in diacritics_oh_vectors]

    return diacritize(text, diacritics)



