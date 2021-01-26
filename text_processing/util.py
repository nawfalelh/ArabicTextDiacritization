from .constants import *
import regex

def text_sanitizer(text):
    return regex.sub('[^' + '^'.join(LETTERS + HARAKAT + (SHADDA, ' ',) + tuple(map(lambda x: regex.escape(x), PUNCTUATION))) + ']', '', text)

def diacritic_char_split(text):
    letters = regex.findall('[^' + '^'.join(HARAKAT + (SHADDA, )) + ']', text)
    diacritics = regex.split('[^' + '^'.join(HARAKAT + (SHADDA, )) + ']', text)
    diacritics = [d if d in TASHKEEL else '' for d in diacritics]
    return letters, diacritics[1:]

def diacritize(text, diacritics):
    if len(text) != len(diacritics):
        raise Exception('Text and diacritics not of the same length')
    else:
        return ''.join([g[0] + g[1] for g in zip(text, diacritics)])

def remove_harakat(text):
    return regex.sub('|'.join(HARAKAT + (SHADDA,)), '', text)
