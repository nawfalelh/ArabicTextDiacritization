DATASET_PATH = './nemlar.xml'

from text_processing.constants import *
from lxml import etree



class NemlarWord:
    def __init__(self, text, lemma, pos, prefix, root, pattern, suffix):
        self.text = text
        self.lemma = lemma
        self.pos = pos
        self.prefix = prefix
        self.root = root
        self.pattern = pattern
        self.suffix = suffix
        self.shape = self.prefix + self.pattern + self.suffix
        self.state = [e != '' for e in self.__dict__.values()]

    def __repr__(self):
        return '<NemlarWord ' + str(self.__dict__) + '>'

class NemlarSentence:
    def __init__(self, text, words):
        self.text = text
        self.words = words
        self.shape = ' '.join([word.shape for word in self.words])

    def __repr__(self):
        return self.text

    def __len__(self):
        return len(self.words)


class NemlarDataset:
    def __init__(self):
        self.xml_tree = etree.parse(DATASET_PATH)
        self.sentences = []
        for s in [NemlarSentence(sentence.xpath('text/text()')[0], [NemlarWord(*arabic_lexical.xpath('@*')) for arabic_lexical in sentence.xpath('annotation/ArabicLexical')]) for sentence in self.xml_tree.xpath('FILE/sentence')]:
            if any(l in s.text for l in LETTERS):
                self.sentences.append(s)
        self.text = ''.join([sentence.text for sentence in self.sentences])
        self.words = [word for sentence in self.sentences for word in sentence.words]
        self.prefixes = set([word.prefix for word in self.words])
        self.suffixes = set([word.suffix for word in self.words])
        self.patterns = set([word.pattern for word in self.words])
        self.pos = set([word.pos for word in self.words])
        self.chars = set([char for word in self.words for char in word.text])
        self.stats = {
            'nb_sentences' : len(self.sentences),
            'nb_words' : len(self.words),
            'nb_unique_words' : len(set([word.text for word in self.words])),
            'pos_distribiution' : dict(zip(self.pos, [len(set(filter(lambda w: True if w.pos == pos else False, self.words))) for pos in self.pos]))
        }



nemlar = NemlarDataset()