import numpy as np
from nltk.stem import SnowballStemmer
from src.tokenizer import Tokenizer

''' Class that enables to compute the bag of words of a text, 
    with eventually deletion of stop-words, and stemming. It
    depends on the booleans is_stop_words and is_stemming'''

class Preprocessor:

    def __init__(self, separator, stop_list, is_stop_words = True, is_stemming = True):
        self.separator = separator
        self.stop_list = stop_list
        self.is_stop_words = is_stop_words
        self.is_stemming = is_stemming

    @staticmethod
    def tokenization(separator, s):
        return Tokenizer(separator).tokenize(s)

    @staticmethod
    # s : list of tokens
    def remove_stop_words(stop_list, s):
        for token in s:
            if token in stop_list:
                while token in s:
                    s.remove(token)
        return s

    @staticmethod
    #s : list of tokens
    def stemming(s):
        stemmer = SnowballStemmer('english')
        for i, token in zip(range(len(s)), s):
            new_token = stemmer.stem(token)
            s[i] = new_token
        return s

    def get_bow(self, s):
        if self.is_stop_words:
            if self.is_stemming:
                s = Preprocessor.stemming(Preprocessor.remove_stop_words(self.stop_list, Preprocessor.tokenization(self.separator, s)))
            else:
                s = Preprocessor.remove_stop_words(self.stop_list, Preprocessor.tokenization(self.separator, s))
        else:
            if self.is_stemming:
                s = Preprocessor.stemming(Preprocessor.tokenization(self.separator, s))
            else:
                s = Preprocessor.tokenization(self.separator, s)
        terms, freqs = np.unique(s, return_counts = True)
        bow = {}
        for freq, term in zip(freqs, terms):
            bow[term] = freq
        return bow

