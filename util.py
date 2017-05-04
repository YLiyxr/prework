import numpy as np


STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "if", "in", "into", "is", "it", "no", "of", "on", "or", "such", "that",
    "the", "their", "then", "there", "these", "they", "this", "to", "was",
    "will", "with"}


def read_data(fname):
    """Read dataset

    Args:
        file path

    Returns: list of sentences, NumPy array of labels
    """
    text, y = [], []
    with open(fname, 'rb') as f:
        for line in f:
            label, s = line.strip().split(' ', 1)
            text.append(s)
            y.append((int(label) + 1) / 2)
    return text, np.ravel(y)


def default_filter(ngram, min_len=3, stopwords=STOPWORDS):
    """A simple word length and stopword filter

    Args:
        ngram: list of words in ngram
        min_len: minimum character length for words
        stopwords: list of stopwords to filter out

    Returns: boolean
    """
    return all(((len(w) >= min_len) and (w not in stopwords)) for w in ngram)


class SymbolTable(object):

    def __init__(self, vocab):
        """SymbolTable encodes sentences by converting tokens to symbols

        Args:
            vocab: an iterable containing the words in the voabulary
        """
        self.v = {w: i for i, w in enumerate(vocab)}

    def sentence_to_vector(self, sentence):
        """Convert a sentence to an count vector for vocab symbols

        Args:
            sentence: a dict mapping ngrams to counts in sentence

        Returns: a NumPy vector of counts for each entry in vocabulary
        """
        x = np.zeros(len(self.v))
        sentence_filtered = filter(lambda x: x[0] in self.v, sentence.items())
        if len(sentence_filtered) == 0:
            return x
        keys, cts = zip(*sentence_filtered)
        x[np.ravel([self.v[k] for k in keys])] = np.ravel(cts)
        return x

    def sentences_to_matrix(self, sentences):
        """Convert a list sentences to an count matrix for vocab symbols

        Args:
            sentences: a list of dicts mapping ngrams to counts in sentences

        Returns: a NumPy matrix of counts for each entry in vocabulary
        """
        return np.vstack([self.sentence_to_vector(s) for s in sentences])
