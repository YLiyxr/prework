from collections import Counter
from util import default_filter


def words_to_ngrams(sentence, n, f):
    """Extracts all ngrams up to length n from a list of words

    Args:
        sentence: a list of words
        n: maximum ngram length
        f: a filter function for ngrams

    Length: 3-7 lines

    Returns:
        a list of all ngrams up to length n in sentence that pass filter
        ngrams should be represented as tuples of words (including unigrams)
    """
    raise NotImplementedError()


def sentence_to_ngram_counts(sentence, n=2, f=default_filter):
    """Convert a space-tokenized sentence string to ngram counts

    Args:
        sentence: a space-tokenized string
        n: maximum ngram length

    Calls:
        words_to_ngrams

    Length: 1-3 lines

    Returns:
        a Counter mapping *all-lowercase* ngrams to their counts in the sentence
        ngrams should be represented as tuples of words (including unigrams)
    """
    raise NotImplementedError()
