import cPickle
import sys
import numpy as np

from logistic_regression import LogisticRegression, TFLogisticRegression
from features import sentence_to_ngram_counts
from util import read_data, SymbolTable


def load_data(fname, symbol_table):
    """Load data from file and convert text to data matrix

    Args:
        fname: file path
        symbol_table: SymbolTable object for converting text to symbols

    Returns: tuple of 2D NumPy array (data matrix) and NumPy vector (labels)
    """
    text, y = read_data(fname)
    data = map(sentence_to_ngram_counts, text)
    X = symbol_table.sentences_to_matrix(data)
    return X, y


if __name__ == '__main__':
    # Create symbol table
    with open('data/vocab.pkl', 'rb') as f:
        vocab = cPickle.load(f)
    symbol_table = SymbolTable(vocab)
    # Process train data
    X_train, y_train = load_data('data/train.txt', symbol_table)
    # Train model
    if len(sys.argv) > 1 and sys.argv[1] == 'tf':
        model = TFLogisticRegression(X_train.shape[1]) 
    else:
        model = LogisticRegression(X_train.shape[1])
    model.train(X_train, y_train, verbose=True)
    train_acc = model.accuracy(X_train, y_train)
    print "Training accuracy: {0:.2f}%".format(100. * train_acc)
    # Get test data
    X_test, y_test = load_data('data/test.txt', symbol_table)
    # Get test accuracy
    test_acc = model.accuracy(X_test, y_test)
    print "Test accuracy: {0:.2f}%".format(100. * test_acc)
