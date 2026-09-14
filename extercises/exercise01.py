"""Starter code for Assignment 1, Part 1."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD

START_TOKEN = "<START>"
END_TOKEN = "<END>"


def _validate_corpus(corpus: list[list[str]]) -> None:
    """
    perform a few basic checks on the corpus.
    raises a TypeError if the corpus is not a list of lists of strings.
    prints the number of documents and words in the corpus if it passes the checks.
    """
    if not isinstance(corpus, list):
        raise TypeError("corpus must be a sequence of documents")

    for document in corpus:
        if not isinstance(document, list):
            raise TypeError("each document must be a list")
        for word in document:
            if not isinstance(word, str):
                raise TypeError("each word in the document must be a string")

    print(f"Corpus has {len(corpus)} documents and {sum(len(doc) for doc in corpus)} words. Good to go!")

def distinct_words(corpus: list[list[str]]) -> tuple[list[str], int]:
    """
    find all distinct words in the corpus.
    returns a tuple of (list of distinct words, number of distinct words).
    """
    _validate_corpus(corpus)

    corpus_words = []
    n_corpus_words = -1

    for document in corpus:
        for word in document:
            if word not in corpus_words:
                corpus_words.append(word)
    n_corpus_words = len(corpus_words)

    return corpus_words, n_corpus_words

def compute_co_occurrence_matrix(
    corpus: list[list[str]], window_size: int = 4
) -> tuple[np.ndarray, dict[str, int]]:
    """
    build a word co-occurrence matrix for the given window size.
    returns a tuple of (co-occurrence matrix, word2ind dictionary).
    """
    _validate_corpus(corpus)
    if not isinstance(window_size, int):
        raise TypeError("window_size must be an integer")
    if window_size < 1:
        raise ValueError("window_size must be at least 1")

    M = None
    word2ind = {}

    distinct_words_list, n_distinct_words = distinct_words(corpus)
    M = np.zeros((n_distinct_words, n_distinct_words))

    for word in distinct_words_list:
        word2ind[word] = len(word2ind)

    for document in corpus:
        for i, word in enumerate(document):
            if word not in word2ind:
                print(f"Warning: word '{word}' not found in word2ind. Skipping.")
                continue
            word_index = word2ind[word]
            start = max(0, i - window_size)
            end = min(len(document), i + window_size + 1)
            for j in range(start, end):
                if j != i:
                    neighbor_word = document[j]
                    if neighbor_word in word2ind:
                        neighbor_index = word2ind[neighbor_word]
                        M[word_index, neighbor_index] += 1

    return M, word2ind


def reduce_to_k_dim(M: np.ndarray, k: int = 2) -> np.ndarray:
    """Reduce the matrix to k dimensions using TruncatedSVD."""
    if not isinstance(M, np.ndarray):
        raise TypeError("M must be a NumPy array")
    if len(M.shape) != 2:
        raise ValueError("M must be a two-dimensional matrix")
    if M.shape[0] != M.shape[1]:
        raise ValueError("M must be square")
    if not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k < 1 or k > M.shape[1]:
        raise ValueError("k must be between 1 and the number of columns")

    # TODO: use TruncatedSVD(n_components=k, n_iter=10).
    raise NotImplementedError("Implement reduce_to_k_dim")


def plot_embeddings(
    M_reduced: np.ndarray, word2ind: dict[str, int], words: list[str]
) -> None:
    """Plot the requested words and label each point."""
    if not isinstance(M_reduced, np.ndarray):
        raise TypeError("M_reduced must be a NumPy array")
    if len(M_reduced.shape) != 2 or M_reduced.shape[1] != 2:
        raise ValueError("M_reduced must have shape (number_of_words, 2)")

    # TODO: scatter the selected rows and add each word as a label.
    raise NotImplementedError("Implement plot_embeddings")


if __name__ == "__main__":
    corpus = [["hello", "world"], ["goodbye", "world"]]
    corpus_words, n_corpus_words = distinct_words(corpus)
    print(f"Distinct words: {corpus_words}")
    print(f"Number of distinct words: {n_corpus_words}")

    M, word2ind = compute_co_occurrence_matrix(corpus, window_size=1)
    print("Co-occurrence matrix:")
    print(M)
    print(f"Word to index mapping: {word2ind}")