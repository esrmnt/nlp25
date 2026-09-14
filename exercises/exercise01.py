"""Assignment 1, Part 1."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD

START_TOKEN = "<START>"
END_TOKEN = "<END>"


def validate_corpus(corpus: list[list[str]]) -> bool:
    """
    perform a few basic checks on the corpus.
    raises a TypeError if the corpus is not a list of lists of strings.
    prints the number of documents and words in the corpus if it passes the checks.
    """
    if len(corpus) == 0:
        print("error: corpus must not be empty")
        return False
    if not isinstance(corpus, list):
        print("error: corpus must be a sequence of documents")
        return False

    for document in corpus:
        if not isinstance(document, list):
            print("error: each document must be a list")
            return False
        for word in document:
            if not isinstance(word, str):
                print("error: each word in the document must be a string")
                return False

    print(f"corpus has {len(corpus)} documents and {sum(len(doc) for doc in corpus)} words. Good to go!")
    return True

def distinct_words(corpus: list[list[str]]) -> tuple[list[str], int]:
    """
    find all distinct words in the corpus.
    returns a tuple of (list of distinct words, number of distinct words).
    """
    corpus_words = []
    n_corpus_words = -1

    for document in corpus:
        for word in document:
            if word not in corpus_words:
                corpus_words.append(word)
    corpus_words.sort()
    n_corpus_words = len(corpus_words)

    return corpus_words, n_corpus_words

def compute_co_occurrence_matrix(
    corpus: list[list[str]], window_size: int = 4
) -> tuple[np.ndarray, dict[str, int]]:
    """
    build a word co-occurrence matrix for the given window size.
    returns a tuple of (co-occurrence matrix, word2ind dictionary).
    """
    if not validate_corpus(corpus):
        return None, None
    if isinstance(window_size, bool) or not isinstance(window_size, int):
        print("error: window_size must be an integer")
        return None, None
    if window_size < 1:
        print("error: window_size must be at least 1")
        return None, None

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
    """
    reduces the matrix to k dimensions using TruncatedSVD.
    returns the reduced matrix of shape (number_of_words, k).
    """
    if M is None or not isinstance(M, np.ndarray):
        print("error: M must be a NumPy array")
        return None
    if len(M.shape) != 2:
        raise ValueError("M must be a two-dimensional matrix")
    if M.shape[0] != M.shape[1]:
        raise ValueError("M must be square")
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k < 1 or k > M.shape[1]:
        raise ValueError("k must be between 1 and the number of columns")
    if M.shape[0] == 0:
        print("error: M must not be empty")
        return None
  
    svd = TruncatedSVD(n_components=k, n_iter=10)
    M_reduced = svd.fit_transform(M)
    return M_reduced


def plot_embeddings(
    M_reduced: np.ndarray, word2ind: dict[str, int], words: list[str]
) -> None:
    """plot the requested words and label each point."""
    if M_reduced is None or not isinstance(M_reduced, np.ndarray):
        print("error: M_reduced must be a NumPy array")
        return None
    if len(M_reduced.shape) != 2 or M_reduced.shape[1] != 2:
        raise ValueError("M_reduced must have shape (number_of_words, 2)")
    if not isinstance(word2ind, dict):
        raise TypeError("word2ind must be a dictionary")
    if not isinstance(words, list):
        raise TypeError("words must be a list")

    plt.figure(figsize=(10, 10))

    for word in words:
        if word not in word2ind:
            print(f"error: '{word}' is not in word2ind")
            return None
        index = word2ind[word]
        if not isinstance(index, int):
            print(f"error: invalid index for '{word}'")
            return None
        if index < 0 or index >= M_reduced.shape[0]:
            print(f"error: invalid index for '{word}'")
            return None
        x = M_reduced[index, 0]
        y = M_reduced[index, 1]
        plt.scatter(x, y)
        plt.annotate(word, (x, y))

    plt.title("Word Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    corpus = [["hello", "world"], ["goodbye", "world"]]

    if not validate_corpus(corpus=corpus):
        print("error: corpus validation failed. Exiting.")
        exit()

    corpus_words, n_corpus_words = distinct_words(corpus=corpus)
    print(f"Number of distinct words: {n_corpus_words}")

    M, word2ind = compute_co_occurrence_matrix(corpus=corpus, window_size=1)
    if M is None or word2ind is None:
        print("error: could not create the co-occurrence matrix. Exiting.")
        exit()
    print("Co-occurrence matrix:")
    print(M)
    print(f"Word to index mapping: {word2ind}")

    M_reduced = reduce_to_k_dim(M, k=2)
    if M_reduced is None:
        print("error: could not reduce the matrix. Exiting.")
        exit()
    print("Reduced matrix:")
    print(M_reduced)

    words_to_plot = ["hello", "world", "goodbye"]
    plot_embeddings(M_reduced, word2ind, words_to_plot)