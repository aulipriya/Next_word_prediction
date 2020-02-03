import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.utils import shuffle


def read_csv_data_to_list(csv_file_path):
    data = pd.read_csv(csv_file_path)
    abstracts = list(data['patent_abstract'])
    return abstracts


def format_patent(patent):
    """Add spaces around punctuation and remove references to images/citations."""

    # Add spaces around punctuation
    patent = re.sub(r'(?<=[^\s0-9])(?=[.,;?])', r' ', patent)

    # Remove references to figures
    patent = re.sub(r'\((\d+)\)', r'', patent)

    # Remove double spaces
    patent = re.sub(r'\s\s', ' ', patent)
    return patent


def prepare_training_sequences(formatted):
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
    tokenizer.fit_on_texts(formatted)

    # Create look-up dictionaries and reverse look-ups
    word_idx = tokenizer.word_index
    idx_word = tokenizer.index_word
    num_words = len(word_idx) + 1
    word_counts = tokenizer.word_counts

    print(f"There are {num_words} unique words")

    sequences = tokenizer.texts_to_sequences(formatted)

    sequence_lengths = [len(sequence) for sequence in sequences]

    over_idx = []
    for i, l in enumerate(sequence_lengths):
        if l > 70:
            over_idx.append(i)

    new_texts = []
    new_sequences = []

    for i in over_idx:
        new_texts.append(formatted[i])
        new_sequences.append(sequences[i])

    training_sequences = []
    labels = []

    for sequence in new_sequences:
        for i in range(50, len(sequence)):
            extract = sequence[i - 50:i + 1]
            training_sequences.append(extract[:-1])
            labels.append(extract[-1])

    print(f'There are {len(training_sequences)} training sequences')

    return word_idx, idx_word, word_counts, num_words, training_sequences, labels


def prepare_training_and_validation_sets(training_sequences, labels, num_words):
    # Randomly shuffle features and labels
    features, labels = shuffle(training_sequences, labels, random_state=0)

    # Decide on number of samples for training
    train_end = int(0.7 * len(labels))

    train_features = np.array(features[:train_end])
    valid_features = np.array(features[train_end:])

    train_labels = labels[:train_end]
    valid_labels = labels[train_end:]

    # Convert to arrays
    X_train, X_valid = np.array(train_features), np.array(valid_features)

    # Using int8 for memory savings
    y_train = np.zeros((len(train_labels), num_words), dtype=np.int8)
    y_valid = np.zeros((len(valid_labels), num_words), dtype=np.int8)

    # One hot encoding of labels
    for example_index, word_index in enumerate(train_labels):
        y_train[example_index, word_index] = 1

    for example_index, word_index in enumerate(valid_labels):
        y_valid[example_index, word_index] = 1

    return X_train, y_train, X_valid, y_valid


def prepare_embedding_matrix(num_words, word_idx):
    glove_vectors = '../data/glove.6B.100d.txt'
    glove = np.loadtxt(glove_vectors, dtype='str', comments=None)
    vectors = glove[:, 1:].astype('float')
    words = glove[:, 0]
    word_lookup = {word: vector for word, vector in zip(words, vectors)}
    embedding_matrix = np.zeros((num_words, vectors.shape[1]))
    not_found = 0

    for i, word in enumerate(word_idx.keys()):
        # Look up the word embedding
        vector = word_lookup.get(word, None)

        # Record in matrix
        if vector is not None:
            embedding_matrix[i + 1, :] = vector
        else:
            not_found += 1

    print(f'There were {not_found} words without pre-trained embeddings.')
    # Normalize and convert nan to 0
    embedding_matrix = embedding_matrix / \
                       np.linalg.norm(embedding_matrix, axis=1).reshape((-1, 1))
    embedding_matrix = np.nan_to_num(embedding_matrix)

    return embedding_matrix, word_lookup

