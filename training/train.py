from keras.optimizers import Adam
from keras.utils import plot_model
from models.build_model import make_word_level_model
from data_preparation.data_preprocessing import *
from keras.callbacks import EarlyStopping, ModelCheckpoint


def make_callbacks(model_name, save=True):
    """Make list of callbacks for training"""
    callbacks = [EarlyStopping(monitor='val_loss', patience=5)]

    if save:
        callbacks.append(
            ModelCheckpoint(
                f'{model_name}.h5',
                save_best_only=True,
                save_weights_only=False))
    return callbacks


def main():
    abstracts = read_csv_data_to_list('../data/neural_network.csv')
    formatted = []

    for abstract in abstracts:
        formatted.append(format_patent(abstract))

    word_idx, idx_word, word_counts, num_words, training_sequences, labels = prepare_training_sequences(formatted)
    X_train, y_train, X_valid, y_valid = prepare_training_and_validation_sets(training_sequences, labels, num_words)
    embedding_matrix, word_lookup = prepare_embedding_matrix(num_words, word_idx)

    model = make_word_level_model(
        num_words,
        embedding_matrix=embedding_matrix,
        lstm_cells=64,
        trainable=False,
        lstm_layers=1)
    model.summary()
    BATCH_SIZE = 2048
    callbacks = make_callbacks('pre-trained-rnn')

    history = model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=callbacks,
        validation_data=(X_valid, y_valid))


if __name__ == "__main__":
    main()

