# Title: vocabulary.py
# Author: Cody Kala
# Date: 6/2/2018
# ========================
# Generates the vocabulary for the decoder stage of our TranslationModel.

from collections import Counter

# Minimum count threshold to be included in vocabulary
THRESHOLD = 1   

# Filepaths to formula files
train_formulas_path = "../im2latex/preprocessed/train_formulas.lst" 
val_formulas_path = "../im2latex/preprocessed/val_formulas.lst"

# Filepath to output vocabulary file
vocab_path = "../im2latex/preprocessed/vocab.lst"

def main():
    train_vocab = get_vocab(train_formulas_path)
    val_vocab = get_vocab(val_formulas_path)
    final_vocab = combine_vocabs(train_vocab, val_vocab)
    print(final_vocab)
    save_vocab(final_vocab, vocab_path)


def get_vocab(formula_path):
    """ Parses the formulas in the file located at |filepath| to obtain
    the tokens/characters contained therein.

    Inputs:
        formula_path : string
            The filepath to the file containing the LaTeX formulas.

    Outputs:
        vocab : Counter
            A Counter containing the tokens/characters that appear
            in the file.
    """
    # Populate vocabulary
    vocab = Counter()
    with open(formula_path, "r") as f:
        for line in f:
            tokens = line.replace(" ", "")  # Remove whitespace
            for token in tokens:
                vocab[token] += 1

    return vocab


def combine_vocabs(*vocabs):
    """ Combines the vocabs from multiple formula files into a single
    vocabulary.

    Inputs:
        non-keyworded arguments:
            Should be vocab dicts (keys are tokens, values are counts)

    Outputs:
        final_vocab : dict
            The combined vocabulary dictionary for all formula files.
    """
    # Combine vocabularies
    final_vocab = Counter()
    for vocab in vocabs:
        final_vocab.update(vocab)

    # Remove infrequent tokens
    for token in final_vocab:
        if final_vocab[token] < THRESHOLD:
            del final_vocab[token]

    return final_vocab


def save_vocab(vocab, filepath):
    """ Writes the |vocab| to a file stored at |filepath|.
    
    Inputs:
        vocab : Counter
            The vocabulary for the formula files.
        filepath : string
            The location to save the vocabulary file.

    Outputs:
        None, but saves the vocabulary to disk.
    """
    with open(filepath, "w") as f:
        for token in sorted(vocab):
            f.write(token)


if __name__ == "__main__":
    main()

    

