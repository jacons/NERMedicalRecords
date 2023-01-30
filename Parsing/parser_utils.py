import argparse
import random
from itertools import groupby

import numpy as np
import pandas as pd
from pandas import DataFrame


class EntityHandler:
    """
    EntityHandler is class used to keep the associations between labels and ids
    """

    def __init__(self, dt: DataFrame, set_entities: set):
        self.dt = dt  # dataframe of sentences
        self.set_entities = set_entities  # set of all entity detected

        # Give a label returns id : label --> id
        self.label2id: dict = {k: v for v, k in enumerate(sorted(set_entities))}
        # Give id returns a label : id --> label
        self.id2label: dict = {v: k for v, k in enumerate(sorted(set_entities))}

    def map_lab2id(self, list_of_labels, is_tensor=False) -> list:
        """
        Mapping a list of labels into a list of label's id
        """
        result = []
        for label in list_of_labels:
            label = label.item() if is_tensor else label
            result.append(self.label2id[label] if label in self.label2id else self.label2id["O"])

        return result

    def map_id2lab(self, list_of_ids, is_tensor=False) -> list:
        """
        Mapping a list of ids into a list of labels
        """
        result = []
        for label_id in list_of_ids:
            label_id = label_id.item() if is_tensor else label_id
            result.append(self.id2label[label_id] if label_id in self.id2label else "O")

        return result


def holdout(df: DataFrame, size: float = 1, verbose=True) -> DataFrame:
    """
    Dividing the dataset based on holdout technique
    """
    # ========== PARAMETERS ==========
    tr_size: float = 0.8  # Training set dimensions
    vl_size: float = 0.1  # Validation set dimensions
    ts_size: float = 0.1  # Test set dimensions
    # ========== PARAMETERS ==========

    # Apply a subsampling to reduce the dimension of dataset, it also shuffles the dataset
    # we fixed the random state for the determinism
    df = df.sample(frac=size, random_state=42)

    length = len(df)  # length of sub-sampled dataframe
    tr = int(tr_size * length)  # Number of rows for the training set
    vl = int(vl_size * length)  # validation
    ts = int(ts_size * length)  # test

    if verbose:
        print("|{:^27}|{:^27}|{:^27}|".format("TR: " + str(tr), "VL: " + str(vl),
                                              "TS: " + str(ts)) + "\n" + "-" * 85)
    return np.split(df, [tr, int((tr_size + vl_size) * length)])


def read_conll(path: str):
    """
    Generator of sentences from CoNLL files
    :param path: path of file
    :return: (sentence string, sequence of label)
    """

    def _is_divider(line: str) -> bool:
        return True if line.strip() == '' else False

    with open(path, encoding="utf-8") as f:
        for is_divider, lines in groupby(f, _is_divider):
            if is_divider:
                continue
            fields = [line.split() for line in lines if not line.startswith('-DOCSTART-')]
            if len(fields) == 0:
                continue
            tokens, entities = [], []
            for row in fields:
                # Sometimes there are more "words" associated to a single tag. Es "date"
                half = int(len(row) / 2) - 1
                token, entity = row[:half], row[-1]
                tokens.append("-".join(token))
                entities.append(entity)
            yield tokens, entities


def align_tags(labels: list, word_ids: list):
    """
    This function aligns the labels associated with a sentence and returns an "aligned" list and a "tag mask".

    "aligned" list: represents a list of labels that are aligned with the word-pieces of the sentence,
                    if a token is split in more than one sub-word, the tag associated is repeated.
                    The second tag (of the sequence of sub-words) always start with "I-".

    "tag mask":     represents a list of boolean, where each value corresponding to a sub-token. The boolean
                    is true if the sub-token is the first one (of sub-words) else false.

    *WordPiece is a sub-word segmentation algorithm
    """
    aligned_labels = []
    mask = [False] * len(word_ids)
    prev_id = None

    for idx, word_id in enumerate(word_ids):

        if word_id is None:
            aligned_labels.append("O")

        elif word_id != prev_id:
            aligned_labels.append(labels[word_id])
            mask[idx] = True

        elif word_id == prev_id:
            if labels[word_id][0] == "B":
                aligned_labels.append("I" + labels[word_id][1:])
            else:
                aligned_labels.append(labels[word_id])

        prev_id = word_id
    return aligned_labels, mask


def buildDataset(path_file: str, verbose=True) -> EntityHandler:
    """
    Build the dataframe of sentences from a path of file
    The dataframe is composed by two columns "sentences" and "labels"
    """
    sentences, list_of_labels = [], []
    set_entities = set()  # set of unique entity found (incrementally updated)

    for fields in read_conll(path_file):  # generator

        tokens, labels = fields[0], fields[1]

        sentences.append(" ".join(tokens))
        list_of_labels.append(" ".join(labels))
        set_entities.update(labels)  # to keep track the entity found

    if verbose:
        print("Building sentences and tags\n" + "-" * 85)
        print("|{:^41}|{:^20}|{:^20}|".format("Sentences and tags", len(sentences), len(set_entities)))
        print("-" * 85)
        print("|{:^83}|".format(" - ".join(sorted(set_entities))))
        print("-" * 85)

    t = {"sentences": sentences, "labels": list_of_labels}
    return EntityHandler(DataFrame(t).drop_duplicates(), set_entities)


def ensembleParser(path_file_a, path_file_b, verbose=True) -> tuple[tuple[EntityHandler, EntityHandler], DataFrame]:
    """
    ensembleParser is used to group in one single dataframe the both to dataset A and B.

    The function returns: two handler (A & B) and a unified dataframe. The df is composed by 3 columns
    "sentences", "labels_a", "labels_b"

    """
    handler_a = buildDataset(path_file_a, verbose)
    handler_b = buildDataset(path_file_b, verbose)

    unified_datasets = pd.concat(
        [handler_a.dt.rename(columns={"labels": "labels_a"}),
         handler_b.dt.rename(columns={"labels": "labels_b"})], axis=1)

    unified_datasets = unified_datasets.loc[:, ~unified_datasets.columns.duplicated()].drop_duplicates()

    return (handler_a, handler_b), unified_datasets


def to_conLL(df: DataFrame, file_name: str):
    """
    util function used to create from sentences' dataframe a file in conll format
    (compatibility with MultiCoNER data sources)
    """

    def random_chars(y):
        dictionary = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
        return ''.join(random.choice(dictionary) for _ in range(y))

    f = open(file_name + ".conll", "w", encoding="utf-8")
    f.write("\n")

    for row in df.itertuples():
        f.write("# id " + random_chars(64) + " domain=it\n")

        tokens, labels = row[1].split(), row[2].split()
        if len(tokens) != len(labels):
            print("Error")
            return

        for (token, lab) in zip(tokens, labels):
            f.write(token + " _ _ " + lab + "\n")
        f.write("\n\n")
    f.close()


def parse_args():
    p = argparse.ArgumentParser(description='Model configuration.', add_help=True)

    p.add_argument('--datasets', type=str, nargs='+',
                   help='Dataset used for training, it will split in training, validation and test', default=None)

    p.add_argument('--models', type=str, nargs='+',
                   help='Model trained ready to evaluate or use, if list, the order must follow the same of datasets',
                   default=None)

    p.add_argument('--model_name', type=str,
                   help='Name to give to a trained model', default=None)

    p.add_argument('--path_model', type=str,
                   help='Directory to save the model', default=".")

    p.add_argument('--bert', type=str,
                   help='Bert model provided by Huggingface', default="dbmdz/bert-base-italian-xxl-cased")

    p.add_argument('--save', type=int,
                   help='set 1 if you want save the model otherwise set 0', default=1)

    p.add_argument('--eval', type=str,
                   help='define the type of evaluation: conlleval or df', default="conlleval")

    p.add_argument('--lr', type=float, help='Learning rate', default=0.001)

    p.add_argument('--momentum', type=float, help='Momentum', default=0.9)

    p.add_argument('--weight_decay', type=float, help='Weight decay', default=0.0002)

    p.add_argument('--batch_size', type=int, help='Batch size', default=2)

    p.add_argument('--max_epoch', type=int, help='Max number of epochs', default=20)

    p.add_argument('--patience', type=float, help='Patience in early stopping', default=3)

    return p.parse_known_args()
