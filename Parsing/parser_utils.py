from itertools import groupby
from typing import Tuple, Any

import numpy as np
import pandas as pd
from pandas import DataFrame


class EntityHandler:
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


class Splitting:
    def __init__(self):
        # ========== PARAMETERS ==========
        self.tr_size: float = 0.8  # Training set dimensions
        self.vl_size: float = 0.1  # Validation set dimensions
        self.ts_size: float = 0.1  # Test set dimensions
        # ========== PARAMETERS ==========

    def holdout(self, df: DataFrame, size: float = 1) -> DataFrame:
        """
        Dividing the final dataset base on holdout technique
        """
        # Apply a subsampling to reduce the dimension of dataset, it also shuffles the dataset
        # we fixed the random state for the determinism
        df = df.sample(frac=size, random_state=42)

        length = len(df)  # length of sub-sampled dataframe
        tr = int(self.tr_size * length)  # Number of row for the training set
        vl = int(self.vl_size * length)
        ts = int(self.ts_size * length)

        print("|{:^27}|{:^27}|{:^27}|".format("TR: " + str(tr), "VL: " + str(vl),
                                              "TS: " + str(ts)) + "\n" + "-" * 85)
        return np.split(df, [tr, int((self.tr_size + self.vl_size) * length)])


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


def align_tags(labels: list, word_ids: list) -> Tuple[list, list]:
    """
    This function aligns the labels associated to a sentence, after that the sentence
    is broken is a word-pieces tokenization, it returns an aligned list and a tag mask.

    The tag mask is a list of boolean, each value corresponding to a sub-token. The value
    is true if the sub-token is the first of token else false.

    If a token is split in more than one sub-token, the tag associated is repeated. The second
    tag always start with "I-"
    """
    aligned_labels = []
    mask = [False] * len(word_ids)
    prev_id = None

    # in the word_ids the number is repeated is the corresponding token is split
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
    buildDataset function takes as input the type of entity (es "a") and creates a dataframe
    where there are the sentences and labels associated to mentioned type
    """
    sentences, list_of_labels = [], []
    set_entities = set()  # set of unique entity found (incrementally updated)

    for field in read_conll(path_file):  # generator

        tokens, labels = field[0], field[1]

        sentences.append(" ".join(tokens))
        list_of_labels.append(" ".join(labels))
        set_entities.update(labels)

    if verbose:
        print("Building sentences and tags\n" + "-" * 85)
        print("|{:^41}|{:^20}|{:^20}|".format("Sentences and tags", len(sentences), len(set_entities)))
        print("-" * 85)
        print("|{:^83}|".format(" - ".join(sorted(set_entities))))
        print("-" * 85)

    t = {"sentences": sentences, "labels": list_of_labels}
    return EntityHandler(DataFrame(t).drop_duplicates(), set_entities)


def ensembleParser(path_file_a, path_file_b, verbose=True) -> tuple[tuple[EntityHandler, EntityHandler], DataFrame]:
    handler_a = buildDataset(path_file_a, verbose)
    handler_b = buildDataset(path_file_b, verbose)

    unified_datasets = pd.concat(
        [handler_a.dt.rename(columns={"labels": "labels_a"}),
         handler_b.dt.rename(columns={"labels": "labels_b"})], axis=1)
    return (handler_a, handler_b), unified_datasets.loc[:, ~unified_datasets.columns.duplicated()].drop_duplicates()
