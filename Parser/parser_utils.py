import itertools
from typing import Tuple

import numpy as np
from pandas import DataFrame

import Configuration


def read_conll(path: str):
    """
    Generator of sentences from CoNLL files
    :param path: path of file
    :return: (sentence string,sequence of pos tag,sequence of label,check condition)
    """

    def _is_divider(line: str) -> bool:
        return True if line.strip() == '' else False

    with open(path, encoding="utf-8") as f:
        for is_divider, lines in itertools.groupby(f, _is_divider):
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
    aligned_labels = []
    maks = [False] * len(word_ids)
    prev_id = None

    for idx, word_id in enumerate(word_ids):

        if word_id is None:
            aligned_labels.append("O")

        elif word_id != prev_id:
            aligned_labels.append(labels[word_id])
            maks[idx] = True

        elif word_id == prev_id:
            if labels[word_id][0] == "B":
                aligned_labels.append("I" + labels[word_id][1:])
            else:
                aligned_labels.append(labels[word_id])

        prev_id = word_id
    return aligned_labels, maks


def buildDataset(type_entity: str, conf: Configuration):
    """
    buildDataset function take as input the type of entity (es "a") and creates a dataframe
    where there are the sentences and labels associated to mentioned type
    :param type_entity: name of group of entity
    :param conf: configuration class
    :return: sentences and labels dataframe
    """
    sentences, list_of_labels = [], []
    set_of_entity = set()  # set of unique entity found (incrementally updated)

    for file_name in conf.files:  # iterate all files "esami","anamesi"..

        # find the correct path to load
        id_file = conf.paths[file_name]["type"].index(type_entity)
        path_file = conf.paths[file_name]["files"][id_file]

        for field in read_conll(path_file):  # generator

            tokens, labels = field[0], field[1]

            sentences.append(" ".join(tokens))
            list_of_labels.append(" ".join(labels))
            set_of_entity.update(labels)

        print("|{:^41}|{:^20}|{:^20}|".format("Sentences and tags", len(sentences), len(set_of_entity)))
        print("-" * 85)

    t = {"Sentences": sentences, "Labels_" + str(type_entity): list_of_labels}
    return DataFrame(t).drop_duplicates(), set_of_entity


class EntityHandler:
    def __init__(self, name: str, dt: DataFrame, set_entities: set):
        self.name = name  # name of group of files
        self.dt = dt  # dataframe of sentences
        self.set_entities = set_entities  # set of all entity detected

        # Give a label returns id : label --> id
        self.label2id: dict = {k: v for v, k in enumerate(sorted(set_entities))}
        # Give id returns a label : id --> label
        self.id2label: dict = {v: k for v, k in enumerate(sorted(set_entities))}

    def get_sentences(self):
        return self.dt

    def labels(self, typ: str):
        if typ == "set":
            return self.set_entities
        if typ == "num":
            return len(self.set_entities)
        elif typ == "dict":
            return self.id2label


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
        # Apply a subsampling to reduce the dimension of dataset
        df = df.sample(frac=size, random_state=42)

        length = len(df)  # length of sub-sampled dataframe
        tr = int(self.tr_size * length)  # Number of row for the training set
        vl = int(self.vl_size * length)
        ts = int(self.ts_size * length)

        print("|{:^27}|{:^27}|{:^27}|".format("TR: " + str(tr), "VL: " + str(vl),
                                              "TS: " + str(ts)) + "\n" + "-" * 85)
        return np.split(df, [tr, int((self.tr_size + self.vl_size) * length)])
