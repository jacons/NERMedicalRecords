import itertools

import numpy as np
from pandas import DataFrame, concat

from NER.Configuration import Configuration


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
            words, tags, entities = [], [], []
            for row in fields:
                # Sometimes there are more "words" associated to a single tag. Es "date"
                half = int(len(row) / 2) - 1
                word, pos_tag, entity = row[:half], row[half], row[-1]
                words.append("-".join(word))
                tags.append(pos_tag)
                entities.append(entity)
            yield " ".join(words), " ".join(tags), entities, True
    yield None, None, None, False


def buildDataset(type_entity: str, conf: Configuration):
    sentences, pos_tags, list_of_labels = [], [], []
    set_of_entity = set()  # set of unique entity found

    for file_name in conf.files:  # iterate all files "esami","anamesi"..

        # find the correct path to load
        id_file = conf.paths[file_name]["type"].index(type_entity)
        path_file = conf.paths[file_name]["files"][id_file]

        gen = read_conll(path_file)  # generator
        is_end = False
        while not is_end:
            values = next(gen)  # extract next sentence
            if not values[-1]:  # Termination check
                is_end = True
            else:
                sentences.append(values[0])
                pos_tags.append(values[1])
                list_of_labels.append(" ".join(values[2]))
                set_of_entity.update(values[2])

        print("|{:^41}|{:^20}|{:^20}|".format("Sentences and tags found",
                                              len(sentences), len(set_of_entity)) + "\n" + "-" * 85)

    t = {"Sentences": sentences, "PosTag": pos_tags, "Labels_" + str(type_entity): list_of_labels}
    return DataFrame(t).drop_duplicates(), set_of_entity


class EntityHandler:
    def __init__(self, name: str, set_entities: set):
        self.name = name
        self.set_entities = set_entities

        # Give a label returns id : label --> id
        self.labels_to_ids: dict = {k: v for v, k in enumerate(sorted(set_entities))}
        # Give id returns a label : id --> label
        self.ids_to_labels: dict = {v: k for v, k in enumerate(sorted(set_entities))}

    def align_label(self, token: list, labels: list) -> list:
        # We can all ids in the token, and we try to associate to a label
        label_ids = [-100 if word_idx is None else self.labels_to_ids[labels[word_idx]] for word_idx in token]
        return label_ids


class Parser:

    def __init__(self, conf: Configuration, keep_separated: bool = False):
        """
        We can use this class in 2 way, if we want to keep separated the tags, keep_separated must be true
        otherwise False. If keep_separated is true will be created a list of EntityHandler one foreach tag.
        If keep_separated is false there are only one instance of EntityHandler that contains all sentences and entities

        Use case:
            keep_separated = False if we train a single and multiple tag and evaluate multiple tag
            keep_separated = True if evaluate in esembling mode multiple models
        :param conf:
        :param keep_separated:
        """
        print("Building sentences\n" + "-" * 85)

        datasets = []

        if keep_separated:

            self.entity_handler = []
            for type_entity in conf.type_of_entity:  # iterate all type of entity es a, b or both
                dt, labels = buildDataset(type_entity, conf)
                datasets.append(dt)
                self.entity_handler.append(EntityHandler(type_entity, labels))
        else:

            set_entities = set()
            for type_entity in conf.type_of_entity:  # iterate all type of entity es a, b or both
                dt, labels = buildDataset(type_entity, conf)
                datasets.append(dt)
                set_entities.update(labels)

            self.entity_handler = EntityHandler("", set_entities)

        datasets = concat(datasets, axis=1)
        datasets = datasets.loc[:, ~datasets.columns.duplicated()].drop_duplicates()
        self.dt = datasets

    def align(self, token: list, labels: list):
        return self.entity_handler.align_label(token, labels)

    def get_sentences(self):
        return self.dt

    def labels(self, typ: str):
        if typ == "num":
            return len(self.entity_handler.set_entities)
        elif typ == "dict":
            return self.entity_handler.ids_to_labels


class Splitting:
    def __init__(self):
        # ========== PARAMETERS ==========
        self.tr_size: float = 0.8  # Training set dimensions
        self.vl_size: float = 0.1  # Validation set dimensions
        self.ts_size: float = 0.1  # Test set dimensions
        # ========== PARAMETERS ==========

    def holdout(self, df: DataFrame, size: float = 0.5) -> DataFrame:
        """
        Dividing the final dataset base on holdout technique
        """
        # Apply a subsampling to reduce the dimension of dataset
        df = df.sample(frac=size, random_state=42)

        length = len(df)  # length of sub-sampled dataframe
        tr = int(self.tr_size * length)  # Number of row for the training set
        vl = int(self.vl_size * length)
        ts = int(self.ts_size * length)

        print("|{:^27}|{:^27}|{:^27}|".format("TR: " + str(tr), "VL: " + str(vl), "TS: " + str(ts)) + "\n" + "-" * 85)
        return np.split(df, [tr, int((self.tr_size + self.vl_size) * length)])
