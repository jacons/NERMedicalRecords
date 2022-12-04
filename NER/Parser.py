import itertools
from os.path import basename

import numpy as np
from pandas import DataFrame, concat

from NER.Configuration import Configuration


class Parser:
    def __init__(self, conf: Configuration):
        """
        Util class used to parse the data from iob file (CoNLL format)
        """
        self.__unique_labels: set = set()  # set of unique labels encountered

        # We create a list of dataframe base on the category of files es. "esami","anamnesi" or both ..
        list_of_dt = [self.build_dataset(name, conf.paths) for name in conf.files]
        self.__df = concat(list_of_dt)

        # Create a Name Entity dictionaries
        # Give a label returns id : label --> id
        self.__labels_to_ids: dict = {k: v for v, k in enumerate(sorted(self.__unique_labels))}
        # Give id returns a label : id --> label
        self.__ids_to_labels: dict = {v: k for v, k in enumerate(sorted(self.__unique_labels))}

    def get_sentences(self):
        return self.__df

    def read_conll(self, path: str):

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
                    half = int(len(row) / 2) - 1
                    word, pos_tag, entity = row[:half], row[half], row[-1]

                    self.__unique_labels.add(entity)

                    words.append("-".join(word))
                    tags.append(pos_tag)
                    entities.append(entity)

                yield " ".join(words), " ".join(tags), " ".join(entities), True

        yield None, None, None, False

    def build_dataset(self, name: str, paths: dict):

        print("Files: ", name)
        generators, sentences, pos_tags, labels = [], [], [], []

        for file_ in paths[name]["files"]:
            generators.append(self.read_conll(file_))
            sentences.append([])
            pos_tags.append([])
            labels.append([])

        is_end = False
        while not is_end:

            for idx, gen in enumerate(generators):
                values = next(gen)

                if not values[-1]:
                    is_end = True
                    break
                sentences[idx].append(values[0])
                pos_tags[idx].append(values[1])
                labels[idx].append(values[2])

        print("\t--INFO-- Number of phrases built: ", len(sentences[0]))
        print("\t--INFO-- Number of tags detected: ", len(self.__unique_labels))

        dt = {"Sentences": sentences[0], "Pos_tag": pos_tags[0]}
        for idx, v in enumerate(paths[name]["type"]):
            dt["lbl-" + str(v)] = labels[idx]

        return DataFrame(dt).drop_duplicates()

    def align_label(self, token: list, labels: list) -> list:
        # We can all ids in the token, and we try to associate to a label
        label_ids = [-100 if word_idx is None else self.__labels_to_ids[labels[word_idx]] for word_idx in token]
        return label_ids

    def labels(self, typ: str):
        if typ == "set":
            return self.__unique_labels
        elif typ == "num":
            return len(self.__unique_labels)
        elif typ == "dict":
            return self.__ids_to_labels


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
        print("\nTotal number of phrases: ", length, " (tr): ", tr, " (vl): ", int(self.vl_size * length), " (ts): ",
              int(self.ts_size * length))
        return np.split(df, [tr, int((self.tr_size + self.vl_size) * length)])
