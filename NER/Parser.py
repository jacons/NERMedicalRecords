import itertools
import os

import numpy as np
import pandas as pd
from pandas import DataFrame


class Parser:
    def __init__(self, paths: list):
        """
        Util class used to parse the data from iob file (CoNLL format)
        :param paths: List of file to take into consideration
        """
        self.__sentences: list = []  # List of sentence and labels
        self.__unique_labels: set = set()  # set of unique labels encountered

        # For each file in a list, we parse a file formatted in ConLL
        for file_ in paths:
            sentences, unique_labels = self.__read_conll(file_)

            print("\tNumber of phrases built: ", len(sentences))
            self.__unique_labels = self.__unique_labels.union(unique_labels)
            print("\tNumber of tags detected: ", len(self.__unique_labels))

            self.__sentences += sentences

        # create a Name Entity dictionaries
        # Give a label returns id : label --> id
        self.__labels_to_ids: dict = {k: v for v, k in enumerate(sorted(self.__unique_labels))}
        # Give id returns a label : id --> label
        self.__ids_to_labels: dict = {v: k for v, k in enumerate(sorted(self.__unique_labels))}

        # We convert the two list in a unique dataframe that contains all phrases detected
        self.__df = pd.DataFrame(self.__sentences, columns=["tokens", "labels"]).drop_duplicates()

        return

    def get_sentences(self):
        return self.__df

    def __read_conll(self, path: str) -> tuple:
        with open(path, encoding="utf-8") as f:

            print("File: ", os.path.basename(path))

            sentences: list = []
            unique_labels: set = set()

            for is_divider, lines in itertools.groupby(f, self._is_divider):

                if is_divider:
                    continue
                fields = [line.split() for line in lines if not line.startswith('-DOCSTART-')]
                if len(fields) == 0:
                    continue

                sentence, label = [i[0].lower() for i in fields], [i[-1] for i in fields]
                for i in label:
                    unique_labels.add(i)
                sentences.append((" ".join(sentence), " ".join(label)))

            return sentences, unique_labels

    @staticmethod
    def _is_divider(line: str) -> bool:
        empty_line = line.strip() == ''
        if empty_line:
            return True

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
