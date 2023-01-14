from typing import Tuple

from torch import IntTensor, BoolTensor, masked_select
from transformers import BertTokenizerFast

import Configuration


class Predictor:
    def __init__(self, conf: Configuration):
        self.conf = conf
        self.tokenizer = BertTokenizerFast.from_pretrained(conf.bert)
        self.models: dict = {}

    @staticmethod
    def prediction_mask(words_ids: list) -> list:
        """
        List of boolean values, True if the sub-word is the first one
        :param words_ids:
        :return:
        """
        mask = [False] * len(words_ids)

        pred = None
        for idx, ids in enumerate(words_ids):
            if ids != pred and ids is not None:
                mask[idx] = True
            pred = ids
        return mask

    @staticmethod
    def map_id2lab(dictionary, list_of_ids, is_tensor=False) -> list:
        """
        Mapping a list of ids into a list of labels
        """
        result = []
        for label_id in list_of_ids:
            label_id = label_id.item() if is_tensor else label_id
            result.append(dictionary[label_id] if label_id in dictionary else "O")

        return result

    @staticmethod
    def unify_labels(labelsA: list, labelsB: list) -> list:
        """
        Creates a unified list of tags, if there are more than one tag to keep it generates a sequence
        delimited by "/"
        """
        unified = []
        for a, b in zip(labelsA, labelsB):
            if a == b:
                unified.append("" if a == "O" else a)
            elif a == "O":
                unified.append(b)
            elif b == "O":
                unified.append(a)
            else:
                unified.append(a + "/" + b)
        return unified

    def add_model(self, group: str, model: NERClassifier, dictionary: dict):
        model.eval()
        self.models[group] = (model, dictionary)

    def predict(self, string: str) -> Tuple[list, list]:

        token_text = self.tokenizer(string)

        input_ids = IntTensor(token_text["input_ids"]).unsqueeze(0)
        att_mask = IntTensor(token_text["attention_mask"]).unsqueeze(0)
        tag_mask = BoolTensor(self.prediction_mask(token_text.word_ids()))

        if self.conf.cuda:
            input_ids = input_ids.to(self.conf.gpu)
            att_mask = att_mask.to(self.conf.gpu)
            tag_mask = tag_mask.to(self.conf.gpu)

        results = []
        for (model, dictionary) in self.models.values():
            logits = model(input_ids, att_mask, None)
            logits = logits[0].squeeze(0).argmax(1)
            logits = masked_select(logits, tag_mask).tolist()

            results.append(
                [lbl[2:] if lbl != "O" else "O" for lbl in self.map_id2lab(dictionary, logits)])

        results = self.unify_labels(results[0], results[1]) if len(results) == 2 else results[0]

        # Mask is used to show only a once the entity. if true on the last word in a group of words
        # where it was detected as entity
        mask = [False] * len(results)
        for idx in range(len(results) - 1):
            if results[idx] != results[idx + 1] and results[idx] != "":
                mask[idx] = True
        mask[-1] = True if results[-1] != "" else False

        return results, mask
