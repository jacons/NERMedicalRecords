from torch import IntTensor, BoolTensor, masked_select
from transformers import BertTokenizerFast

from Usage import Configuration
from Training import NERClassifier


class Predictor:
    def __init__(self, conf: Configuration):
        self.conf = conf
        self.tokenizer = BertTokenizerFast.from_pretrained(conf.bert)
        self.models: dict = {}

    @staticmethod
    def prediction_mask(words_ids: list) -> list:
        """
        List of boolean values, True if the sub-token is the first
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
        # Creates a list of tags, if there are more than one tag to keep it generates a list
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

    def predict(self, string: str) -> list:

        token_text = self.tokenizer(string)

        input_ids = IntTensor(token_text["input_ids"]).unsqueeze(0)
        att_mask = IntTensor(token_text["attention_mask"]).unsqueeze(0)
        tag_mask = BoolTensor(self.prediction_mask(token_text.word_ids()))

        if self.conf.cuda:
            input_ids = input_ids.to("cuda:0")
            att_mask = att_mask.to("cuda:0")
            tag_mask = tag_mask.to("cuda:0")

        results = []
        for (model, dictionary) in self.models.values():
            logits = model(input_ids, att_mask, None)
            logits = logits[0].squeeze(0).argmax(1)
            logits = masked_select(logits, tag_mask).tolist()

            results.append(
                [lbl[2:] if lbl != "O" else "O" for lbl in self.map_id2lab(dictionary, logits)])

        results = self.unify_labels(results[0], results[1]) if len(results) == 2 else results[0]
        return results
