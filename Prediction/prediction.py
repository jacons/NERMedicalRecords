from torch import IntTensor, BoolTensor, masked_select
from transformers import BertTokenizerFast

from Parser.parser_utils import EntityHandler
from Training import NERClassifier
from configuration import Configuration


class Predictor:
    def __init__(self, conf: Configuration):
        self.conf = conf
        self.tokenizer = BertTokenizerFast.from_pretrained(conf.bert)

        self.models: dict = {}

    @staticmethod
    def prediction_mask(words_ids: list) -> list:
        mask = [False] * len(words_ids)

        pred = None
        for idx, ids in enumerate(words_ids):
            if ids != pred and ids is not None:
                mask[idx] = True
            pred = ids
        return mask

    @staticmethod
    def unify_labels(labelsA: list, labelsB: list) -> list:
        # creates a list of tags, if there are more than one tag to keep, it generates a list
        unified = []
        for a, b in zip(labelsA, labelsB):
            if a == b or b == "O":
                unified.append(a)
            elif a == "O":
                unified.append(b)
            else:
                unified.append([a, b])
        return unified

    def add_model(self, group: str, model: NERClassifier, handler: EntityHandler):
        self.models[group] = (model, handler)

    def predict(self, string: str):

        token_text = self.tokenizer(string)

        input_ids = IntTensor(token_text["input_ids"]).unsqueeze(0)
        att_mask = IntTensor(token_text["attention_mask"]).unsqueeze(0)
        tag_mask = BoolTensor(self.prediction_mask(token_text.word_ids()))

        if self.conf.cuda:
            input_ids = input_ids.to("cuda:0")
            att_mask = att_mask.to("cuda:0")
            tag_mask = tag_mask.to("cuda:0")

        results = []
        for (model, handler) in self.models.values():
            logits = model(input_ids, att_mask, None)
            logits = logits[0].squeeze(0).argmax(1)
            logits = masked_select(logits, tag_mask)
            results.append([handler.id2label[i.item()] for i in logits])

        results = self.unify_labels(results[0], results[1]) if len(results) == 2 else results
        return results
