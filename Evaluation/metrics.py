import sys
from collections import defaultdict
from io import StringIO

import torch
from pandas import DataFrame
from torch import Tensor, zeros, IntTensor, BoolTensor, LongTensor, masked_select, nn
from tqdm import tqdm
from transformers import BertTokenizerFast

import Configuration
from Evaluation.conlleval import evaluate
from Parsing.parser_utils import EntityHandler, align_tags


def scores(confusion: Tensor, all_metrics=False):
    """
    Given a Confusion matrix, returns an F1-score, if all_metrics is false, then returns only a mean of F1-score
    """
    length = confusion.shape[0]
    iter_label = range(length)

    accuracy: Tensor = zeros(length)
    precision: Tensor = zeros(length)
    recall: Tensor = zeros(length)
    f1: Tensor = zeros(length)

    for i in iter_label:
        fn = torch.sum(confusion[i, :i]) + torch.sum(confusion[i, i + 1:])  # false negative
        fp = torch.sum(confusion[:i, i]) + torch.sum(confusion[i + 1:, i])  # false positive
        tn, tp = 0, confusion[i, i]  # true negative, true positive

        for x in iter_label:
            for y in iter_label:
                if (x != i) & (y != i):
                    tn += confusion[x, y]

        accuracy[i] = (tp + tn) / (tp + fn + fp + tn)
        precision[i] = tp / (tp + fp)
        recall[i] = tp / (tp + fn)
        f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

    if all_metrics:
        return DataFrame({
            "Accuracy": accuracy.tolist(),
            "Precision": precision.tolist(),
            "Recall": recall.tolist(),
            "F1": f1.tolist()})
    else:
        return f1.mean()


class DictErrors:
    def __init__(self):
        self.e_dict = defaultdict(int)

    def add(self, lst_tokens, lst_pred: Tensor, lst_labels: Tensor):

        for token, pred, lab in zip(lst_tokens, lst_pred, lst_labels):
            if pred != lab:
                self.e_dict[token] += 1

    def result(self) -> dict:
        return dict(sorted(self.e_dict.items(), key=lambda item: item[1], reverse=True))


def eval_model(model: nn.Module, dataset: DataFrame, conf: Configuration,
               handler: EntityHandler, return_dict: bool = False, result="conlleval"):
    model.eval()
    true_label, pred_label = [], []  # using for conlleval
    max_labels = len(handler.set_entities)
    confusion = zeros(size=(max_labels, max_labels))  # Confusion matrix
    tokenizer = BertTokenizerFast.from_pretrained(conf.bert)

    dict_errors = DictErrors() if return_dict else None

    for row in tqdm(dataset.itertuples(), total=dataset.shape[0], desc="Evaluating", mininterval=conf.refresh_rate):

        # tokens = ["Hi","How","are","you"], labels = ["O","I-TREAT" ...]
        tokens, labels = row[1].split(), row[2].split()

        token_text = tokenizer(tokens, is_split_into_words=True)
        aligned_labels, tag_mask = align_tags(labels, token_text.word_ids())

        # prepare a model's inputs
        input_ids = IntTensor(token_text["input_ids"])
        att_mask = IntTensor(token_text["attention_mask"])
        tag_mask = BoolTensor(tag_mask)  # using to correct classify the tags

        # mapping the list of labels e.g. ["I-DRUG","O"] to list of id of labels e.g. ["4","7"]
        labels_ids = LongTensor(handler.map_lab2id(aligned_labels))

        if conf.cuda:
            input_ids = input_ids.to(conf.gpu).unsqueeze(0)
            att_mask = att_mask.to(conf.gpu).unsqueeze(0)
            tag_mask = tag_mask.to(conf.gpu)
            labels_ids = labels_ids.to(conf.gpu)

        # Perform the prediction
        path, _ = model(input_ids, att_mask, None)[0][0]  # path is a list of int
        path = LongTensor(path)

        if conf.cuda:
            path = path.to(conf.gpu)

        logits = masked_select(path, tag_mask)
        labels = masked_select(labels_ids, tag_mask)

        # before mapping id -> labels , we have to build a confusion matrix
        for lbl, pre in zip(labels, logits):
            confusion[lbl, pre] += 1

        if dict_errors is not None:
            dict_errors.add(tokens, logits, labels)

        labels = handler.map_id2lab(labels, is_tensor=True)
        logits = handler.map_id2lab(logits, is_tensor=True)

        true_label.extend(labels)
        pred_label.extend(logits)

    if result == "conlleval":

        old_stdout = sys.stdout
        sys.stdout = output_results = StringIO()

        # ConLL script evaluation https://github.com/sighsmile/conlleval
        evaluate(true_label, pred_label)
        sys.stdout = old_stdout

    else:
        output_results = scores(confusion, all_metrics=True)
        output_results.index = handler.map_id2lab([*range(0, max_labels)])

    return output_results, dict_errors.result() if return_dict else None
