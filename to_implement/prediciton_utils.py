"""
from torch import nn, zeros, IntTensor, BoolTensor, LongTensor, masked_select
from transformers import BertTokenizerFast

from Configuration import Configuration
from Evaluation.metrics import DictErrors
from Parsing.parser_utils import read_conll, align_tags


def map_id2lab(id2label: dict, list_of_ids, is_tensor=False) -> list:

    result = []
    for label_id in list_of_ids:
        label_id = label_id.item() if is_tensor else label_id
        result.append(id2label[label_id] if label_id in id2label else "O")

    return result


def eval_fromfile(model: nn.Module, path: str, conf: Configuration,
                  id2lab: dict, return_dict: bool = False, result="conlleval"):
    model.eval()
    true_label, pred_label = [], []  # using for conlleval
    max_labels = len(id2lab)
    confusion = zeros(size=(max_labels, max_labels))  # Confusion matrix
    tokenizer = BertTokenizerFast.from_pretrained(conf.bert)

    dict_errors = DictErrors() if return_dict else None

    for fields in read_conll(path=path):

        # tokens = ["Hi","How","are","you"], labels = ["O","I-TREAT" ...]
        tokens, labels = fields[0], fields[1]

        token_text = tokenizer(tokens, is_split_into_words=True)
        aligned_labels, tag_mask = align_tags(labels, token_text.word_ids())

        # prepare a model's inputs
        input_ids = IntTensor(token_text["input_ids"])
        att_mask = IntTensor(token_text["attention_mask"])
        tag_mask = BoolTensor(tag_mask)  # using to correct classify the tags

        # mapping the list of labels e.g. ["I-DRUG","O"] to list of id of labels e.g. ["4","7"]
        labels_ids = LongTensor(map_id2lab(id2lab, aligned_labels))

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

    if return_dict:
        return output_results, dict_errors.result()

    return output_results

"""
