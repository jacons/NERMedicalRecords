import torch
from pandas import DataFrame
from torch import Tensor, no_grad, masked_select, zeros
from torch.utils.data import DataLoader
from tqdm import tqdm
from Evaluation.conlleval import evaluate
from Parser.NERDataset import NerDataset
from Parser.parser_utils import EntityHandler
from Training.NER_model import NERClassifier
from configuration import Configuration


def scores(confusion: Tensor, all_metrics=False):
    """
    Given a Confusion matrix, returns a F1-score, if all_metrics if true then returns all metrics
    """
    length = confusion.shape[0]
    iter_label = range(length)

    accuracy: Tensor = torch.zeros(length)
    precision: Tensor = torch.zeros(length)
    recall: Tensor = torch.zeros(length)
    f1: Tensor = torch.zeros(length)

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


def eval_model(model: NERClassifier, dataset: DataFrame, conf: Configuration,
               handler: EntityHandler, result="conlleval"):

    true_label, pred_label = [], []  # using for conlleval
    max_labels = handler.labels("num")
    confusion = zeros(size=(max_labels, max_labels))

    # evaluate the dataframe with a single model
    ts = DataLoader(NerDataset(dataset, conf, handler))
    with no_grad():  # Validation phase
        for inputs_ids, att_mask, tag_maks, labels in tqdm(ts):
            logits = model(inputs_ids, att_mask, None)
            logits = logits[0].squeeze(0).argmax(1)
            logits = masked_select(logits, tag_maks)
            labels = masked_select(labels, tag_maks)

            # before mapping id -> labels
            for lbl, pre in zip(labels, logits):
                confusion[lbl, pre] += 1

            labels = handler.map_id2lab(labels, is_tensor=True)
            logits = handler.map_id2lab(logits, is_tensor=True)

            true_label.extend(labels)
            pred_label.extend(logits)

    if result == "conlleval":
        evaluate(true_label, pred_label)
        return
    elif result == "df":

        df_result = scores(confusion, all_metrics=True)
        df_result.index = handler.map_id2lab([*range(0, max_labels)])

        return df_result
