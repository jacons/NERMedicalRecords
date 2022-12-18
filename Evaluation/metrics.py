import torch
from pandas import DataFrame
from torch import Tensor, zeros, no_grad, masked_select
from torch.utils.data import DataLoader
from tqdm import tqdm

import configuration
from Parser.parser_utils import EntityHandler
from Training import NERClassifier
from Parser.NERDataset import NerDataset
from Training.training_utils import padding_batch


def scores(confusion: Tensor, all_metrics=False):
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


def single_eval(model: NERClassifier, dataset: DataFrame, conf: configuration.Configuration, handler: EntityHandler) -> DataFrame:
    # evaluate the dataframe with a single model

    ts = DataLoader(NerDataset(dataset, conf, handler), collate_fn=padding_batch)

    n_labs = handler.labels("num")  # num of tags
    # they represent a dictionary that map the ids(int) into entity (str)

    # instantiate a confusion matrix
    confusion = zeros(size=(n_labs, n_labs))

    with no_grad():  # Validation phase
        for inputs_ids, att_mask, tag_maks, labels in tqdm(ts):

            logits = model(inputs_ids, att_mask, None)
            logits = logits[0].squeeze(0).argmax(1)

            logits = masked_select(logits, tag_maks)
            labels = masked_select(labels, tag_maks)
            for lbl, pre in zip(labels, logits):
                confusion[lbl, pre] += 1

    df_result = scores(confusion, all_metrics=True)
    df_result.index = [handler.id2label[i] for i in range(0, n_labs)]

    return df_result
