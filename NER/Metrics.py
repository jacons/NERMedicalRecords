"""
import torch
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from CustomDataset import NerDataset
from Parser import Parser


class Metrics:
    def __init__(self, model, bert, parser: Parser, df_test: DataFrame):
        self.model = model

        print("Creating a Dataloader for Test set")
        self.ts = DataLoader(NerDataset(df_test, bert, parser), batch_size=1)
        self.n_labels = parser.labels("num")

    def custom_metrics(self):

        ts_loss: float = 0
        accuracy: Tensor = torch.zeros(self.n_labels)
        precision: Tensor = torch.zeros(self.n_labels)
        recall: Tensor = torch.zeros(self.n_labels)
        f1: Tensor = torch.zeros(self.n_labels)

        matrix_results = torch.zeros(size=(self.n_labels, self.n_labels))

        for input_id, mask, ts_label in tqdm(self.ts):

            loss, logits = self.model(input_id, mask, ts_label)
            label_clean = ts_label[0][ts_label[0] != -100]
            predictions = logits[0][ts_label[0] != -100].argmax(dim=1)

            ts_loss += loss.item()
            for lbl, pre in zip(label_clean, predictions):
                matrix_results[lbl, pre] += 1

        ts_loss /= len(self.ts)

        iter_label = range(self.n_labels)

        for i in iter_label:
            fn = torch.sum(matrix_results[i, :i]) + torch.sum(matrix_results[i, i + 1:])
            fp = torch.sum(matrix_results[:i, i]) + torch.sum(matrix_results[i + 1:, i])
            tn, tp = 0, matrix_results[i, i]

            for x in iter_label:
                for y in iter_label:
                    if (x != i) & (y != i):
                        tn += matrix_results[x, y]

            accuracy[i] = (tp + tn) / (tp + fn + fp + tn)
            precision[i] = tp / (tp + fp)
            recall[i] = tp / (tp + fn)

            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 score": f1}
"""