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
        self.ts = DataLoader(NerDataset(df_test, bert, parser), batch_size=1)
        self.n_labels = parser.labels("num")

        self.ts_loss: float = 0
        self.__accuracy: Tensor = torch.zeros(self.n_labels)
        self.__precision: Tensor = torch.zeros(self.n_labels)
        self.__recall: Tensor = torch.zeros(self.n_labels)
        self.__f1: Tensor = torch.zeros(self.n_labels)

    def perform(self):
        matrix_results = torch.zeros(size=(self.n_labels, self.n_labels))
        ts_loss: float = 0

        for input_id, mask, ts_label in tqdm(self.ts):

            loss, logits = self.model(input_id, mask, ts_label)
            label_clean = ts_label[0][ts_label[0] != -100]
            predictions = logits[0][ts_label[0] != -100].argmax(dim=1)

            ts_loss += loss.item()
            for lbl, pre in zip(label_clean, predictions):
                matrix_results[lbl, pre] += 1

        self.ts_loss = ts_loss / len(self.ts)

        iter_label = range(self.n_labels)

        for i in iter_label:
            fn = torch.sum(matrix_results[i, :i]) + torch.sum(matrix_results[i, i + 1:])
            fp = torch.sum(matrix_results[:i, i]) + torch.sum(matrix_results[i + 1:, i])
            tn, tp = 0, matrix_results[i, i]

            for x in iter_label:
                for y in iter_label:
                    if (x != i) & (y != i):
                        tn += matrix_results[x, y]

            self.__accuracy[i] = (tp + tn) / (tp + fn + fp + tn)
            self.__precision[i] = tp / (tp + fp)
            self.__recall[i] = tp / (tp + fn)

            self.__f1[i] = 2 * (self.__precision[i] * self.__recall[i]) / (self.__precision[i] + self.__recall[i])

    def loss(self):
        return  self.ts_loss

    def overall_result(self, typ: str) -> dict:

        if typ == "mean":
            return {
                "Accuracy mean": self.__accuracy.mean(),
                "Precision mean": self.__precision.mean(),
                "Recall mean": self.__recall.mean(),
                "F1 score mean": self.__f1.mean()
            }
        elif typ == "all":
            return {
                "Accuracy": self.__accuracy,
                "Precision": self.__precision,
                "Recall": self.__recall,
                "F1 score": self.__f1
            }

        return {}