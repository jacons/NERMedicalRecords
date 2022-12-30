import os
from math import inf, isnan

from torch import save
from torch.nn.utils.rnn import pad_sequence


class EarlyStopping:
    """
    The early stopping it used to avoid the over-fitting.
    """

    def __init__(self, patience: int):
        self.patience: int = patience
        self.curr_pat: int = patience + 1
        self.current_vl: float = -inf
        self.earlyStop = False

    def update(self, vl_loss: float):
        if self.current_vl < vl_loss:
            self.curr_pat -= 1
        else:
            self.curr_pat = self.patience
        self.current_vl = vl_loss
        if self.curr_pat == 0:
            self.earlyStop = True


class ModelVersion:
    """
    ModelVersion at each epoch choose if it is better save the model base on validation loss
    """

    def __init__(self, folder: str, name: str):
        self.folder = folder
        self.model_name = name
        self.list_vl_loss: list = []

        if not os.path.exists(folder + "/saved_models"):
            os.makedirs(folder + "/saved_models")

    def update(self, model, metric: float):
        """
        If it is the first epoch or the model reach the minimum validation loss, it saves the model
        otherwise it maintains the previous version.
        """
        if len(self.list_vl_loss) == 0 or (metric < min(self.list_vl_loss)):
            save(model.state_dict(), self.folder + "/saved_models/" + self.model_name + ".pt")
            print("Saved")
        if not isnan(metric):
            self.list_vl_loss.append(metric)


def padding_batch(batch: list):
    # data = list of element, length = batch_size

    inputs_ids, att_mask, tag_maks, labels = [], [], [], []

    for inputs_ids_, att_mask_, tag_maks_, labels_ in batch:
        inputs_ids.append(inputs_ids_)
        att_mask.append(att_mask_)
        tag_maks.append(tag_maks_)
        labels.append(labels_)

    inputs_ids = pad_sequence(inputs_ids, batch_first=True)
    att_mask = pad_sequence(att_mask, batch_first=True)
    tag_maks = pad_sequence(tag_maks, batch_first=True)
    labels = pad_sequence(labels, batch_first=True)

    return inputs_ids, att_mask, tag_maks, labels
