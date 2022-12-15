from math import inf

from torch import save
from torch.nn.utils.rnn import pad_sequence


class EarlyStopping:
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
    def __init__(self, folder: str, name: str):
        self.folder = folder
        self.model_name = name
        self.list_vl_loss: list = []

    def update(self, model, vl_loss: float):
        if (len(self.list_vl_loss) == 0) or (vl_loss < min(self.list_vl_loss)):
            save(model.state_dict(), self.folder + "tmp/" + self.model_name + ".pt")
            print("Saved")
        self.list_vl_loss.append(vl_loss)


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
