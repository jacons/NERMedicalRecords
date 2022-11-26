from math import inf

from torch.nn.utils.rnn import pad_sequence


def padding_batch(data):
    inputs_ids = [d[0] for d in data]
    mask = [d[1] for d in data]
    label = [d[2] for d in data]

    inputs_ids = pad_sequence(inputs_ids, batch_first=True)
    mask = pad_sequence(mask, batch_first=True)
    label = pad_sequence(label, batch_first=True)

    return inputs_ids, mask, label


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience: int = patience + 1
        self.curr_pat: int = patience + 1

        self.current_vl: float = -inf
        self.earlyStop: bool = False

    def update(self, vl_loss: float):
        if self.current_vl <= vl_loss:
            self.curr_pat -= 1
        else:
            self.curr_pat = self.patience

        self.current_vl = vl_loss

        if self.curr_pat == 0:
            self.earlyStop = True
