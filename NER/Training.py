import torch
from pandas import DataFrame
from torch.nn.utils.rnn import pad_sequence
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from CustomDataset import NerDataset


def custom_collate(data):
    inputs_ids = [d[0] for d in data]
    mask = [d[1] for d in data]
    label = [d[2] for d in data]

    inputs_ids = pad_sequence(inputs_ids, batch_first=True)
    mask = pad_sequence(mask, batch_first=True)
    label = pad_sequence(label, batch_first=True)

    return inputs_ids, mask, label


def train(model, bert, parser, df_train: DataFrame, df_val: DataFrame, param: dict):
    # We create an iterator for training e validation dataset
    print("Creating Dataloader for Training set")
    tr = DataLoader(NerDataset(df_train, bert, parser), collate_fn=custom_collate, batch_size=param["batch_size"],
                    shuffle=True)

    print("Creating Dataloader for Validation set")
    vl = DataLoader(NerDataset(df_val, bert, parser), collate_fn=custom_collate, batch_size=1)
    tr_size, vl_size = len(tr), len(vl)

    earlyS_flag: int = 0
    epoch: int = 0
    previous_vl: float = float("inf")

    optimizer = SGD(model.parameters(), lr=param["lr"], momentum=param["momentum"], weight_decay=param["weight_decay"],
                    nesterov=param["nesterov"])

    while (epoch < param["max_epoch"]) & (earlyS_flag <= 1):

        loss_train, loss_val = 0, 0

        # ========== Training Phase ==========
        model.train()
        for input_id, mask, tr_label in tqdm(tr):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model(input_id, mask, tr_label)
            loss_train += loss.item()
            loss.backward()
            optimizer.step()
            # ========== Training Phase ==========

        if param["cache"]:
            # torch.save(model.state_dict(), folder + "tmp/" + param["model_name"])
            torch.save(model.state_dict(), "./" + param["model_name"])

        # ========== Validation Phase ==========
        model.eval()  # Validation phase
        for input_id, mask, val_label in tqdm(vl):
            loss, _ = model(input_id, mask, val_label)
            loss_val += loss.item()
            # ========== Validation Phase ==========

        tr_loss, val_loss = (loss_train / tr_size), (loss_val / vl_size)

        # Early stopping
        if param["early_stopping"]:
            if previous_vl < val_loss:
                earlyS_flag += 1
            previous_vl = val_loss

        print(f'Epochs: {epoch + 1} | Loss: {tr_loss: .3f} | Val_Loss: {val_loss: .3f}')
        epoch += 1
