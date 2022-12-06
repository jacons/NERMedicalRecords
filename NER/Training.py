from pandas import DataFrame
from torch import save, no_grad
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from CustomDataset import NerDataset
from NER.Configuration import Configuration
from NER.Model import BertModel
from NER.Utils import padding_batch, EarlyStopping


def train(model: BertModel, parser, df_train: DataFrame, df_val: DataFrame, conf: Configuration):
    # We create an iterator for training e validation dataset
    print("Creating Dataloader for Training set")
    tr = DataLoader(NerDataset(df_train, conf, parser), collate_fn=padding_batch, batch_size=conf.param["batch_size"],
                    shuffle=True)

    print("\nCreating Dataloader for Validation set")
    vl = DataLoader(NerDataset(df_val, conf, parser), collate_fn=padding_batch, batch_size=1)

    tr_size, vl_size = len(tr), len(vl)
    total_epochs, stopping = conf.param["max_epoch"], conf.param["early_stopping"]
    epoch = 0

    es = EarlyStopping(total_epochs if stopping <= 0 else stopping)

    optimizer = SGD(model.parameters(), lr=conf.param["lr"], momentum=conf.param["momentum"],
                    weight_decay=conf.param["weight_decay"], nesterov=conf.param["nesterov"])

    model_version = ModelVersion(folder=conf.folder, name=conf.param["model_name"]) if conf.param["cache"] else None

    print("\nTraining")
    while (epoch < total_epochs) and (not es.earlyStop):

        loss_train, loss_val = 0, 0

        # ========== Training Phase ==========
        for input_id, mask, tr_label in tqdm(tr):
            optimizer.zero_grad(set_to_none=True)

            loss, _ = model(input_id, mask, tr_label)
            loss_train += loss.item()
            loss.backward()
            optimizer.step()
        # ========== Training Phase ==========

        # ========== Validation Phase ==========
        with no_grad():  # Validation phase
            for input_id, mask, val_label in tqdm(vl):
                loss, _ = model(input_id, mask, val_label)
                loss_val += loss.item()
        # ========== Validation Phase ==========

        tr_loss, val_loss = (loss_train / tr_size), (loss_val / vl_size)

        # Update the early stopping controller
        es.update(val_loss)

        print(f'Epochs: {epoch + 1} | Loss: {tr_loss: .4f} | Val_Loss: {val_loss: .4f}')
        epoch += 1

        if model_version is not None:
            model_version.update(model, val_loss)

class ModelVersion:
    def __init__(self, folder: str, name: str):
        self.folder: str = folder
        self.model_name: str = name
        self.list_vl_loss: list = []

    def update(self, model: BertModel, vl_loss: float):
        if (len(self.list_vl_loss) == 0) or (vl_loss < min(self.list_vl_loss)):
            save(model.state_dict(), self.folder + "tmp/" + self.model_name + ".pt")
            print("Saved")

        self.list_vl_loss.append(vl_loss)
