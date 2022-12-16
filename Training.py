from pandas import DataFrame
from torch import no_grad, zeros, masked_select
from torch.nn.utils import clip_grad_norm_
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

import Configuration
from BuildDataset import NerDataset
from Evaluation.Metrics import metrics
from Utils import padding_batch, EarlyStopping, ModelVersion


def train(model, e_handler, df_train: DataFrame, df_val: DataFrame, conf: Configuration):
    # We create an iterator for training e validation dataset
    print("Creating Dataloader for Training set")
    tr = DataLoader(NerDataset(df_train, conf, e_handler), collate_fn=padding_batch,
                    batch_size=conf.param["batch_size"], shuffle=True)

    print("\nCreating Dataloader for Validation set")
    vl = DataLoader(NerDataset(df_val, conf, e_handler), collate_fn=padding_batch, batch_size=1)

    tr_size, vl_size = len(tr), len(vl)
    total_epochs, stopping = conf.param["max_epoch"], conf.param["early_stopping"]
    max_labels = e_handler.labels("num")
    epoch = 0

    es = EarlyStopping(total_epochs if stopping <= 0 else stopping)

    optimizer = SGD(model.parameters(), lr=conf.param["lr"], momentum=conf.param["momentum"],
                    weight_decay=conf.param["weight_decay"], nesterov=conf.param["nesterov"])

    model_version = ModelVersion(folder=conf.folder, name=conf.param["model_name"]) if conf.param["cache"] else None

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=tr_size * total_epochs)

    print("\nTraining")
    while (epoch < total_epochs) and (not es.earlyStop):

        loss_train, loss_val = 0, 0

        # ========== Training Phase ==========
        for inputs_ids, att_mask, _, labels in tqdm(tr):
            optimizer.zero_grad(set_to_none=True)

            loss, _ = model(inputs_ids, att_mask, labels)
            loss_train += loss.item()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        # ========== Training Phase ==========

        confusion = zeros(size=(max_labels, max_labels))
        # ========== Validation Phase ==========
        with no_grad():  # Validation phase
            for inputs_ids, att_mask, tag_maks, labels in tqdm(vl):

                loss, logits = model(inputs_ids, att_mask, labels)
                loss_val += loss.item()
                logits = logits[0].argmax(1)

                logits = masked_select(logits, tag_maks)
                labels = masked_select(labels, tag_maks)
                for lbl, pre in zip(labels, logits):
                    confusion[lbl, pre] += 1
        # ========== Validation Phase ==========

        tr_loss, val_loss = (loss_train / tr_size), (loss_val / vl_size)

        # Update the early stopping controller
        es.update(val_loss)

        print(f'Epochs: {epoch + 1} | Loss: {tr_loss: .4f} | Val_Loss: {val_loss: .4f} | F1: {metrics(confusion): .4f}')
        epoch += 1

        if model_version is not None:
            model_version.update(model, val_loss)
