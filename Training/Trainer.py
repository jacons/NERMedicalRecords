from pandas import DataFrame
from torch import no_grad, zeros, masked_select
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

import Configuration
from Evaluation.metrics import scores
from Parsing.CustomDataset import NerDataset
from Parsing.parser_utils import EntityHandler
from Training.trainer_utils import padding_batch, EarlyStopping, ModelVersion


def train(model, e_handler: EntityHandler, df_train: DataFrame, df_val: DataFrame, conf: Configuration):
    # --------- DATASETS ---------
    print("--INFO--\tCreating Dataloader for Training set")
    tr = DataLoader(NerDataset(df_train, conf, e_handler), collate_fn=padding_batch,
                    batch_size=conf.param["batch_size"], shuffle=True, num_workers=4)

    print("\n--INFO--\tCreating Dataloader for Validation set")
    vl = DataLoader(NerDataset(df_val, conf, e_handler), num_workers=4)
    # --------- DATASETS ---------

    epoch = 0
    tr_size, vl_size = len(tr), len(vl)
    total_epochs = conf.param["max_epoch"]
    stopping = conf.param["early_stopping"]  # "Patience in early stopping"
    max_labels = len(e_handler.set_entities)

    # --------- Early stopping ---------
    es = EarlyStopping(total_epochs if stopping <= 0 else stopping)

    # --------- Optimizer ---------
    optimizer = SGD(model.parameters(), lr=conf.param["lr"], momentum=conf.param["momentum"],
                    weight_decay=conf.param["weight_decay"], nesterov=True)

    # --------- Save only the best model (which have minimum validation loss) ---------
    model_version = ModelVersion(folder=conf.folder, name=conf.model_name) if conf.save_model else None

    # --------- Scheduling the learning rate to improve the convergence ---------
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2)

    print("\n--INFO--\tThe Training is started")
    model.train()
    while (epoch < total_epochs) and (not es.earlyStop):

        loss_train, loss_val = 0, 0

        # ========== Training Phase ==========

        #  There inputs are created in "NerDataset" class
        for inputs_ids, att_mask, _, labels in tqdm(tr, mininterval=60):
            optimizer.zero_grad(set_to_none=True)

            loss, _ = model(inputs_ids, att_mask, labels)
            loss_train += loss.item()
            loss.backward()
            clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        # ========== Training Phase ==========

        # ========== Validation Phase ==========
        confusion = zeros(size=(max_labels, max_labels))
        with no_grad():  # Validation phase
            for inputs_ids, att_mask, tag_maks, labels in tqdm(vl, mininterval=60):

                loss, logits = model(inputs_ids, att_mask, labels)
                loss_val += loss.item()
                logits = logits[0].argmax(1)

                logits = masked_select(logits, tag_maks)
                labels = masked_select(labels, tag_maks)
                for lbl, pre in zip(labels, logits):
                    confusion[lbl, pre] += 1
        # ========== Validation Phase ==========

        tr_loss, val_loss = (loss_train / tr_size), (loss_val / vl_size)
        f1_score = scores(confusion)

        print(f'Epochs: {epoch + 1} | Loss: {tr_loss: .4f} | Val_Loss: {val_loss: .4f} | F1: {f1_score: .4f}')
        epoch += 1

        if model_version is not None:
            # save the model, if it is the best model until now
            model_version.update(model, val_loss)

        # Update the scheduler
        scheduler.step(val_loss)
        # Update the early stopping
        es.update(val_loss)
