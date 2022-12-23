import torch

from Evaluation.metrics import eval_model
from Parser.parser_utils import Splitting
from Parser.parsers import Parser, EnsembleParser
from Training.NER_model import NERClassifier
from Training.training import train
from configuration import Configuration

if __name__ == "__main__":

    conf = Configuration(["files", "param", "bert"])
    # e_handler = Parser(conf)
    handlers, unified_dt = EnsembleParser(conf)
    df_train, df_val, df_test = Splitting().holdout(unified_dt)

    modelA = NERClassifier(conf.bert, handlers["a"].labels("num"))
    modelA.load_state_dict(torch.load("K:/NoSyncCache/Models/A/modelH1.pt"))

    modelB = NERClassifier(conf.bert, handlers["b"].labels("num"))
    modelB.load_state_dict(torch.load("K:/NoSyncCache/Models/B/modelE2.pt"))

    if conf.cuda:
        modelA = modelA.to("cuda:0")
        modelB = modelB.to("cuda:0")

    eval_model(modelA, df_test[["Sentences", "Labels_a"]], conf, handlers["a"])
    #  train(model, e_handler, df_train, df_val, conf)

# Epochs: 1  | Loss:  0.0344 | Val_Loss:  0.0260 | F1:  0.9143
# Epochs: 2  | Loss:  0.0165 | Val_Loss:  0.0227 | F1:  0.9299
# Epochs: 3  | Loss:  0.0137 | Val_Loss:  0.0183 | F1:  0.9388
# Epochs: 4  | Loss:  0.0124 | Val_Loss:  0.0184 | F1:  0.9419
# Epochs: 5  | Loss:  0.0119 | Val_Loss:  0.0188 | F1:  0.9297
# Epochs: 6  | Loss:  0.0118 | Val_Loss:  0.0183 | F1:  0.9424
# Epochs: 7  | Loss:  0.0118 | Val_Loss:  0.0190 | F1:  0.9405
# Epochs: 8  | Loss:  0.0097 | Val_Loss:  0.0168 | F1:  0.9461
# Epochs: 9  | Loss:  0.0094 | Val_Loss:  0.0167 | F1:  0.9479
# Epochs: 10 | Loss:  0.0095 | Val_Loss:  0.0166 | F1:  0.9471 saved

# Epochs: 1 | Loss:  0.0119 | Val_Loss:  0.0189 | F1:  0.9343
