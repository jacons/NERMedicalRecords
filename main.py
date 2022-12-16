import torch

from Configuration import Configuration
from Model import BertModel
from Parser.Parsers import Parser
from Parser.parser_utils import Splitting
from Training import train

if __name__ == "__main__":

    conf = Configuration(["files", "param", "bert"])
    e_handler = Parser(conf)
    df_train, df_val, df_test = Splitting().holdout(e_handler.get_sentences())

    model = BertModel(conf.bert, e_handler.labels("num"))
    model.load_state_dict(torch.load(conf.folder + "tmp/modelB2.pt"))

    if conf.cuda:
        model = model.to("cuda:0")

    train(model, e_handler, df_train, df_val, conf)

"""
Epochs: 1 | Loss:  0.0956 | Val_Loss:  0.1102 | F1:  0.6610
Epochs: 2 | Loss:  0.0831 | Val_Loss:  0.1060 | F1:  0.6847
Epochs: 3 | Loss:  0.0798 | Val_Loss:  0.1033 | F1:  0.6890
Epochs: 4 | Loss:  0.0779 | Val_Loss:  0.1010 | F1:  0.7182
Epochs: 5 | Loss:  0.0754 | Val_Loss:  0.1002 | F1:  0.7170
Epochs: 6 | Loss:  0.0817 | Val_Loss:  0.1052 | F1:  0.7197
Epochs: 7 | Loss:  0.0793 | Val_Loss:  0.1105 | F1:  0.6957
Epochs: 8 | Loss:  0.0776 | Val_Loss:  0.1028 | F1:  0.7256

"""