import torch

from configuration import Configuration
from Training.NERClassifier import NERClassifier
from Parser.parsers import Parser
from Parser.parser_utils import Splitting
from Training.training import train

if __name__ == "__main__":

    conf = Configuration(["files", "param", "bert"])
    e_handler = Parser(conf)
    df_train, df_val, df_test = Splitting().holdout(e_handler.get_sentences())

    model = NERClassifier(conf.bert, e_handler.labels("num"))
    model.load_state_dict(torch.load(conf.folder + "tmp/modelB2.pt"))

    if conf.cuda:
        model = model.to("cuda:0")

    train(model, e_handler, df_train, df_val, conf)