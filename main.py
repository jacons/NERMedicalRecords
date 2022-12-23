from Parser.parser_utils import Splitting
from Parser.parsers import Parser
from Training.NER_model import NERClassifier
from Training.training import train
from configuration import Configuration

if __name__ == "__main__":

    conf = Configuration(["files", "param", "bert"])
    e_handler = Parser(conf)
    df_train, df_val, df_test = Splitting().holdout(e_handler.get_sentences())

    model = NERClassifier(conf.bert, e_handler.labels("num"))
    # model.load_state_dict(torch.load(conf.folder + "tmp/modelA2.pt"))

    if conf.cuda:
        model = model.to("cuda:0")

    train(model, e_handler, df_train, df_val, conf)

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
