import torch

from Configuration import Configuration
from Evaluation.metrics import eval_model
from Parsing.parser_utils import Splitting, ensembleParser
from Training.NERClassifier import NERClassifier

conf = Configuration()
conf.update_params("lr", 0.001)
conf.show_parameters(["files", "bert"])

path_a = "../Source/dataset.a.conll"
path_b = "../Source/dataset.b.conll"

(handler_a, handler_b), unified_dt = ensembleParser(path_a, path_b)
df_train, df_val, df_test = Splitting().holdout(unified_dt)

modelA = NERClassifier(conf.bert, 9)
# modelA.load_state_dict(torch.load(conf.folder + "tmp/modelH1.pt"))
modelA.load_state_dict(torch.load("K:/NoSyncCache/Models/A/modelH1.pt"))

modelB = NERClassifier(conf.bert, 5)
# modelB.load_state_dict(torch.load(conf.folder + "tmp/modelE2.pt"))
modelB.load_state_dict(torch.load("K:/NoSyncCache/Models/B/modelE2.pt"))

if conf.cuda:
    modelA = modelA.to("cuda:0")
    modelB = modelB.to("cuda:0")

eval_model(modelA, df_test[["sentences", "labels_a"]], conf, handler_a)
eval_model(modelB, df_test[["sentences", "labels_b"]], conf, handler_b)
