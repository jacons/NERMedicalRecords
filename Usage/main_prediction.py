import torch

from Configuration import Configuration
from Prediction.prediction import Predictor
from Training.NERClassifier import NERClassifier

conf = Configuration()
conf.update_params("lr", 0.001)
conf.show_parameters(["bert"])

modelA = NERClassifier(conf.bert, 9)
# modelA.load_state_dict(torch.load(conf.folder + "tmp/modelH1.pt"))
modelA.load_state_dict(torch.load("K:/NoSyncCache/Models/A/modelH1.pt"))

modelB = NERClassifier(conf.bert, 5)
# modelB.load_state_dict(torch.load(conf.folder + "tmp/modelE2.pt"))
modelB.load_state_dict(torch.load("K:/NoSyncCache/Models/B/modelE2.pt"))

if conf.cuda:
    modelA = modelA.to("cuda:0")
    modelB = modelB.to("cuda:0")

predictor = Predictor(conf)

id2lab_group_a = {0: 'B-ACTI', 1: 'B-DISO', 2: 'B-DRUG', 3: 'B-SIGN', 4: 'I-ACTI', 5: 'I-DISO', 6: 'I-DRUG',
                  7: 'I-SIGN', 8: 'O'}

id2lab_group_b = {0: 'B-BODY', 1: 'B-TREA', 2: 'I-BODY', 3: 'I-TREA', 4: 'O'}

predictor.add_model("a", modelA, id2lab_group_a)
predictor.add_model("b", modelB, id2lab_group_b)


print(predictor.predict("Hello!!"))
