import torch

from Model import BertModel
from Parser import Parser, Splitting
from Training import train

# folder = "/content/drive/MyDrive/NERForMedicalRecords/"
# folder = "/content/drive/Othercomputers/Il mio Laptop/Universita/[IA] Artificial Intelligence/[HLT] Human Language Technologies/NERforMedicalRecords/"
folder = "./../../NERForMedicalRecords/"

bert = "dbmdz/bert-base-italian-xxl-cased"

# list of file to take into account
datasets = [folder + "Corpus/anamnesi.a.iob", folder + "Corpus/esami.a.iob"]

parser = Parser(datasets)
df_train, df_val, df_test = Splitting().holdout(parser.get_sentences(), size=1)

param = {
    "lr": 0.009,
    "momentum": 0.6,
    "weight_decay": 0,
    "batch_size": 4,
    "model_name": "modelC1.pt",
    "max_epoch": 2,
    "early_stopping": 2,
    "nesterov": True,
    "cache": True
}

# 31.38
model = BertModel(bert, parser.labels("num")).to("cuda:0")
model.load_state_dict(torch.load("modelC1.pt"))

train(model, bert, parser, df_train, df_val, param)

"""
model.load_state_dict(torch.load("K:/NoSyncCache/Models/ModelI.pt"))
model.eval()
m = Metrics(model, bert, parser, df_test)
m.perform()
m.overall_result("all")
m.overall_result("mean")
"""
