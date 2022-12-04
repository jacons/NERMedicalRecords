import torch


class Configuration:
    def __init__(self):
        self.param = {
            "lr": 0.011,
            "momentum": 0.9,
            "weight_decay": 0,
            "batch_size": 4,
            "model_name": "modelXX.pt",
            "max_epoch": 2,
            "early_stopping": 2,
            "nesterov": True,
            "cache": True
        }

        self.cuda = True if torch.cuda.is_available() else False
        self.columns_tag = ["a"]
        self.files = ["esami", "anamnesi"]

        self.bert = "dbmdz/bert-base-italian-xxl-cased"
        self.folder = "/content/drive/MyDrive/NERforMedicalRecords/"
        self.paths = {
            "anamnesi": {
                "files": [self.folder + "Corpus/anamnesi.a.iob", self.folder + "Corpus/anamnesi.b.iob"],
                "type": self.columns_tag},
            "esami": {
                "files": [self.folder + "Corpus/esami.a.iob", self.folder + "Corpus/esami.b.iob"],
                "type": self.columns_tag}
        }
