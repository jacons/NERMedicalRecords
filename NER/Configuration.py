import torch


class Configuration:
    """
    Class used to store all parameters and configuration for the execution
    """

    def __init__(self, conf=None, verbose=True):
        if conf is None:
            conf = []

        # Hyperparameters
        self.param: dict = {
            "lr": 0.011,
            "momentum": 0.9,
            "weight_decay": 0,
            "batch_size": 2,
            "model_name": "modelAB",
            "max_epoch": 1,
            "early_stopping": 2,
            "nesterov": True,
            "cache": True
        }

        # We can choose if train the model with different group of entity defined in the path dictionary
        self.type_of_entity = ["a", "b"]

        # self.folder: str = "/content/drive/MyDrive/NERforMedicalRecords/"
        self.folder: str = "/content/drive/Othercomputers/Il mio Laptop/Universita/[IA] Artificial Intelligence/[HLT] " \
                           "Human Language Technologies/NERforMedicalRecords/ "

        # The system recognize if there are some GPU available
        self.cuda = True if torch.cuda.is_available() else False
        # Constants
        self.bert = "dbmdz/bert-base-italian-xxl-cased"  # Bert model as baseline
        self.files = ["anamnesi", "esami"]
        self.paths: dict = {
            "anamnesi": {
                "files": [self.folder + "Corpus/anamnesi.a.iob", self.folder + "Corpus/anamnesi.b.iob"],
                "type": ["a", "b"]},
            "esami": {
                "files": [self.folder + "Corpus/esami.a.iob", self.folder + "Corpus/esami.b.iob"],
                "type": ["a", "b"]}
        }

        if verbose:
            self.show_parameters(conf)

    def show_parameters(self, conf: list) -> None:

        if "bert" in conf:
            print("{:<85}".format("Bert model"))
            print("-" * 85)
            print("|{:^83}|".format(self.bert))
            print("-" * 85)

        if "files" in conf:
            print("{:<85}".format("File used"))
            for name in self.files:
                print("-" * 85)
                print("|{:^41}|{:^41}|".format(name, " - ".join(self.type_of_entity)))
            print("-" * 85)

        if "param" in conf:
            print("{:<85}".format("Parameters & Values"))
            print("-" * 85)
            for idx, (k, v) in enumerate(self.param.items()):

                if (idx + 1) % 3 != 0:
                    print("|{:^14} {:^12}".format(k, v), end='')

                if (idx + 1) % 3 == 0:
                    print("|{:^14} {:^12}|".format(k, v))
                    print("-" * 85)

        return
