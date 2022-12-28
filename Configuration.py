import torch


class Configuration:
    """
    Class used to store all parameters and configuration for the execution
    """

    def __init__(self):

        # Hyperparameters
        self.param: dict = {
            "lr": 0.008,
            "momentum": 0.8,
            "weight_decay": 0.0002,
            "batch_size": 2,
            "model_name": "modelA2",
            "max_epoch": 1,
            "early_stopping": 3,
            "nesterov": True,
            "cache": False
        }

        # The system recognize if there are some GPU available
        self.cuda = True if torch.cuda.is_available() else False
        # Constants
        self.bert = "dbmdz/bert-base-italian-xxl-cased"  # Bert model as baseline

    def update_params(self, param: str, value: float):
        self.param[param] = value

    def show_parameters(self, conf=None) -> None:
        if conf is None:
            conf = []

        if "bert" in conf:
            print("{:<85}".format("Bert model"))
            print("-" * 85)
            print("|{:^83}|".format(self.bert))
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
