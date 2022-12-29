import torch


class Configuration:
    """
    Class used to store all parameters and configuration for the execution
    """

    def __init__(self, p):

        # Hyperparameters
        self.param: dict = {
            "lr": p.lr,
            "momentum": p.momentum,
            "weight_decay": p.weight_decay,
            "batch_size": p.batch_size,
            "max_epoch": p.max_epoch,
            "early_stopping": p.early_stopping,
        }

        self.save_model = True if p.save_model == 1 else False
        self.bert = p.bert  # Bert model as baseline

        self.model_name = p.model_name
        self.folder = p.path_model  # Directory to save the model

        # The system recognize if there are some GPU available
        self.cuda = True if torch.cuda.is_available() else False
        self.gpu = "cuda:0"

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
