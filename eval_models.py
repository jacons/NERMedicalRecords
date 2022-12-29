import torch

from Configuration import Configuration
from Evaluation.metrics import eval_model
from Parsing.parser_utils import parse_args, ensembleParser, Splitting
from Training.NERClassifier import NERClassifier

if __name__ == '__main__':

    args, _ = parse_args()

    conf = Configuration(args)
    conf.show_parameters(["bert"])

    if (args.datasets is None) or (args.models is None) or len(args.datasets) != 2 or len(args.models) != 2:
        raise Exception("Define datasets and models in the same order!")

    paths = args.datasets
    models = args.models

    (handler_a, handler_b), unified_dt = ensembleParser(paths[0], paths[1])
    _, _, df_test = Splitting().holdout(unified_dt)

    modelA = NERClassifier(conf.bert, 9)
    modelA.load_state_dict(torch.load(models[0]))

    modelB = NERClassifier(conf.bert, 5)
    modelB.load_state_dict(torch.load(models[1]))

    if conf.cuda:
        modelA = modelA.to(conf.gpu)
        modelB = modelB.to(conf.gpu)

    eval_model(modelA, df_test[["sentences", "labels_a"]], conf, handler_a)
    eval_model(modelB, df_test[["sentences", "labels_b"]], conf, handler_b)

    # path_a = "Source/dataset.a.conll"
    # path_b = "Source/dataset.b.conll"

    # model_a = "saved_models/model.a.pt"
    # model_b = "saved_models/model.b.pt"

    #  C:\ProgramData\Anaconda3\envs\deeplearning\python.exe eval_models.py --models saved_models/model.a.pt saved_models/model.b.pt --datasets Source/dataset.a.conll Source/dataset.b.conll

