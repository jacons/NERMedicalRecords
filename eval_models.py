import torch

from Configuration import Configuration
from Evaluation.metrics import eval_model
from Parsing.parser_utils import parse_args, ensembleParser, holdout
from Training.BERTClassifier import BERTClassifier

if __name__ == '__main__':

    args, _ = parse_args()

    conf = Configuration(args)
    conf.show_parameters(["bert"])

    if (args.datasets is None) or (args.models is None) or len(args.datasets) != 2 or len(args.models) != 2:
        raise Exception("Define datasets and models in the same order!")

    paths = args.datasets
    models = args.models

    (handler_a, handler_b), unified_dt = ensembleParser(paths[0], paths[1])
    _, _, df_test = holdout(unified_dt)

    modelA = BERTClassifier(conf.bert, 9, frozen=False)
    modelA.load_state_dict(torch.load(models[0]))

    modelB = BERTClassifier(conf.bert, 5, frozen=False)
    modelB.load_state_dict(torch.load(models[1]))

    if conf.cuda:
        modelA = modelA.to(conf.gpu)
        modelB = modelB.to(conf.gpu)

    eval_model(modelA, df_test[["sentences", "labels_a"]], conf, handler_a, result=args.type_eval)
    eval_model(modelB, df_test[["sentences", "labels_b"]], conf, handler_b, result=args.type_eval)
