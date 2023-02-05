import torch
from pandas import DataFrame

from Configuration import Configuration
from Evaluation.metrics import eval_model
from Parsing.parser_utils import parse_args, ensembleParser, holdout
from Training.NERCRFClassifier import NERCRFClassifier


def outputs(results, error_dict):
    if isinstance(results, DataFrame):
        print(results)
    else:
        print(results.getvalue())

    if error_dict is not None:
        print(error_dict)


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

    modelA = NERCRFClassifier(conf.bert, handler_a.id2label)
    modelA.load_state_dict(torch.load(models[0]))

    modelB = NERCRFClassifier(conf.bert, handler_b.id2label)
    modelB.load_state_dict(torch.load(models[1]))

    if conf.cuda:
        modelA = modelA.to(conf.gpu)
        modelB = modelB.to(conf.gpu)

    output_results, error_dict_A = eval_model(modelA, df_test[["sentences", "labels_a"]], conf, handler_a,
                                              result=args.eval, return_dict=True)

    outputs(output_results, error_dict_A)

    output_results, error_dict_B = eval_model(modelB, df_test[["sentences", "labels_b"]], conf, handler_b,
                                              result=args.eval, return_dict=True)

    outputs(output_results, error_dict_B)
