import torch

from Configuration import Configuration
from Parsing.parser_utils import parse_args
from Prediction.Predictor import Predictor
from Training.NERCRFClassifier import NERCRFClassifier

if __name__ == '__main__':

    args = parse_args()

    conf = Configuration(args)
    conf.show_parameters(["bert"])

    if (args.models is None) or len(args.models) != 2:
        raise Exception("Define models!")

    models = args.models

    id2lab_group_a = {0: 'B-ACTI', 1: 'B-DISO', 2: 'B-DRUG', 3: 'B-SIGN', 4: 'I-ACTI', 5: 'I-DISO', 6: 'I-DRUG',
                      7: 'I-SIGN', 8: 'O'}

    id2lab_group_b = {0: 'B-BODY', 1: 'B-TREA', 2: 'I-BODY', 3: 'I-TREA', 4: 'O'}

    modelA = NERCRFClassifier(conf.bert, id2lab_group_a)
    modelA.load_state_dict(torch.load(models[0]))

    modelB = NERCRFClassifier(conf.bert, id2lab_group_b)
    modelB.load_state_dict(torch.load(models[1]))

    if conf.cuda:
        modelA = modelA.to(conf.gpu)
        modelB = modelB.to(conf.gpu)

    predictor = Predictor(conf)

    predictor.add_model("a", modelA, id2lab_group_a)
    predictor.add_model("b", modelB, id2lab_group_b)
