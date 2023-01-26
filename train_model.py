from Configuration import Configuration
from Parsing.parser_utils import parse_args, buildDataset, holdout
from Training.NERClassifier import NERClassifier
from Training.NERCRFClassifier import NERCRFClassifier
from Training.Trainer import train

if __name__ == '__main__':

    args, _ = parse_args()

    conf = Configuration(args)
    conf.show_parameters(["param", "bert"])

    if args.datasets is None or len(args.datasets) != 1:
        raise Exception("Train only a dataset per time!")

    if args.model_name is None:
        raise Exception("Define a model name!")

    handler = buildDataset(args.datasets[0], verbose=True)
    df_train, df_val, df_test = holdout(handler.dt)

    model = NERCRFClassifier(conf.bert, len(handler.set_entities), handler)

    if conf.cuda:
        model = model.to(conf.gpu)

    train(model, handler, df_train, df_val, conf)
