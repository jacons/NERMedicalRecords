from Configuration import Configuration
from Parsing.parser_utils import parse_args, buildDataset, Splitting
from Training.NERClassifier import NERClassifier
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
    df_train, df_val, df_test = Splitting().holdout(handler.dt)
    model = NERClassifier(conf.bert, len(handler.set_entities))
    # model.load_state_dict(torch.load(conf.folder + "tmp/modelA2.pt"))

    if conf.cuda:
        model = model.to(conf.gpu)

    train(model, handler, df_train, df_val, conf)

    """
    C:\ProgramData\Anaconda3\envs\deeplearning\python.exe train_model.py --model_name prova.pt --max_epoch 1 --datasets .\Source\dataset.a.conll
    """