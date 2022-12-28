from Configuration import Configuration
from Training.NERClassifier import NERClassifier
from Training.Trainer import train
from Parsing.parser_utils import buildDataset, Splitting

conf = Configuration()
conf.update_params("lr", 0.001)
conf.show_parameters(["files", "param", "bert"])

path = "../Source/dataset.a.conll"
handler = buildDataset(path, verbose=True)
df_train, df_val, df_test = Splitting().holdout(handler.dt)
model = NERClassifier(conf.bert, len(handler.set_entities))
# model.load_state_dict(torch.load(conf.folder + "tmp/modelA2.pt"))

if conf.cuda:
    model = model.to("cuda:0")

train(model, handler, df_train, df_val, conf)
