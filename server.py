import torch
from flask import Flask, render_template, request

from Configuration import Configuration
from Parsing.parser_utils import parse_args
from Prediction.Predictor import Predictor
from Training.NERClassifier import NERClassifier

app = Flask(__name__)

args, _ = parse_args()

conf = Configuration(args)
conf.show_parameters(["bert"])

models = ["saved_models/model.a.pt", "saved_models/model.b.pt"]

modelA = NERClassifier(conf.bert, 9)
modelA.load_state_dict(torch.load(models[0], map_location=torch.device('cpu')))

modelB = NERClassifier(conf.bert, 5)
modelB.load_state_dict(torch.load(models[1], map_location=torch.device('cpu')))

if conf.cuda:
    modelA = modelA.to(conf.gpu)
    modelB = modelB.to(conf.gpu)

predictor = Predictor(conf)

id2lab_group_a = {0: 'B-ACTI', 1: 'B-DISO', 2: 'B-DRUG', 3: 'B-SIGN', 4: 'I-ACTI', 5: 'I-DISO', 6: 'I-DRUG',
                  7: 'I-SIGN', 8: 'O'}

id2lab_group_b = {0: 'B-BODY', 1: 'B-TREA', 2: 'I-BODY', 3: 'I-TREA', 4: 'O'}

predictor.add_model("a", modelA, id2lab_group_a)
predictor.add_model("b", modelB, id2lab_group_b)


@app.route('/', methods=('GET', 'POST'))
def create():
    if request.method == 'POST':
        sentence = request.form['Sentence']
        tag_pred = predictor.predict(sentence)
        result_ = [*zip(sentence.split(), tag_pred)]
    else:
        result_ = []
    return render_template('main.html', result=result_)


"""
    set FLASK_APP=server.py;$env:FLASK_APP = "server.py";flask run
"""
