import torch
from flask import Flask, render_template, request

from Parser.parsers import EnsembleParser
from Prediction.prediction import Predictor
from Training.NER_model import NERClassifier
from configuration import Configuration

app = Flask(__name__)

conf = Configuration()
conf.update_params("lr", 0.001)
conf.update_entities(["a", "b"])
conf.show_parameters()

handlers, _ = EnsembleParser(conf, verbose=False)
modelA = NERClassifier(conf.bert, handlers["a"].labels("num"))
modelA.load_state_dict(torch.load("K:/NoSyncCache/Models/A/modelE1.pt",
                                  map_location=torch.device('cpu')))  # map_location=torch.device('cpu')

modelB = NERClassifier(conf.bert, handlers["b"].labels("num"))
modelB.load_state_dict(torch.load("K:/NoSyncCache/Models/B/modelE2.pt", map_location=torch.device('cpu')))

predictor = Predictor(conf)
predictor.add_model("a", modelA, handlers["a"])
predictor.add_model("b", modelB, handlers["b"])


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
