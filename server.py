import torch
from flask import Flask, render_template, request

from Configuration import Configuration
from Parsing.parser_utils import parse_args
from Prediction.Predictor import Predictor
from Training.NERCRFClassifier import NERCRFClassifier

app = Flask(__name__)

args, _ = parse_args()

conf = Configuration(args)
conf.show_parameters(["bert"])

models = ["saved_models/model.a.pt", "saved_models/model.b.pt"]

id2lab_group_a = {0: 'B-ACTI', 1: 'B-DISO', 2: 'B-DRUG', 3: 'B-SIGN', 4: 'I-ACTI', 5: 'I-DISO', 6: 'I-DRUG',
                  7: 'I-SIGN', 8: 'O'}

id2lab_group_b = {0: 'B-BODY', 1: 'B-TREA', 2: 'I-BODY', 3: 'I-TREA', 4: 'O'}

modelA = NERCRFClassifier(conf.bert, id2lab_group_a)
modelA.load_state_dict(torch.load(models[0], map_location=torch.device('cpu')))

modelB = NERCRFClassifier(conf.bert, id2lab_group_b)
modelB.load_state_dict(torch.load(models[1], map_location=torch.device('cpu')))

if conf.cuda:
    modelA = modelA.to(conf.gpu)
    modelB = modelB.to(conf.gpu)

predictor = Predictor(conf)

predictor.add_model("a", modelA, id2lab_group_a)
predictor.add_model("b", modelB, id2lab_group_b)

list_of_result = []


@app.route('/', methods=('GET', 'POST'))
def create():
    if request.method == 'POST':

        sentence = request.form['Sentence']

        if "predict" in request.form and sentence != "":
            tag_pred, mask = predictor.predict(sentence)
            result_ = [*zip(sentence.split(), tag_pred, mask)]
            list_of_result.append(result_)

        elif "clear" in request.form:
            list_of_result.clear()

    return render_template('main.html', list_of_result=list_of_result)
