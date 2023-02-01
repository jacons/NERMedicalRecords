> University of Pisa, UNIPI \
> Academic year 2022/23 \
> Authors: [Iommi Andrea](https://github.com/jacons) \
> January, 2022
>

# Named entity recognition for Clinical records.

As final project for Human Language Technologies (HLT) I developed a project that extracts knowledge from Italian
medical records written by physician and provides a simple web interface to make prediction on sentences. I also
compared the quality of project’s result with the result of MultiCoNER competition. Both models uses BERT + CRF .

Current results:

| Entity   | Precision | Recall |     F1 |
|:---------|:---------:|:------:|-------:|
| **ACTI** |  98.30%   | 97.85% | 98.07% |
| **DISO** |  96.05%   | 96.79% | 96.42% |
| **DRUG** |  98.67%   | 99.30% | 98.98% |
| **SIGN** |  95.53%   | 95.97% | 95.75% |
| **BODY** |  98.71%   | 98.71% | 98.71% |
| **TREA** |  98.80%   | 99.12% | 98.96% |

#### *Report outdated further improvement have been applied*

report : [NER_for_Medical_Records.pdf](https://github.com/jacons/NERMedicalRecords/files/10427990/NER_for_Medical_Records.pdf)

### Running the Code

#### Arguments:

```
p = argparse.ArgumentParser(description='Model configuration.', add_help=True)
p.add_argument('--datasets', type=str, nargs='+',
    help='Dataset used for training, it will split in training, validation and test', default=None)
    
p.add_argument('--models', type=str, nargs='+',
    help='Model trained ready to evaluate or use, if list, the order must follow the same of datasets',
    default=None)
    
p.add_argument('--model_name', type=str,
    help='Name to give to a trained model', default=None)
    
p.add_argument('--path_model', type=str,
    help='Directory to save the model', default=".")
    
p.add_argument('--bert', type=str,
    help='Bert model provided by Huggingface', default="dbmdz/bert-base-italian-xxl-cased")

p.add_argument('--save', type=int,
    help='set 1 if you want save the model otherwise set 0', default=1)

p.add_argument('--eval', type=str,
    help='define the type of evaluation: conlleval or df', default="conlleval")
    
p.add_argument('--lr', type=float, help='Learning rate', default=0.004)
    
p.add_argument('--momentum', type=float, help='Momentum', default=0.9)
    
p.add_argument('--weight_decay', type=float, help='Weight decay', default=0.0002)
    
p.add_argument('--batch_size', type=int, help='Batch size', default=16)
    
p.add_argument('--max_epoch', type=int, help='Max number of epochs', default=15)
    
p.add_argument('--patience', type=float, help='Patience in early stopping', default=3)
``` 

#### Running

###### Train model

```
python train_model.py --model_name modelA --lr 0.0004 --max_epoch 12 --batch_size 16 --datasets dataset.a.conll
```

###### Evaluate the trained model

```
python eval_models.py --models modelA.pt modelB.pt --datasets dataset.a.conll dataset.b.conll --eval conlleval
```

###### Prediction by command line

```
python prediction.py --models saved_models/model.a.pt saved_models/model.b.pt
```

###### Start web interface

```
set FLASK_APP=server.py;$env:FLASK_APP = "server.py";flask run
```

### Setting up the code environment

```
$ pip install -r requirements.txt
```
