import numpy as np
import torch
from pandas import DataFrame
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer

from Configuration import Configuration
from Model import BertModel
from Parser.parser_utils import EntityHandler


"""

def convert(labels: Tensor, dic: dict) -> list:

    converts the numeric identification of the tag into entity (str)
    :param labels: tensor of numerical values
    :param dic: dictionary k -> v k = id of tag, v -> string representation
    :return: list of string-based tags
    
    return [dic[i.item()] for i in labels]


def merge_labels(labelsA: list, labelsB: list) -> list:
    # join two list of labels into one
    return [[a, b] for a, b in zip(labelsA, labelsB)]


def unify_labels(labelsA: list, labelsB: list) -> list:
    # creates a list of tags, if there are more than one tag to keep, it generates a list
    unified = []
    for a, b in zip(labelsA, labelsB):
        if a == b or b == "O":
            unified.append(a)
        elif a == "O":
            unified.append(b)
        else:
            unified.append([a, b])
    return unified


def scores(confusion: Tensor, labels: dict):
    # takes a confusion matrix and generate a dataframe with labels as index and metrics as column

    length = confusion.shape[0]

    iter_label = range(length)
    accuracy: Tensor = torch.zeros(length)
    precision: Tensor = torch.zeros(length)
    recall: Tensor = torch.zeros(length)
    f1: Tensor = torch.zeros(length)

    for i in iter_label:
        fn = torch.sum(confusion[i, :i]) + torch.sum(confusion[i, i + 1:])  # false negative
        fp = torch.sum(confusion[:i, i]) + torch.sum(confusion[i + 1:, i])  # false positive
        tn, tp = 0, confusion[i, i]  # true negative, true positive

        for x in iter_label:
            for y in iter_label:
                if (x != i) & (y != i):
                    tn += confusion[x, y]

        accuracy[i] = (tp + tn) / (tp + fn + fp + tn)
        precision[i] = tp / (tp + fp)
        recall[i] = tp / (tp + fn)
        f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

    return DataFrame({
        "Accuracy": accuracy.tolist(),
        "Precision": precision.tolist(),
        "Recall": recall.tolist(),
        "F1": f1.tolist()
    }, index=list(labels.values()))


def multiple_eval(modelA: BertModel, modelB: BertModel, dataset: DataFrame, conf: Configuration,
                  handler: dict) -> DataFrame:
    # evaluates the dataframe with two separated model

    max_labelsA = handler["a"].labels("num")
    max_labelsB = handler["b"].labels("num")

    # they represent a dictionary that map the ids(int) into entity (str)
    mapperA = handler["a"].ids_to_labels
    mapperB = handler["b"].ids_to_labels

    # instantiate a confusion matrix
    confusionA = torch.zeros(size=(max_labelsA, max_labelsA))
    confusionB = torch.zeros(size=(max_labelsB, max_labelsB))

    tokenizer = AutoTokenizer.from_pretrained(conf.bert)
    for row in tqdm(dataset.itertuples(), total=dataset.shape[0]):

        tokens, labelsA, labelsB = row[1].split(), row[3].split(), row[4].split()

        # Create a tokenized text
        token_text = tokenizer(tokens, max_length=512, truncation=True, is_split_into_words=True, return_tensors="pt")

        # ============ HANDLE LABELING ============
        # Align both list of labels
        labelsA = handler["a"].align_label(token_text.word_ids(), labelsA)
        labelsB = handler["b"].align_label(token_text.word_ids(), labelsB)

        # Create a mask to select values that are different from -100 (it is equal for both group of tags)
        mask = [True if i != -100 else False for i in labelsA]

        labelsA = convert(np.array(labelsA)[mask], mapperA)
        labelsB = convert(np.array(labelsB)[mask], mapperB)
        # generate a list of list that contains the entities
        # es. [[entity(groupA),entity(groupB)],[entity(groupA),entity(groupB)]..]
        unified_labels = merge_labels(labelsA, labelsB)
        # ============ HANDLE LABELING ============

        input_ids = token_text['input_ids']
        attention_mask = token_text['attention_mask']

        if conf.cuda:
            input_ids = input_ids.to("cuda:0")
            attention_mask = attention_mask.to("cuda:0")

        # ============ PREDICTION ============
        # Give the same sentence to both model
        logitsA = modelA(input_ids, attention_mask, None)
        logitsB = modelB(input_ids, attention_mask, None)
        # Return different labels
        logitsA = logitsA[0].squeeze().argmax(dim=1)[mask]
        logitsB = logitsB[0].squeeze().argmax(dim=1)[mask]
        # generate a list of list that contains the entities
        # es. [[entity(groupA),entity(groupB)],[entity(groupA),entity(groupB)]..]
        unified_predict = merge_labels(convert(logitsA, mapperA), convert(logitsB, mapperB))
        # ============ PREDICTION ============

        # Update the confusion matrices
        for lbl, pre in zip(unified_labels, unified_predict):
            # update matrix with group of tag A
            idx_l = handler["a"].labels_to_ids[lbl[0]]
            idx_p = handler["a"].labels_to_ids[pre[0]]
            confusionA[idx_l, idx_p] += 1

            # update matrix with group of tag B
            idx_l = handler["b"].labels_to_ids[lbl[1]]
            idx_p = handler["b"].labels_to_ids[pre[1]]
            confusionB[idx_l, idx_p] += 1

    # Create two dataframe that contains the metric for each group of tags
    dfA, dfB = scores(confusionA, mapperA), scores(confusionB, mapperB)
    return dfA[dfA.index != "O"].append(dfB[dfB.index != "O"])


def single_eval(model: BertModel, dataset: DataFrame, conf: Configuration, handler: EntityHandler) -> DataFrame:
    # evaluate the dataframe with a single model

    n_labs = handler.labels("num")  # num of tags

    # they represent a dictionary that map the ids(int) into entity (str)
    mapper = handler.ids_to_labels

    # instantiate a confusion matrix
    confusion = torch.zeros(size=(n_labs, n_labs))

    tokenizer = AutoTokenizer.from_pretrained(conf.bert)
    for row in tqdm(dataset.itertuples(), total=dataset.shape[0]):

        tokens, labelsA, labelsB = row[1].split(), row[3].split(), row[4].split()

        # Create a tokenized text
        token_text = tokenizer(tokens, max_length=512, truncation=True, is_split_into_words=True, return_tensors="pt")

        # Align both list of labels with the same token text
        labelsA = handler.align_label(token_text.word_ids(), labelsA)
        labelsB = handler.align_label(token_text.word_ids(), labelsB)

        # Create a mask to select values that are different from -100 (it is equal for both group of tags)
        mask = [True if i != -100 else False for i in labelsA]

        input_ids = token_text['input_ids']
        attention_mask = token_text['attention_mask']

        if conf.cuda:
            input_ids = input_ids.to("cuda:0")
            attention_mask = attention_mask.to("cuda:0")

        # ============ PREDICTION ============
        logits = model(input_ids, attention_mask, None)
        logits = logits[0].squeeze().argmax(dim=1)[mask]
        predict = convert(logits, mapper)
        # ============ PREDICTION ============

        labelsA = convert(np.array(labelsA)[mask], mapper)
        labelsB = convert(np.array(labelsB)[mask], mapper)
        unified_labels = unify_labels(labelsA, labelsB)

        # Update the confusion matrices
        for lbl, pre in zip(unified_labels, predict):

            if isinstance(lbl, str):
                idx_l = handler.labels_to_ids[lbl]
                idx_p = handler.labels_to_ids[pre]
                confusion[idx_l, idx_p] += 1

            elif isinstance(lbl, list):
                for tag in lbl:
                    idx_l = handler.labels_to_ids[tag]
                    idx_p = handler.labels_to_ids[pre]
                    confusion[idx_l, idx_p] += 1

    return scores(confusion, mapper)

"""