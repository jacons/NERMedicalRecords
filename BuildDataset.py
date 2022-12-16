from pandas import DataFrame
from torch import Tensor, LongTensor, IntTensor, BoolTensor
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizerFast

from Configuration import Configuration
from Parser.parser_utils import EntityHandler, align_tags


class NerDataset(Dataset):
    # We try to preprocess the data as much as possible.
    def __init__(self, dataset: DataFrame, conf: Configuration, e_handler: EntityHandler):

        self.list_of_tokens, self.list_of_att_masks, self.list_of_tag_masks, self.list_of_labels = [], [], [], []

        tokenizer = BertTokenizerFast.from_pretrained(conf.bert)
        for type_of_entity in conf.type_of_entity:
            for _, row in tqdm(dataset[["Sentences", "Labels_" + str(type_of_entity)]].iterrows(),
                               total=dataset.shape[0]):

                tokens, labels = row[0].split(), row[1].split()

                token_text = tokenizer(tokens, is_split_into_words=True)
                aligned_labels, tag_mask = align_tags(labels, token_text.word_ids())

                input_ids = IntTensor(token_text["input_ids"])
                att_mask = IntTensor(token_text["attention_mask"])
                tag_mask = BoolTensor(tag_mask)
                labels_ids = LongTensor([e_handler.label2id[tag] for tag in aligned_labels])

                if conf.cuda:
                    input_ids = input_ids.to("cuda:0")
                    att_mask = att_mask.to("cuda:0")
                    tag_mask = tag_mask.to("cuda:0")
                    labels_ids = labels_ids.to("cuda:0")

                self.list_of_tokens.append(input_ids)
                self.list_of_att_masks.append(att_mask)
                self.list_of_tag_masks.append(tag_mask)
                self.list_of_labels.append(labels_ids)

    def __len__(self):
        return len(self.list_of_labels)

    def __getitem__(self, idx):
        return self.list_of_tokens[idx], self.list_of_att_masks[idx],\
            self.list_of_tag_masks[idx], self.list_of_labels[idx]