from pandas import DataFrame
from torch import LongTensor
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from NER.Configuration import Configuration
from Parser import Parser


class NerDataset(Dataset):
    # We try to preprocess the data as much as possible.
    def __init__(self, dataset: DataFrame, conf: Configuration, parser: Parser):
        self.__input_ids, self.__mask, self.__labels = [], [], []

        tokenizer = AutoTokenizer.from_pretrained(conf.bert)

        for column in conf.columns_tag:
            print("File type: " + column)
            for _, row in tqdm(dataset[["Sentences", "lbl-" + str(column)]].iterrows(), total=dataset.shape[0]):

                tokens, _labels = row[0].split(), row[1].split()

                # Apply the tokenization at each row
                token_text = tokenizer(tokens, max_length=512, truncation=True, is_split_into_words=True,
                                       return_tensors="pt")

                label_ids = parser.align_label(token_text.word_ids(), _labels)

                input_ = token_text['input_ids'].squeeze(0)
                mask_ = token_text['attention_mask'].squeeze(0)
                label_ = LongTensor(label_ids)

                if conf.cuda:
                    input_ = input_.to("cuda:0")
                    mask_ = mask_.to("cuda:0")
                    label_ = label_.to("cuda:0")

                self.__input_ids.append(input_)
                self.__mask.append(mask_)
                self.__labels.append(label_)

    def __len__(self):
        return len(self.__labels)

    def __getitem__(self, idx):
        return self.__input_ids[idx], self.__mask[idx], self.__labels[idx]
