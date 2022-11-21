import torch
from pandas import DataFrame
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from Parser import Parser


class NerDataset(Dataset):
    # We try to preprocess the data as much as possible.
    def __init__(self, dataset: DataFrame, bert: str, parser: Parser):
        self.__input_ids, self.__mask, self.__labels = [], [], []

        tokenizer = AutoTokenizer.from_pretrained(bert)

        for _, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
            tokens, _labels = row[0].split(), row[1].split()
            # Apply the tokenization at each row
            token_text = tokenizer(tokens, padding='max_length', max_length=512, truncation=True,
                                   is_split_into_words=True, return_tensors="pt")

            label_ids = parser.align_label(token_text.word_ids(), _labels)

            self.__input_ids.append(token_text['input_ids'].squeeze(0).to("cuda:0"))
            self.__mask.append(token_text['attention_mask'].squeeze(0).to("cuda:0"))
            self.__labels.append(torch.LongTensor(label_ids).to("cuda:0"))

    def __len__(self):
        return len(self.__labels)

    def __getitem__(self, idx):
        return self.__input_ids[idx], self.__mask[idx], self.__labels[idx]