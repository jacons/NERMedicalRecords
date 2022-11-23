import torch
from transformers import AutoModelForMaskedLM


class BertModel(torch.nn.Module):
    def __init__(self, bert: str, tot_labels: int, frozen=True):
        """
        :param bert: Name of bert used
        :param tot_labels: Total number of label for the classification
        :param frozen: True to freeze the deep parameters
        """
        super(BertModel, self).__init__()
        self.bert = AutoModelForMaskedLM.from_pretrained(bert, num_labels=tot_labels)
        if frozen:
            for param in self.bert.bert.parameters():
                param.requires_grad = False
        return

    def forward(self, input_id, mask, label):
        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        return output