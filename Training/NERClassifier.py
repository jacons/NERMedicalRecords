from torch.nn import Module
from transformers import BertForTokenClassification


class NERClassifier(Module):
    def __init__(self, bert: str, tot_labels: int, frozen: bool = True):
        """
        Bert model
        :param bert: Name of bert used
        :param tot_labels: Total number of label for the classification
        :param frozen: True to freeze the deep parameters
        """
        super(NERClassifier, self).__init__()

        self.bert = BertForTokenClassification.from_pretrained(bert, num_labels=tot_labels)
        if frozen:
            for name, param in self.bert.bert.named_parameters():
                if not name.startswith("encoder.layer.11"):
                    param.requires_grad = False
        return

    def forward(self, input_id, mask, label):
        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        return output
