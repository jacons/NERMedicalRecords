from typing import Optional

import torch
from torch import nn
from torch.nn import Module, CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput


class BertGruForTokenClassification(BertPreTrainedModel):  # noqa

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.n_layers = 1
        self.hidden_dim = 768

        self.bert = BertModel(config)

        self.lstm = nn.LSTM(768, self.hidden_dim, self.n_layers, batch_first=True)

        self.dropout = nn.Dropout(0.1)

        self.fc = nn.Linear(self.hidden_dim, self.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        # print("bert",sequence_output.shape)

        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().float().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().float().cuda())

        out, (h, c) = self.lstm(sequence_output, hidden)
        # print("hidden last",out.shape)

        out = self.dropout(out)
        # print("out",out.shape)

        logits = self.fc(out)
        # print("logits",logits.shape)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BERTGRUClassifier(Module):
    def __init__(self, bert: str, tot_labels: int, frozen: bool = True):
        """
        Bert model
        :param bert: Name of bert used
        :param tot_labels: Total number of label for the classification
        :param frozen: True to freeze the deep parameters
        """
        super(BERTGRUClassifier, self).__init__()
        self.bert = BertGruForTokenClassification.from_pretrained(bert, num_labels=tot_labels)
        if frozen:
            for name, param in self.bert.bert.named_parameters():
                if not name.startswith("encoder.layer.11"):
                    param.requires_grad = False
        return

    def forward(self, input_id, mask, label):
        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        return output

