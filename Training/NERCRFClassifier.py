from typing import Optional

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput

from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions

from Parsing.parser_utils import EntityHandler


class CustomBert(BertPreTrainedModel):  # noqa

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, handler: EntityHandler):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.crf_layer = ConditionalRandomField(num_tags=config.num_labels,
                                                constraints=allowed_transitions(constraint_type="BIO",
                                                                                labels=handler.id2label))

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

        outputs = self.bert(
            input_ids,
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

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        logits = F.log_softmax(logits, dim=-1)

        loss = None
        if labels is not None:
            loss = -self.crf_layer(logits, labels, attention_mask) / float(input_ids.size(0))

        best_path = self.crf_layer.viterbi_tags(logits, attention_mask)

        if not return_dict:
            output = (best_path,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output


class NERCRFClassifier(Module):
    def __init__(self, bert: str, tot_labels: int, handler):
        """
        Bert model
        :param bert: Name of bert used
        :param tot_labels: Total number of label for the classification
        :param frozen: True to freeze the deep parameters
        """
        super(NERCRFClassifier, self).__init__()

        self.bert = CustomBert.from_pretrained(bert, num_labels=tot_labels, handler=handler)

        return

    def forward(self, input_id, mask, label):
        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        return output
