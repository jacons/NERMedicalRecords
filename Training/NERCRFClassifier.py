from typing import Optional

import torch
from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from torch import nn, LongTensor
from torch.nn import Module
from torch.nn.functional import leaky_relu
from transformers import BertPreTrainedModel, BertModel


class BertForNERTagging(BertPreTrainedModel):  # noqa
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, entity_handler, use_gpu):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.use_gpu = use_gpu
        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.crf_layer = ConditionalRandomField(num_tags=len(entity_handler.id2label),
                                                constraints=allowed_transitions(constraint_type="BIO",
                                                                                labels=entity_handler.id2label))
        self.log_softmax = nn.LogSoftmax(dim=1)

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
        sequence_output = leaky_relu(sequence_output)
        sequence_output = self.dropout(sequence_output)
        token_scores = self.classifier(sequence_output)
        token_scores = self.log_softmax(token_scores)

        best_path = self.crf_layer.viterbi_tags(token_scores, attention_mask)

        if self.use_gpu:
            best_path = [LongTensor(item[0]).to("cuda:0") for item in best_path]
        else:
            best_path = [LongTensor(item[0]) for item in best_path]

        if labels is not None:
            loss = -self.crf_layer(token_scores, labels, attention_mask) / float(token_scores.shape[0])
            return loss, best_path

        return best_path


class NERCRFClassifier(Module):
    def __init__(self, bert: str, num_labels: int, entity_handler, use_gpu: bool):
        super(NERCRFClassifier, self).__init__()

        self.bert = BertForNERTagging.from_pretrained(bert, num_labels=num_labels, entity_handler=entity_handler,
                                                      use_gpu=use_gpu)
        return

    def forward(self, input_id, mask, label):
        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        return output
