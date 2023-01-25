from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from torch import nn, LongTensor
from torch.nn import Module
from transformers import AutoModel
import torch.nn.functional as F


class NERCRFClassifier(Module):
    def __init__(self, bert: str, num_labels: int, entity_handler, use_gpu: bool):
        super(NERCRFClassifier, self).__init__()

        self.bert = AutoModel.from_pretrained(bert, return_dict=True)
        self.feedforward = nn.Linear(in_features=self.bert.config.hidden_size, out_features=num_labels)

        self.crf_layer = ConditionalRandomField(num_tags=num_labels,
                                                constraints=allowed_transitions(constraint_type="BIO",
                                                                                labels=entity_handler.id2label))
        self.dropout = nn.Dropout(0.1)
        self.use_gpu = use_gpu
        return

    def forward(self, input_id, mask, label):
        batch_size = input_id.size(0)

        embedded = self.bert(input_ids=input_id, attention_mask=mask)
        embedded = embedded.last_hidden_state
        embedded = self.dropout(F.leaky_relu(embedded))

        token_scores = self.feedforward(embedded)
        token_scores = F.log_softmax(token_scores, dim=-1)

        loss = -self.crf_layer(token_scores, label, mask) / float(batch_size)
        best_path = self.crf_layer.viterbi_tags(token_scores, mask)

        if self.use_gpu:
            best_path = [LongTensor(seq).to("cuda:0") for seq, _ in best_path]
        else:
            best_path = [LongTensor(seq) for seq, _ in best_path]

        return loss, best_path
