from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from torch import nn, LongTensor
from torch.nn import Module
from transformers import AutoModel


class NERCRFClassifier(Module):
    def __init__(self, bert: str, num_labels: int, entity_handler, use_gpu: bool):
        super(NERCRFClassifier, self).__init__()

        self.bert = AutoModel.from_pretrained(bert, return_dict=True)

        self.forward_steps = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size, num_labels),
            nn.LogSoftmax(dim=1),
        )

        self.crf_layer = ConditionalRandomField(num_tags=num_labels,
                                                constraints=allowed_transitions(constraint_type="BIO",
                                                                                labels=entity_handler.id2label))
        self.use_gpu = use_gpu
        return

    def forward(self, input_id, mask, label):

        output_embedding = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        output_classifier = self.forward_steps(output_embedding)

        loss = -self.crf_layer(output_classifier, label, mask) / float(input_id.shape[0])
        best_path = self.crf_layer.viterbi_tags(output_classifier, mask)

        if self.use_gpu:
            best_path = [LongTensor(item[0]).to("cuda:0") for item in best_path]
        else:
            best_path = [LongTensor(item[0]) for item in best_path]

        return loss, best_path
