import torch.nn as nn
# import torch
# from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output

# class SentenceClassifier(BertPreTrainedModel):
#     def __init__(self, config, num_classes, vocab) -> None:
#         super(SentenceClassifier, self).__init__(config)
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, num_classes)
#         self.vocab = vocab
#         self.apply(self.init_weights)
#
#     def forward(self, input_ids):
#         # pooled_output is not same hidden vector corresponds to first token from last encoded layers
#         attention_mask = input_ids.ne(self.vocab.to_indices(self.vocab.padding_token)).float()
#         _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#         return logits
