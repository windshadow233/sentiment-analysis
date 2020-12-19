import torch
from torch import nn


class BertSentimentPredict(nn.Module):
    def __init__(self, bert_model):
        super(BertSentimentPredict, self).__init__()
        self.bert_model = bert_model
        self.dense = nn.Linear(1024 * 2, 1024)
        self.final_dense = nn.Linear(1024, 2)
        self.activation = nn.Sigmoid()

    def forward(self, text_input):
        encoded_layers, pooled = self.bert_model(text_input)
        avg_pooled = encoded_layers.mean(1)
        max_pooled = torch.max(encoded_layers, dim=1)
        pooled = torch.cat((avg_pooled, max_pooled[0]), dim=1)
        pooled = self.dense(pooled)
        pooled = self.activation(pooled)

        predictions = self.final_dense(pooled)
        # predictions = self.softmax(predictions)
        return predictions
