import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from dataset.sentiment_dataset import CLSDataset
from models.sentiment_analysis import BertSentimentPredict


MODELPATH = './chinese_rbtl3_pytorch'


class Trainer:
    def __init__(self, model, lr, batch_size=16, use_cuda=True):

        bert_model = BertModel.from_pretrained(MODELPATH)
        self.model = model(bert_model)
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device('cuda') if use_cuda and torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

        self.tokenizer = BertTokenizer.from_pretrained(MODELPATH)

        self.train_data = CLSDataset('corpus/train_sentiment.txt', self.tokenizer, max_seq_len=300)
        self.test_data = CLSDataset('corpus/test_sentiment.txt', self.tokenizer, max_seq_len=300)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.1)

    def padding(self, one_batch):
        text = [data['text_input'] for data in one_batch]
        label = torch.cat([data['label'] for data in one_batch])
        text = torch.nn.utils.rnn.pad_sequence(text, batch_first=True)
        return text, label

    def test(self, preds, labels):
        preds = torch.argmax(preds, dim=-1).flatten().to(torch.device('cpu'))
        labels = labels.flatten().to(torch.device('cpu'))
        correct = torch.sum(preds == labels)
        return correct

    def iteration(self, epoch, is_train=True):
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        data = self.train_data if is_train else self.test_data
        dataloader = DataLoader(data, shuffle=True, collate_fn=lambda x: x, batch_size=self.batch_size)
        acc_number = 0
        counter = 0
        for i, data in tqdm(enumerate(dataloader), desc=f'epoch_{epoch}'):
            data, labels = self.padding(data)
            data, labels = data.to(self.device), labels.to(self.device)
            preds = self.model(data)
            if is_train:
                self.optimizer.zero_grad()
                loss = self.loss(preds, labels)
                print(f'loss:{loss.item():.4f}')
                loss.backward()
                self.optimizer.step()
            else:
                acc_number += self.test(preds, labels).item()
                counter += labels.nelement()
        if not is_train:
            print(f'Acc:{acc_number / counter}')


if __name__ == '__main__':
    trainer = Trainer(BertSentimentPredict, lr=1e-4)
    for i in range(10):
        trainer.iteration(i)
