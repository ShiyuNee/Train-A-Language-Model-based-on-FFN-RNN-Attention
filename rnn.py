import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import tqdm

os.environ['CUDA_VISIBLE_DEVICES']='1'

def get_data(path):
    res_data = []
    with open(path) as file:
        data = file.readlines()
        for sample in data:
            res_data.append(sample)
    print('get data success!')
    return res_data

def get_args():
    parser = argparse.ArgumentParser()

    # 添加命令行参数
    parser.add_argument('--vocab_size', type=int, default=21128)
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    return args

class AttenDataset(Dataset):
    def __init__(self, data_path):
        self.data = get_data(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

class RNNLanguageModel(nn.Module):
    def __init__(self, args, embedding_dim, hidden_dim):
        super(RNNLanguageModel, self).__init__()
        self.args = args
        self.embeddings = nn.Embedding(self.args.vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, self.args.vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        output, hidden = self.rnn(embeds)
        output = self.linear(output)
        return output
        # return self.loss_fn(log_probs[:, :-1].transpose(-1, -2), inputs[:, 1:])

class Trainer():
    def __init__(self, args, embed_dim, head_dim):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('/users/nishiyu/ict/Models/bart-base-chinese')
        self.model = RNNLanguageModel(args, embed_dim, head_dim)
        self.train_dataset = AttenDataset(self.args.train_data)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)
        self.test_dataset = AttenDataset(self.args.test_data)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=False)

    def train(self):
        self.model.to(self.args.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr=5e-5)
        for epoch in range(args.epoch):
            self.model.train()
            total_loss = 0.0
            for step, batch in enumerate(self.train_dataloader):
                tokens = self.tokenizer(batch, truncation=True, padding=True, max_length=self.args.max_len, return_tensors='pt').to(self.args.device)
                input_ids = tokens['input_ids'][:, :-1]

                label_ids = tokens['input_ids'][:, 1:].clone()
                pad_token_id = self.tokenizer.pad_token_id
                label_ids[label_ids == pad_token_id] = -100 

                logits = self.model(input_ids)
                loss = criterion(logits.view(-1, self.args.vocab_size), label_ids.view(-1))
                loss.backward()
                optimizer.step()
                self.model.zero_grad()

                total_loss += loss
            print(f'epoch: {epoch}, train loss: {total_loss / len(self.train_dataloader)}')
            self.evaluation()
            
    def evaluation(self):
        self.model.eval()
        self.model.to(self.args.device)
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        for step, batch in enumerate(self.test_dataloader):
            tokens = self.tokenizer(batch, truncation=True, padding=True, max_length=self.args.max_len, return_tensors='pt').to(self.args.device)
            input_ids = tokens['input_ids'][:, :-1]
            label_ids = tokens['input_ids'][:, 1:].clone()
            pad_token_id = self.tokenizer.pad_token_id
            label_ids[label_ids == pad_token_id] = -100 

            logits = self.model(input_ids)
            loss = criterion(logits.view(-1, self.args.vocab_size), label_ids.view(-1))

            total_loss += loss
        print(f'test loss: {total_loss / len(self.test_dataloader)}')

if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args, 512, 128)
    trainer.train()
