# encoding utf-8
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

class SelfAttention(nn.Module):
    """Causal SelfAttention Layer"""

    def __init__(
        self,
        args,
        embed_dim: int,
        num_heads: int,
        bias = True
    ):
        super(SelfAttention, self).__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        """Input shape: Batch x seq_len x dim"""
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
       
        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz) # bsz, heads, seq_len, dim
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape) # bsz_heads, seq_len, dim
        # print(query_states)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2)) # bsz*head_num, tgt_len, src_len

        src_len = key_states.size(1)
        # causal_mask
        mask_value = torch.finfo(attn_weights.dtype).min
        matrix = torch.ones(bsz * self.num_heads, src_len, tgt_len).to(self.args.device)
        causal_mask = torch.triu(matrix, diagonal=1)
        causal_weights = torch.where(causal_mask.byte(), mask_value, causal_mask.double())
        attn_weights += causal_weights


        # do not need attn_mask
        attn_probs = nn.functional.softmax(attn_weights, dim=-1)

        # get output
        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output

class AttentionModel(nn.Module):
    def __init__(self, args, embed_dim, head_num):
        super(AttentionModel, self).__init__()
        self.args = args
        self.embeddings = nn.Embedding(self.args.vocab_size, embed_dim)
        self.p_embeddings = nn.Embedding(self.args.max_len, embed_dim)
        self.attention = SelfAttention(self.args, embed_dim, head_num)
        self.output = nn.Linear(embed_dim, self.args.vocab_size)
    
    def forward(self, input_ids, attn_mask):
        embeddings = self.embeddings(input_ids)
        position_embeddings = self.p_embeddings(torch.arange(0, input_ids.shape[1], device=self.args.device))
        embeddings = embeddings + position_embeddings
        output = self.attention(embeddings, attn_mask)
        logits = self.output(output)
        return logits

class Trainer():
    def __init__(self, args, embed_dim, head_dim):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('/users/nishiyu/ict/Models/bart-base-chinese')
        self.model = AttentionModel(args, embed_dim, head_dim)
        self.train_dataset = AttenDataset(self.args.train_data)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)
        self.test_dataset = AttenDataset(self.args.test_data)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=False)

    def train(self):
        self.model.train()
        self.model.to(self.args.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr=5e-5)
        for epoch in range(args.epoch):
            total_loss = 0.0
            for step, batch in enumerate(self.train_dataloader):
                tokens = self.tokenizer(batch, truncation=True, padding=True, max_length=self.args.max_len, return_tensors='pt').to(self.args.device)
                input_ids = tokens['input_ids'][:, :-1]
            
                attn_mask = tokens['attention_mask'][:, :-1].clone()

                label_ids = tokens['input_ids'][:, 1:].clone()
                pad_token_id = self.tokenizer.pad_token_id
                label_ids[label_ids == pad_token_id] = -100 

                logits = self.model(input_ids, attn_mask)
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
        
            attn_mask = tokens['attention_mask'][:, :-1].clone()

            label_ids = tokens['input_ids'][:, 1:].clone()
            pad_token_id = self.tokenizer.pad_token_id
            label_ids[label_ids == pad_token_id] = -100 

            logits = self.model(input_ids, attn_mask)
            loss = criterion(logits.view(-1, self.args.vocab_size), label_ids.view(-1))

            total_loss += loss
        print(f'test loss: {total_loss / len(self.test_dataloader)}')

if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args, 512, 8)
    trainer.train()


