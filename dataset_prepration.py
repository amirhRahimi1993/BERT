import itertools
import math
import os.path
from random import random
from random import randint
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
from glob import glob
from torch.utils.data import Dataset, DataLoader
from typing import List


class convert_text_to_dictionary:
    def __init__(self,movie_lines_path,movie_conversations_path,MAX_LEN=64):
        self.MAX_LEN= MAX_LEN
        self.movie_lines_path =movie_lines_path
        self.movie_conversations_path = movie_conversations_path
        self.pairs = []
        self.movie_lines = {}
        self.movie_conversations = {}
    def __load_movie_lines(self):
        with open(self.movie_lines_path,'r',encoding='iso-8859-1') as f:
            for line in tqdm(f.readlines()):
                self.movie_lines[line.split(" +++$+++ ")[0]] = line.split("+++$+++")[-1]
    def __load_movie_conversations(self):

        with open(self.movie_conversations_path,'r',encoding='iso-8859-1') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                id = eval(line.split(' +++$+++ ')[-1])
                for index,value in enumerate(id):
                    if index > len(id)-2:
                        break
                    first= self.movie_lines[id[index]].strip()
                    second = self.movie_lines[id[index+1]].strip()
                    self.pairs.append((' '.join(first.split()[:self.MAX_LEN]),' '.join(second.split()[:self.MAX_LEN])))

    def load_movie_lines(self):
        self.__load_movie_lines()
        self.__load_movie_conversations()
        return self.pairs,self.movie_lines

class save_data_into_txt:
    def __init__(self,folder_name,pairs,movie_lines,each_file_length = 10000):
        if os.path.exists(folder_name) == False:
            os.makedirs(folder_name)
        self.folder_name = folder_name
        self.pairs = pairs
        self.movie_lines = movie_lines
        self.each_file_length = each_file_length
    def __writer(self,pair_storer,counter):
        with open(os.path.join(self.folder_name, "pair_" + str(counter) + ".txt"), "w", encoding='utf-8') as f:
            f.write('\n'.join(pair_storer))
    def __save_data_to_txt(self):
        counter= 1
        pair_storer = []
        for sample in tqdm(self.pairs):
            pair_storer.append(sample[0])
            pair_storer.append(sample[1])
            if len(pair_storer) % self.each_file_length == 0 :
                self.__writer(pair_storer,counter)
                pair_storer.clear()
                counter+=1
        if len(pair_storer) > 0:
            self.__writer(pair_storer,counter)
        return list(glob(os.path.join(self.folder_name, "*.txt")))
    def convert_to_tokens(self):
        paths = self.__save_data_to_txt()
        tokenizer= BertWordPieceTokenizer(clean_text= True,handle_chinese_chars= False,strip_accents=False,lowercase= True)
        tokenizer.train(
            files= paths,
            vocab_size=30_000,
            min_frequency= 5,
            limit_alphabet= 1000,
            wordpieces_prefix= '##',
            special_tokens= ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
        )
        if os.path.exists('bert-it-1') == False:
            os.mkdir('bert-it-1')
        tokenizer.save_model('bert-it-1','bert-it')
        tokenizer = BertTokenizer.from_pretrained('bert-it-1/bert-it-vocab.txt', local_files_only= True)
        return tokenizer

class BERTDataset(Dataset):
    def __init__(self,pairs:List,tokenizer:BertTokenizer,seq_len:int=64):
        self.pairs = pairs
        self.tokenizers = tokenizer
        self.seq_len = seq_len
        self.__length__ = len(pairs)
        self.__random_mask("Hi my name is amirhossein")
    def __len__(self):
        return self.__length__
    def __getitem__(self, idx):
        sent1,sent2,is_next_sentence = self.__nsp(idx)
        randomize_sent1,sent1_mask_lbl= self.__random_mask(sent1)
        randomize_sent2, sent2_mask_lbl= self.__random_mask(sent2)
        randomize_sentnces1_tokens = [self.tokenizers.vocab["[CLS]"]]+randomize_sent1+[self.tokenizers.vocab["[SEP]"]]
        remain_length= self.seq_len-len(randomize_sentnces1_tokens)-1 # it must greater then 5
        MAX_LENGTH= len(randomize_sent2) if remain_length>len(randomize_sent2) else len(randomize_sent2)
        randomize_sentnces2_tokens = randomize_sent2[:MAX_LENGTH] + [self.tokenizers.vocab["[SEP]"]]
        sent1_mask_lbl = [self.tokenizers.vocab["[PAD]"]] + sent1_mask_lbl + [self.tokenizers.vocab["[PAD]"]]
        sent2_mask_lbl = sent2_mask_lbl[:MAX_LENGTH] + [self.tokenizers.vocab["[PAD]"]]

        segment_lbl = [1 for _ in range(len(randomize_sentnces1_tokens))] + [2 for _ in range(len(randomize_sentnces1_tokens))]
        sentence = randomize_sentnces1_tokens + randomize_sentnces2_tokens
        sentence_label = sent1_mask_lbl + sent2_mask_lbl
        Remain_LENGTH =  self.seq_len-len(segment_lbl) if len(segment_lbl) < self.seq_len else 0
        Padder= [self.tokenizers.pad_token_id for _ in range(Remain_LENGTH)]
        segment_lbl.extend(Padder) , sentence.extend(Padder) , segment_lbl.extend(Padder) , sentence_label.extend(Padder)
        return {"bert_input": torch.tensor(sentence),
                "bert_label": torch.tensor(sentence_label),
                "segment_label": torch.tensor(segment_lbl),
                "is_next_sentence": torch.tensor(is_next_sentence)
                }

    def __nsp(self,index:int):
        random_index = randint(0,self.__length__-1)
        if random() < 0.5:
            return self.pairs[index][0] , self.pairs[index][1] , 1
        else:
            return self.pairs[index][0] , self.pairs[random_index][1] , 0

    def __random_mask(self,sentence:str):
        tokens = sentence.split()
        output_lbl = []
        output_id = []
        for token in tokens:
            is_mask = random()
            input_token = self.tokenizers(token)["input_ids"][1:-1]
            MASK = self.tokenizers.vocab['[MASK]']
            if is_mask < 0.85:
                for _ in input_token:
                    output_lbl.append(0)
                output_id.append(input_token)
            elif is_mask > 0.85:
                is_mask/= 0.85
                if is_mask <0.8:
                    for _ in input_token:
                        output_id.append(MASK)
                elif is_mask < 0.9:
                    output_lbl.append(random.randrange(self.tokenizers.vocab_size))
                else:
                    output_lbl.append(input_token)
                output_id.append(input_token)
        output_id = list(itertools.chain(*[x if isinstance(x,list) else [x] for x in output_id]))
        output_lbl = list(itertools.chain(*[x if isinstance(x, list) else [x] for x in output_lbl]))
        assert len(output_id) == len(output_lbl)
        return output_id, output_lbl

class positional_embedding(nn.Module):
    def __init__(self,d_model,max_len = 128,n= 10_000):
        super().__init__()
        self.pos= torch.zeros(max_len,d_model).float()
        self.pos.requires_grad = False
        for i in range(max_len):
            for j in range(0,d_model,2):
                self.pos[i,j] = math.sin(i / n ** (2 * j / d_model))
                self.pos[i,j+1] = math.cos(i / n ** (2 * (j+1) / d_model))
        self.pos = self.pos.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return self.pos[:, :seq_len, :]

class embedder(nn.Module):
    def __init__(self,vocab_size,max_len,d_model,dropout = 0.1):
        super().__init__()
        self.tokenizer= torch.nn.Embedding(vocab_size,d_model,padding_idx=0).float()
        self.segment = torch.nn.Embedding(3,d_model,padding_idx=0).float()
        self.positional = positional_embedding(d_model,max_len) # torch.nn.Embedding(max_len,d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self,sequence,segment_label):
        x= self.tokenizer(sequence) + self.segment(segment_label) + self.positional(sequence)
        x= self.dropout(x)
        return x

class multihead_attention(nn.Module):
    def __init__(self,d_model,num_heads,dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q= nn.Linear(d_model,d_model)
        self.k= nn.Linear(d_model,d_model)
        self.v= nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model,d_model)

    def forward(self,x):

        query = self.q(x)
        key = self.k(x)
        value = self.v(x)
        query= query.view(query.size(0),-1,self.num_heads,self.head_dim).permute(0,2,1,3)
        key = key.view(key.size(0),-1,self.num_heads,self.head_dim).permute(0,2,1,3)
        value= value.view(value.size(0),-1,self.num_heads,self.head_dim).permute(0,2,1,3)
        score = torch.matmul(query,key.transpose(-2))/torch.sqrt_(value.shape[-1])
        score = torch.softmax(score)
        output = torch.matmul(score,value)
        output = self.dropout(output)
        output= output.permute(0,2,1,3).contiguous()
        output = output.view(output.size(0),-1,self.num_heads*self.head_dim)

        return self.output(output)

class FeedForward(nn.Module):
    def __init__(self
                 ,d_model=768,middle_layer = 3072,dropout= 0.1):
        super().__init__()
        self.layers= nn.Sequential(
            nn.Linear(d_model,middle_layer),
            nn.ReLU(),
            nn.Linear(middle_layer,d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.layers(x)

class Encoder(nn.Module):
    def __init__(self,d_model= 768,head_size= 12, dropout= 0.1,middle_layer_coef=4):
        super().__init__()
        self.norm_layer= nn.LayerNorm(d_model)
        self.dropout= nn.Dropout(dropout)
        self.multi_head = multihead_attention(d_model,head_size)
        self.layers= FeedForward(d_model=d_model,middle_layer=d_model * middle_layer_coef,dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        self.residual = x.clone()
        x = self.multi_head(x)
        x += self.residual
        x= self.dropout(x)
        x = self.norm_layer(x)
        self.residual = x.clone()
        x= self.layers(x)
        x= self.dropout(x)
        x+= self.residual
        x= self.norm_layer(x)
        return x

class BERT(nn.Module):
    def __init__(self,vocab_size=30_000,max_len=128,d_model= 768,head_size=12,number_of_layer = 12,middle_layer_coef=4):
        super().__init__()
        self.embder = Embedder(vocab_size,max_len,d_model)
        self.encoder_layer = nn.ModuleList([Encoder(d_model=d_model,head_size=head_size,middle_layer_coef=middle_layer_coef) for _ in range(number_of_layer)])

    def forward(self,x,segment_info):
        x = self.embder(x,segment_info)
        for encoder in self.encoder_layer:
            x= encoder(x)
        return x

if __name__ == '__main__':
    vocab_size = 30522  # Example vocabulary size for BERT
    max_len = 128
    d_model = 768

    Embedder = embedder(vocab_size, max_len, d_model)

    # Create example input data
    batch_size = 1
    seq_len = 10

    # Example sequence of token IDs (randomly generated for illustration)
    sequence = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Example segment labels (all 0s for a single sentence, or 0s and 1s for sentence pairs)
    segment_label = torch.zeros(batch_size, seq_len).long()  # Single sentence example

    # Get the embeddings
    embeddings = Embedder(sequence, segment_label)
    print(embeddings.shape)  # Should be (batch_size, seq_len, d_model)