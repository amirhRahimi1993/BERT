import torch
import torch.nn as nn

class BERT_EMBEDDING(nn.Module):
    def __init__(self,segment_size,hidden_size,vocab_size,dropout,sentence_size):
        super().__init__()
        self.token_embedding= nn.Embedding(vocab_size,hidden_size)
        self.position_embedding= nn.Embedding(sentence_size,hidden_size)
        self.segment_embedding= nn.Embedding(segment_size,hidden_size)
        self.dropout= nn.Dropout(dropout)
        self.postion = torch.tensor([i for i in range(sentence_size)])

    def forward(self,seq,seg):
        x= self.token_embedding(seq) + self.segment_embedding(seg) + self.position_embedding(self.postion)
        x= self.dropout(x)
        return x

class BERT(nn.Module):
    def __init__(self,segment_size,hidden_size,vocab_size,dropout,sentence_size,attn_head,num_layers,class_number):
        super().__init__()
        self.embedder = BERT_EMBEDDING(segment_size,hidden_size,vocab_size,dropout,sentence_size)
        self.transform_oneblock = nn.TransformerEncoderLayer(hidden_size,attn_head,hidden_size* 4)
        self.transformers = nn.TransformerEncoder(self.transform_oneblock,num_layers)
        self.classifier= nn.Linear(hidden_size,class_number)

    def forward(self,seq,seg):
        x= self.embedder(seq,seg)
        x= self.transformers(x)
        x= x.mean(dim=0)
        x= self.classifier(x)
        return x

if __name__ == '__main__':
    #base implementation
    nn_segment= 3
    nn_token_size= 30000
    nn_embeddim = 768 # in large model is 1024
    nn_dropout= 0.1
    nn_layers= 12 # in large 24
    nn_attn_head= 12 # in large 16
    seqeunce_length = 512
    class_number= 5
    seq= torch.randint(high= nn_token_size,size=[seqeunce_length])
    seg= torch.randint(high= nn_segment,size=[seqeunce_length])
    model = BERT(nn_segment,nn_embeddim,nn_token_size,nn_dropout,seqeunce_length,nn_attn_head,nn_layers,class_number)
    Model_output= model(seq,seg)
    print(Model_output)