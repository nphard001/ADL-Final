from __future__ import print_function
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel
from src.bert_util import InputFeatures, convert_examples_to_features
import pdb

import sys
import inspect


class ResponseEncoder(nn.Module):
    def __init__(self, num_emb, hid_dim=256, out_dim=256, max_len=16, bert_dim=768):
        super(ResponseEncoder, self).__init__()
        self.num_emb = num_emb
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.max_len = max_len
        self.bert_dim = bert_dim

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        #self.emb_txt = torch.nn.Embedding(num_embeddings=self.num_emb, embedding_dim=self.hid_dim * 2)
        #self.bn2 = nn.BatchNorm1d(self.hid_dim * 2)
        #self.cnn_txt = torch.nn.Conv1d(self.max_len, self.hid_dim * 2, 2, bias=True)
        #self.fc_txt = nn.Linear(self.hid_dim * 2, self.hid_dim, bias=False)
        self.img_linear = nn.Linear(self.hid_dim, self.out_dim, bias=True)
        self.txt_linear = nn.Linear(self.bert_dim, self.out_dim, bias=False)
        
        # Initialize bert model for encoding text
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        
        # If you have the memory size uncomment this...
        self.bert_model = self.bert_model.to(self.device)
        
        self.bert_model.eval()
        # Use bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    # fine-tuning the history tracker and policy part
    # TODO: NO idea of why
    def set_rl_mode(self):
        self.train()
        for param in self.img_linear.parameters():
            param.requires_grad = False

    # TODO: NO idea of why
    def clear_rl_mode(self):
        for param in self.img_linear.parameters():
            param.requires_grad = True

    def encode_image(self, image_input):
        return self.img_linear(image_input)

#     def encode_text(self, text_input):
#         x = self.emb_txt(text_input)
#         x = self.cnn_txt(x)
#         x, _ = torch.max(x, dim=2)
#         # x = x.squeeze()
#         x = self.fc_txt(self.bn2(x))
#         return x
    
    def encode_text(self, text_input):
        """
        Given a list of natural language text, returns the encoding from BERT
        :param text_input: size N list of text responses 
        :return: N(batch_size) * M(out_dim) tensor
        
        TODO : use a finetuned bert model instead to obtain better word representations
        """
        features = convert_examples_to_features(text_input,self.max_len,self.tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(self.device)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(self.device)
        _, pooled_output = self.bert_model(all_input_ids, token_type_ids=None, attention_mask=all_input_mask)
        text_rep = self.txt_linear(pooled_output).to(self.device)
        return text_rep
        

    def forward(self, img_rep, text):
        text_rep = self.encode_text(text)
        return img_rep + text_rep


class StateTracker(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super(StateTracker, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.fc_joint = nn.Linear(self.input_dim, self.hid_dim, bias=False)
        self.rnn = nn.GRUCell(self.hid_dim, self.hid_dim, bias=False)
        self.head = nn.Linear(self.hid_dim, self.out_dim, bias=False)
        self.history_rep = None

    def forward(self, response_rep, history_rep=None):
        """
        
        :param response_rep: N, M
        :param history_rep: N, k-1, M  if k==0 history_rep = None
        :return: 
        """"""
        concat(res_rep_0 ~ res_rep_i) -> shape N, i, input_dim
        
        Transformer input szie N, seq_len, input_dim
        
        """
        # x shape: N, M
        # x[0]:
        # history_rep: size(N, k-1, M) if k==0 histroy_rep = None
        # so your transformer encoder input size would be (N, k, M)

        # if history_rep is not None:
        #     history_rep = torch.cat((history_rep, x.unsqueeze(1)), 1)
        # else:
        #     pass

        x = self.fc_joint(response_rep)
        if history_rep is None:
            if self.history_rep is None:
                self.history_rep = self.rnn(x)
            else:
                self.history_rep = self.rnn(x, self.history_rep)
        else:
            self.history_rep = self.rnn(x, history_rep)
        x = self.head(self.history_rep)
        return x, self.history_rep

