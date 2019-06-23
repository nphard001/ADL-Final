from __future__ import print_function
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel
from src.bert_util import InputFeatures, convert_examples_to_features
from src.encoder_layers import Encoder


class ResponseEncoder(nn.Module):
    def __init__(self, num_emb, hid_dim=256, out_dim=256, max_len=16, bert_dim=768, embedding=None):
        super(ResponseEncoder, self).__init__()
        # self.num_emb = num_emb
        # self.hid_dim = hid_dim
        # self.out_dim = out_dim
        # self.max_len = max_len
        # self.bert_dim = bert_dim
        #
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #
        # self.img_linear = nn.Linear(self.hid_dim, self.out_dim, bias=True)
        # self.txt_linear = nn.Linear(self.bert_dim, self.out_dim, bias=False)
        #
        # self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        # self.bert_model = self.bert_model.to(self.device)
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.hid_dim = hid_dim
        self.rep_dim = hid_dim
        self.out_dim = hid_dim

        self.emb_txt = torch.nn.Embedding(embedding.size(0),
                                          embedding.size(1))
        self.emb_txt.weight = torch.nn.Parameter(embedding)
        self.emb_fc = nn.Linear(in_features=embedding.size(1), out_features=hid_dim, bias=True)

        # self.emb_txt = torch.nn.Embedding(num_embeddings=num_emb, embedding_dim=hid_dim * 2 )
        self.bn2 = nn.BatchNorm1d(num_features=hid_dim * 2)
        self.cnn_txt = torch.nn.Conv1d(max_len, hid_dim * 2, 2, bias=True)
        self.fc_txt = nn.Linear(in_features=hid_dim * 2, out_features=hid_dim, bias=False)
        self.img_linear = nn.Linear(in_features=256, out_features=hid_dim, bias=True)

    def set_rl_mode(self):
        self.train()
        for param in self.img_linear.parameters():
            param.requires_grad = False

    def clear_rl_mode(self):
        for param in self.img_linear.parameters():
            param.requires_grad = True

    def encode_image(self, image_input):
        return self.img_linear(image_input)

    def encode_text(self, text_input):
        """
        Given a list of natural language text, returns the encoding from BERT
        :param text_input: size N list of text responses
        :return: N(batch_size) * M(out_dim) tensor

        TODO : use a finetuned bert model instead to obtain better word representations
        """
        # features = convert_examples_to_features(text_input, self.max_len, self.tokenizer)
        # all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(self.device)
        # all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(self.device)
        # _, pooled_output = self.bert_model(all_input_ids, token_type_ids=None, attention_mask=all_input_mask)
        # text_rep = self.txt_linear(pooled_output).to(self.device)
        # return text_rep
        with torch.no_grad():
            x = self.emb_txt(text_input)
        x = self.emb_fc(x)
        x = self.cnn_txt(x)
        x, _ = torch.max(x, dim=2)
        # x = x.squeeze(2)
        x = self.fc_txt(self.bn2(x))
        return x

    def forward(self, img_rep, text):
        x2 = self.encode_text(text)
        x = (img_rep + x2) / 2
        return x
        # text_rep = self.encode_text(text)
        # return img_rep + text_rep


class StateTracker(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim, num_encoder=4, num_heads=2):
        super(StateTracker, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_encoder = num_encoder
        self.num_heads = num_heads

        self.encoder = Encoder(self.input_dim, self.num_encoder, self.num_heads)
        self.state_project = nn.Linear(self.out_dim, self.out_dim, False)

        self.history_rep = None

    def forward(self, response_rep, history_rep=None):
        """

        :param response_rep: N, M
        :param history_rep: N, k-1, M  if k==0 history_rep = None
        :return:
                :current_state: N, k , M
                :history_rep: N, k , M

        concat(res_rep_0 ~ res_rep_i) -> shape N, i, input_dim

        Transformer input size N, seq_len, input_dim

        """
        # 1 , N , input_dim
        response_rep = response_rep.unsqueeze(0)
        # seq_len , N , input_dim
        history_rep = torch.cat((history_rep, response_rep), 0) if history_rep is not None else response_rep
        self.history_rep = history_rep

        # seq_len , N , input_dim
        x = self.encoder(history_rep)
        # N , input_dim
        x = torch.max(x, dim=0)[0]
        x = self.state_project(x)

        return x, history_rep


class Reconstruct(nn.Module):
    def __init__(self, dim):
        super(Reconstruct, self).__init__()
        self.fc = nn.Linear(dim, 10000)

    def forward(self, x):
        x = self.fc(x)
        return x
