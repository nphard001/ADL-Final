from __future__ import print_function
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel
from src.bert_util import InputFeatures, convert_examples_to_features
from src.encoder_layers import FeedForward, Norm, PositionalEncoder, EncoderLayer, Encoder


class ResponseEncoder(nn.Module):
    def __init__(self, num_emb, hid_dim=256, out_dim=256, max_len=16, bert_dim=768):
        super(ResponseEncoder, self).__init__()
        self.num_emb = num_emb
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.max_len = max_len
        self.bert_dim = bert_dim

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.img_linear = nn.Linear(self.hid_dim, self.out_dim, bias=True)
        self.txt_linear = nn.Linear(self.bert_dim, self.out_dim, bias=False)

        # Initialize bert model for encoding text
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        # If you have the memory size uncomment this...
        self.bert_model = self.bert_model.to(self.device)

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

    def encode_text(self, text_input):
        """
        Given a list of natural language text, returns the encoding from BERT
        :param text_input: size N list of text responses 
        :return: N(batch_size) * M(out_dim) tensor
        
        TODO : use a finetuned bert model instead to obtain better word representations
        """
        features = convert_examples_to_features(text_input, self.max_len, self.tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(self.device)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(self.device)
        _, pooled_output = self.bert_model(all_input_ids, token_type_ids=None, attention_mask=all_input_mask)
        text_rep = self.txt_linear(pooled_output).to(self.device)
        return text_rep

    def forward(self, img_rep, text):
        text_rep = self.encode_text(text)
        return img_rep + text_rep


class StateTracker(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim, num_encoder=1, num_heads=1):
        super(StateTracker, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_encoder = num_encoder
        self.num_heads = num_heads
        self.history_rep = None

        self.encoder = Encoder(self.input_dim, self.num_encoder, self.num_heads)

    def forward(self, response_rep, history_rep=None):
        """

        :param response_rep: N, M
        :param history_rep: N, k-1, M  if k==0 history_rep = None
        :return:
                :current_state: N, k , M
                :history_rep: N, k , M
        """"""
        concat(res_rep_0 ~ res_rep_i) -> shape N, i, input_dim

        Transformer input szie N, seq_len, input_dim

        """
        if history_rep is not None:
            history_rep = torch.cat((history_rep, response_rep.unsqueeze(1)), 1)
        else:
            history_rep = response_rep.unsqueeze(1)

        x = history_rep.permute(1, 0, 2)  # seq_len , N , input_dim

        x = self.encoder(x)
        x = torch.max(x, 1)[0].squeeze()

        return x, self.history_rep
