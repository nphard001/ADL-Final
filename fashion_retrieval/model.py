from __future__ import print_function
import torch
import torch.nn as nn


class NetSynUser(nn.Module):
    def __init__(self, num_emb, hid_dim=256, txt_len=16):
        super(NetSynUser, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hid_dim = hid_dim
        self.rep_dim = hid_dim

        self.emb_txt = torch.nn.Embedding(num_embeddings=num_emb, embedding_dim=hid_dim * 2 )
        self.bn2 = nn.BatchNorm1d(num_features=hid_dim * 2)
        self.cnn_txt = torch.nn.Conv1d(txt_len, hid_dim * 2, 2, bias=True)
        self.fc_txt = nn.Linear(in_features=hid_dim * 2, out_features=hid_dim, bias=False)
        self.img_linear = nn.Linear(in_features=256, out_features=hid_dim, bias=True)

        self.fc_joint = nn.Linear(in_features=hid_dim, out_features=hid_dim, bias=False)
        self.rnn = nn.GRUCell(hid_dim, hid_dim, bias=False)
        self.head = nn.Linear(in_features=hid_dim, out_features=hid_dim, bias=False)

    # fine-tuning the history tracker and policy part
    def set_rl_mode(self):
        self.train()
        for param in self.img_linear.parameters():
            param.requires_grad = False

    def clear_rl_mode(self):
        for param in self.img_linear.parameters():
            param.requires_grad = True

    def forward_image(self, image_input):
        return self.img_linear(image_input)

    def forward_text(self, text_input):
        x = self.emb_txt(text_input)
        x = self.cnn_txt(x)
        x, _ = torch.max(x, dim=2)
        x = x.squeeze()
        x = self.fc_txt(self.bn2(x))
        return x

    def forward(self, img_input, txt_input):
        x1 = self.forward_image(img_input)
        x2 = self.forward_text(txt_input)
        x = x1 + x2
        x = self.fc_joint(x)
        self.hx = self.rnn(x, self.hx)
        x = self.head(self.hx)
        return x

    def merge_forward(self, img_emb, txt_input):
        x2 = self.forward_text(txt_input)
        x = img_emb + x2
        x = self.fc_joint(x)
        self.hx = self.rnn(x, self.hx)
        x = self.head(self.hx)
        return x

    def init_hid(self, batch_size):
        self.hx = torch.zeros(batch_size, self.hid_dim, device=self.device)


class ResponseEncoder(nn.Module):
    def __init__(self, num_emb, hid_dim=256, txt_len=16):
        super(ResponseEncoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hid_dim = hid_dim

        self.emb_txt = torch.nn.Embedding(num_embeddings=num_emb, embedding_dim=hid_dim * 2 )
        self.bn2 = nn.BatchNorm1d(num_features=hid_dim * 2)
        self.cnn_txt = torch.nn.Conv1d(txt_len, hid_dim * 2, 2, bias=True)
        self.fc_txt = nn.Linear(in_features=hid_dim * 2, out_features=hid_dim, bias=False)
        self.img_linear = nn.Linear(in_features=256, out_features=hid_dim, bias=True)

    # fine-tuning the history tracker and policy part
    def set_rl_mode(self):
        self.train()
        for param in self.img_linear.parameters():
            param.requires_grad = False

    def clear_rl_mode(self):
        for param in self.img_linear.parameters():
            param.requires_grad = True

    def encode_image(self, image_input):
        if image_input is None:
            return None

        return self.img_linear(image_input)

    def encode_text(self, text_input):
        if text_input is None:
            return None

        x = self.emb_txt(text_input)
        x = self.cnn_txt(x)
        x, _ = torch.max(x, dim=2)
        x = x.squeeze()
        x = self.fc_txt(self.bn2(x))
        return x

    def forward(self, image=None, text=None):
        return self.forward_image(image), self.forward_text(text)


class StateTracker(nn.Module):
    def __init__(self, hid_dim=256):
        super(StateTracker, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hid_dim = hid_dim

        self.fc_joint = nn.Linear(in_features=hid_dim, out_features=hid_dim, bias=False)
        self.rnn = nn.GRUCell(hid_dim, hid_dim, bias=False)
        self.head = nn.Linear(in_features=hid_dim, out_features=hid_dim, bias=False)

    def forward(self, response_rep):
        x = self.fc_joint(response_rep)
        self.hx = self.rnn(x, self.hx)
        x = self.head(self.hx)
        return x

    def init_hid(self, batch_size):
        self.hx = torch.zeros(batch_size, self.hid_dim, device=self.device)
