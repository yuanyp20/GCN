import torch
import torch.nn as nn
from torch.nn import init
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
# torch.cuda.set_device(1)
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

torch.set_default_dtype(torch.float64)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.num_layers = 1
        self.hidden_size = 32
        self.feature_dim = 8
        self.conv1 = GCNConv(1, self.feature_dim)
        # self.linear1 = nn.Linear(self.feature_dim, 2)
        self.lstm = nn.LSTM(input_size=128, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(32, 1)

        for name, param in self.lstm.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)
        for name, param in self.conv1.named_parameters():
            if name.endswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)

        init.xavier_uniform_(self.linear2.weight, gain=1)
        # init.xavier_uniform_(self.linear3.weight, gain=1)

    def forward(self, data, hidden):
        input, edge_index = data.x, data.edge_index

        input = torch.unsqueeze(input, 3)
        input = torch.transpose(input, 1, 2)

        x = input

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = x.reshape(32, 10, -1)

        output, hidden = self.lstm(x, hidden)
        output = F.relu(output)
        output = self.dropout(output)
        output = output[:, -1, :]
        output = self.linear2(output)

        output = F.sigmoid(output)

        return output

    def init_hidden(self, batch_size):
        hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                  torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

        return hidden
