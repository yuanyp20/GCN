import torch
import torch.nn as nn
from torch.nn import init
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.feature_dim = 5
        self.conv1 = GCNConv(1, self.feature_dim)
        self.linear = nn.Linear(800, 3)
        self.dropout = torch.nn.Dropout(p=0.5)

        for name, param in self.conv1.named_parameters():
            if name.endswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)
        init.xavier_uniform_(self.linear.weight, gain=1)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = torch.unsqueeze(x, 3)
        x = torch.transpose(x, 1, 2)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = x.reshape(32, -1)
        x = self.linear(x)
        return F.softmax(x, dim=1)
