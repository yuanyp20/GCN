import torch
import torch.nn as nn
from torch.nn import init

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'


class Net(nn.Module):
    def __init__(self, enose_dim=16, num_class=3):
        super(Net, self).__init__()  # 调用父类的构造方法

        self.encoder = nn.Conv1d(in_channels=enose_dim,out_channels=1,kernel_size=2)
        self.classifier = nn.Linear(9, num_class)
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        init.xavier_uniform_(self.encoder.weight, gain=1)
        init.xavier_uniform_(self.classifier.weight, gain=1)
    #去掉hidden
    def forward(self, input):
        '''
        input shape : (win_len, batch_size, enose_dim)
        '''
        input = input.permute(0,2,1)
        output = self.encoder(input)

        RO = self.relu(output)
        RO = self.dropout(RO)

        RO = self.classifier(RO[:, -1, :])
        RO = self.softmax(RO)
        # return idx_class, RO
        return RO

