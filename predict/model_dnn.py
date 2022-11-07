import torch
import torch.nn as nn
from torch.nn import init

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'


class Net(nn.Module):
    def __init__(self, enose_dim=16, num_class=3):
        super(Net, self).__init__()  # 调用父类的构造方法

        self.linear64 = nn.Linear(160, 64)

        self.predictor = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        init.xavier_uniform_(self.linear64.weight, gain=1)
        init.xavier_uniform_(self.predictor.weight, gain=1)

    def forward(self, input):
        '''
        input shape : (win_len, batch_size, enose_dim)
        '''
        input = input.reshape(input.size(0),-1)

        # # print('softmax_out shape : ', sigmoid_out.shape) # torch.Size([batch_size, 3])
        # preds_class, idx_class = softmax_out.max(1)

        # RO = self.predictor(output[:, -1, :])

        RO = self.linear64(input)
        
        RO = self.relu(RO)
        RO = self.dropout(RO)
        RO = self.predictor(RO)
        RO = self.sigmoid(RO)
        # return idx_class, RO
        return RO

