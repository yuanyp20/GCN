import torch
import torch.nn as nn
from torch.nn import init

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'


class Net(nn.Module):
    def __init__(self, num_class=3, bidirectional=False):
        super(Net, self).__init__()  # 调用父类的构造方法

        self.num_layers = 1
        self.hidden_size = 20
        self.bidirectional = bidirectional
        self.encoder = nn.LSTM(input_size=16, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True,
                               bidirectional=self.bidirectional)
        
        for name, param in self.encoder.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)
        if self.bidirectional:
            self.linear = nn.Linear(self.hidden_size * 2, 3)
        else:
            self.linear = nn.Linear(self.hidden_size, 3)
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=0.5)

        init.xavier_uniform_(self.linear.weight, gain=1)

    def forward(self, input, hidden):
        '''
        input shape : (win_len, batch_size, enose_dim)
        '''
        input = input
        output, hidden = self.encoder(input, hidden)
        output = self.relu(output)
        output = self.dropout(output)
        '''
        the first usage method of output
        '''

        RO = self.linear(output[:, -1, :])

        RO = self.softmax(RO)
        # return idx_class, RO
        return RO, hidden

        '''
        the second usage method of output
        '''
        # win_len, batch_size, hidden_size = output.size()
        # output = output.view(win_len * batch_size, hidden_size)
        # output = self.predictor(output)
        # output = output.view(win_len, batch_size, -1)
        # return output
        # output.shape :win_len,batch_size,num_class

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        # print("weghit :", weight.shape)
        if self.bidirectional:
            hidden = (weight.new(self.num_layers * 2, batch_size, self.hidden_size).zero_().to(device),
                      weight.new(self.num_layers * 2, batch_size, self.hidden_size).zero_().to(device))
        else:
            hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device),
                      weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device))
        return hidden

# # input shape : (win_len, batch_size, enose_dim)
# data = torch.randn(10, 8, 16).to(device)
# lstm = simple_LSTM(hidden_size=32, enose_dim=16, num_layers=2)
# softmax_out, preds_class, idx_class = lstm(data)
# print(softmax_out.shape)
