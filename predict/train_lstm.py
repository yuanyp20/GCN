import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from data_process import list_file_name
import matplotlib.pyplot as plt
import sys
import torch
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from model_lstm import Net
from torch import optim
from tensorboardX import SummaryWriter
from sklearn.metrics import r2_score
from torch_geometric.data import Data
import networkx as nx
import math
G = nx.Graph()
num = 16
nodes = list(range(num))  # [0,1,2,3,4,5]
G.add_nodes_from(nodes)  # 从列表中加点
edges = []  # 存放所有的边，构成无向图（去掉最后一个结点，构成一个环）
for idx in range(num):
    for idy in range(num):
        edges.append((idx, idy))

#  将所有边加入网络
G.add_edges_from(edges)

print(G.nodes())
print(G.edges())
edges_graph = [list(i) for i in G.edges()]
edge_index = torch.tensor(edges_graph, dtype=torch.long)

# seed = 42
# torch.manual_seed(seed) # 为CPU设置随机种子
# torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
# np.random.seed(seed)  # Numpy module.
# random.seed(seed)  # Python random module.
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


pd.set_option('display.max_columns', None)

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
print(device)
# 加载数据集
data5 = r'./HD5_label_two_50'
file_list5 = list_file_name(data5)
file_list5.sort()
print('file_list of HD5:', file_list5)
# data20 = 'D:\oil0107\oil\HD20_label_50'
# file_list20 = list_file_name(data20)

features_train = []
labels_train = []
features_val = []
labels_val = []
features_test = []
labels_test = []


def create_dataset(x, y, seq_len, len_step, features, labels):
    for i in range(0, len(x) - seq_len, len_step):
        data = x[i:i + seq_len]  # 序列数据
        label = y[i + seq_len]  # 标签数据
        # 保存到features和labels
        features.append(data)
        labels.append(label)
    return features, labels


#
# filenames是数据列表，labels是标签列表
class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
train_mean_std = pd.read_csv('trainmerge.csv', engine='python', header=None, index_col=None)
scaler = []
for train_col in range(16):
    scaler.append(MinMaxScaler())
    scaler[train_col].fit(train_mean_std[train_col].values.reshape(-1,1))

file_index = 0
for file in file_list5:
    # 读取一个文件中的数据
    data = pd.read_csv(file, engine='python', header=None)
    # print('init_data.info():', data.info())
    # print('init_data.head():', data.head())
    # print('init_data.shape:', data.shape)
    # print('init_data.describe():', data.describe())

    # # 可视化特征与标签的关系
    # plt.figure(figsize=(100, 50))
    # sns.lineplot(data[0], data[16], data=data)
    # plt.show()
    # 归一化，缩放到0-1
    for i in range(16):
        data[i] = scaler[i].fit_transform(data[i].values.reshape(-1,1))
    
# print('scaler_data.head():', data.head())

    # 拆分成x特征和y标签
    x = data.values[:,0:16]
    y = data.values[:,16]
    # print('x.shape:', x.shape)
    # print('y.shape:', y.shape)

    # 加窗后，创建数据集。窗口大小为10,步长为1

    if file_index < 24:
        features_train, labels_train = create_dataset(x, y, seq_len=10, len_step=1, features=features_train,
                                                      labels=labels_train)
    elif file_index < 27:
        features_val, labels_val = create_dataset(x, y, seq_len=10, len_step=1, features=features_val,
                                                  labels=labels_val)
    else:
        features_test, labels_test = create_dataset(x, y, seq_len=10, len_step=1, features=features_test,
                                                    labels=labels_test)
    file_index = file_index + 1

features_train = np.array(features_train)
labels_train = np.array(labels_train)
features_val = np.array(features_val)
labels_val = np.array(labels_val)
features_test = np.array(features_test)
labels_test = np.array(labels_test)

scaler = MinMaxScaler()
scaler.fit(train_mean_std[16].values.reshape(-1,1))
labels_train = scaler.fit_transform(labels_train.reshape(-1, 1))
labels_val = scaler.fit_transform(labels_val.reshape(-1, 1))
labels_test = scaler.fit_transform(labels_test.reshape(-1, 1))
# # 手动打乱数据
# index = [i for i in range(len(labels_train))]
# random.shuffle(index)
# X_train = features_train[index]
# Y_train = labels_train[index]
#
# index = [i for i in range(len(labels_val))]
# random.shuffle(index)
# X_val = features_val[index]
# Y_val = labels_val[index]
#
# index = [i for i in range(len(labels_test))]
# random.shuffle(index)
# X_test = features_test[index]
# Y_test = labels_test[index]
# # 手动打乱数据
# index = [i for i in range(len(labels))]
# random.shuffle(index)
# X = features[index]
# Y = labels[index]

X_train = torch.from_numpy(features_train)
Y_train = torch.from_numpy(labels_train)
X_val = torch.from_numpy(features_val)
Y_val = torch.from_numpy(labels_val)
X_test = torch.from_numpy(features_test)
Y_test = torch.from_numpy(labels_test)

train_dataset = MyDataset(features=X_train, labels=Y_train)
val_dataset = MyDataset(features=X_val, labels=Y_val)
test_dataset = MyDataset(features=X_test, labels=Y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# net = simple_LSTM(hidden_size=32, enose_dim=16, num_layers=2)
net = Net()
print(net)
net.to(device)

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# save_path = './train_3.pth'

train_steps = len(train_loader)
val_steps = len(val_loader)
# writer = SummaryWriter('./train_3')

num_epochs = 100


def train(net, device, train_loader, criterion, optimizer, num_epochs, val_loader, save_path, writer):
    bestmse = 1e6
    for epoch in range(num_epochs):
        # scheduler.step()
        # Train:
        net.train()

        hs = net.init_hidden(batch_size)

        running_loss = 0.0
        train_bar = tqdm(train_loader)

        for step, (train_data, train_label) in enumerate(train_bar):
            # print(train_data.shape)
            # print(train_label.shape)

            # train_data = train_data.float().to(device)  # 部署到device
            # train_label = train_label.long().to(device)
            train_data = train_data.to(device).float()
            train_label = train_label.to(device).float()
            optimizer.zero_grad()  # 梯度置零

            y_hat,hs = net(train_data, hs)  # 模型训练

            # y_hat, hs = net(train_data, hs)  # 模型训练
            # 每一次在读取并计算完一个batch的数据之后会有一个ht连接到计算图中，这个ht参与反向传播求梯度，梯度在求完结果之后就被释放掉了，当到下一个也就是h(t+1)的时候，
            # 在计算梯度时还会经过ht，（因为这两个在计算图中连着），但是ht的相关信息已经被释放了，所以会产生报错。

            hs = tuple([h.data for h in hs])

            # 得到的loss是batch内损失的平均
            loss = criterion(y_hat, train_label)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch [{}/{}] loss :{:.4f}".format(epoch + 1, num_epochs, loss)

        # 模型验证
        net.eval()
        hs = net.init_hidden(batch_size)
        val_loss = 0.0
        val_bar = tqdm(val_loader)
        with torch.no_grad():
            for step, (val_data, val_label) in enumerate(val_bar):
                
                val_data = val_data.to(device).float()
                val_label = val_label.to(device).float()
                val_preds,hs = net(val_data, hs)
                hs = tuple([h.data for h in hs])
                loss = criterion(val_preds, val_label)  # 计算损失
                val_loss += loss.item()

        print(
            f'Epoch {epoch + 1}/{num_epochs} --- train loss {running_loss / train_steps} --- val loss {val_loss / val_steps}'
        )

        if (val_loss / val_steps) < bestmse:
            bestmse = val_loss / val_steps
            torch.save(net.state_dict(), save_path)

        writer.add_scalars('logs', {'train_loss': running_loss / train_steps, 'val_loss': val_loss / val_steps}, epoch)
    writer.close()
    print('finish training')


# train(net, device, train_loader, criterion, optimizer, num_epochs, val_loader)


test_num = len(test_dataset)

predict_result = []


def predict(test_num, save_path, test_loader, device, net, times):
    print("{} data for test".format(test_num))
    print("loading model...")
    test_mse = 0.0
    test_r2 = 0.0
    net.load_state_dict(torch.load(save_path))
    net.eval()
    net.to(device)
    test_steps = len(test_loader)
    hs = net.init_hidden(batch_size)
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for step, (test_data, test_label) in enumerate(test_bar):
            
            test_data = test_data.to(device).float()
            test_label = test_label.to(device).float()
            test_label = test_label.reshape(batch_size,1)
            test_preds,hs = net(test_data, hs)
            hs = tuple([h.data for h in hs])
            # print('preds_shape', test_preds)
            # print('label_shape', test_label)
            test_preds = scaler.inverse_transform(test_preds.cpu())
            test_label = scaler.inverse_transform(test_label.cpu())
            test_preds = torch.from_numpy(test_preds)
            test_label = torch.from_numpy(test_label)
            # print('pred',test_preds)
            # print('label',test_label)
            score = r2_score(test_label.cpu(), test_preds.cpu())
            test_loss = criterion(test_label, test_preds)
            # print('score',score)
            # print('loss',test_loss)
            test_r2 += score.item()
            test_mse += test_loss.item()

        print('times', times)
        print('test_mse: %.4f' % (test_mse / test_steps))
        print('r2_score: %.4f' % (test_r2 / test_steps))
        result = test_r2 / test_steps
        rmse = math.sqrt(test_mse / test_steps)
        tmp = []
        tmp.append(result)
        tmp.append(rmse)
        predict_result.append(tmp)


# predict(test_num, save_path, test_loader, device, net)
if __name__ == "__main__":
    times = sys.argv[1]
    times = int(times)

    save_path = './train_3/save_path/time_%d' % times + '.pth'
    writer = SummaryWriter('./train_3/writer/time_%d' % times)
    train(net, device, train_loader, criterion, optimizer, num_epochs, val_loader, save_path, writer)
    predict(test_num, save_path, test_loader, device, net, times)

    print(predict_result)
    predict_result = pd.DataFrame(columns=None, data=predict_result)
    predict_result.to_csv('./train_3/result.csv', mode='a', index=None, header=None)
