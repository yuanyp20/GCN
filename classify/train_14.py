# data: h5_50
# model: lstm
# window_len: 10
# loss: ce loss
# loss setting: weight
import random
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import pandas as pd
from data_process import list_file_name
import matplotlib.pyplot as plt
import sys
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from model_cnn import Net
from torch import optim
from torch import nn
from tensorboardX import SummaryWriter
from sklearn.metrics import r2_score
from torch_geometric.data import Data
import networkx as nx

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


class MultiFocalLoss(nn.Module):
    """
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor shape=[num_class, ]
        gamma: hyper-parameter
        reduction: reduction type
    """

    def __init__(self, num_class, alpha=None, gamma=2, reduction='mean'):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(num_class, ) - 0.5
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')

    def forward(self, logit, target):
        # assert isinstance(self.alpha,torch.Tensor)\
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=1)

        if prob.dim() > 2:
            # used for 3d-conv:  N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            N, C = logit.shape[:2]
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]

        ori_shp = target.shape
        target = target.view(-1, 1)

        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        logpt = torch.log(prob)
        # alpha_class = alpha.gather(0, target.squeeze(-1))
        alpha_weight = alpha[target.squeeze().long()]
        loss = -alpha_weight * torch.pow(torch.sub(1.0, prob), self.gamma) * logpt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)

        return loss


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
# torch.cuda.set_device(2)
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
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
train_mean_std = pd.read_csv('train_mean_std.csv', engine='python', header=None, index_col=None)
scaler = []
for train_col in range(16):
    scaler.append(StandardScaler())
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
    for col in range(16):
        data[col] = scaler[col].fit_transform(data[col].values.reshape(-1, 1))

    x = data.values[:, 0:16]
    y = data.values[:, 17]

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
# print(X_test)
# print(Y_test)
train_dataset = MyDataset(features=X_train, labels=Y_train)
val_dataset = MyDataset(features=X_val, labels=Y_val)
test_dataset = MyDataset(features=X_test, labels=Y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# net = simple_LSTM(hidden_size=32, enose_dim=16, num_layers=2)
net = Net()
print(net)
net.to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)
# criterion = facal
optimizer = optim.Adam(net.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# save_path = './train_3.pth'

train_steps = len(train_loader)
val_steps = len(val_loader)
# writer = SummaryWriter('./train_3')

num_epochs = 50


def train(net, device, train_loader, criterion, optimizer, num_epochs, val_loader, save_path, writer):
    bestmse = 1e6
    for epoch in range(num_epochs):
        # scheduler.step()
        # Train:
        net.train()


        running_loss = 0.0
        train_bar = tqdm(train_loader)
        num_train_true = 0
        num_val_true = 0

        for step, (train_data, train_label) in enumerate(train_bar):
            # print(train_data.shape)
            # print(train_label.shape)

            train_data = train_data.float().to(device)  # 部署到device
            train_label = train_label.long().to(device)

            optimizer.zero_grad()  # 梯度置零

            y_hat = net(train_data)  # 模型训练

            # y_hat, hs = net(train_data, hs)  # 模型训练
            # 每一次在读取并计算完一个batch的数据之后会有一个ht连接到计算图中，这个ht参与反向传播求梯度，梯度在求完结果之后就被释放掉了，当到下一个也就是h(t+1)的时候，
            # 在计算梯度时还会经过ht，（因为这两个在计算图中连着），但是ht的相关信息已经被释放了，所以会产生报错。


            # print(y_hat.size())
            # 得到的loss是batch内损失的平均
            loss = criterion(y_hat, train_label)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch [{}/{}] loss :{:.4f}".format(epoch + 1, num_epochs, loss)

            # 预测的分类值及其对应索引
            pred_value, pred_index = y_hat.max(1)
            # 预测索引与标签对应的bool标识
            pred_true_bool = train_label.eq(pred_index)
            # 一个batch中预测正确的数量
            pred_true_num = torch.sum(pred_true_bool)
            # num_train_true表示整个epoch预测正确的数量
            num_train_true += pred_true_num

        # 模型验证
        net.eval()

        val_loss = 0.0
        val_bar = tqdm(val_loader)
        with torch.no_grad():
            for step, (val_data, val_label) in enumerate(val_bar):
                val_data = val_data.float().to(device)
                val_label = val_label.long().to(device)
                val_preds = net(val_data)

                loss = criterion(val_preds, val_label)  # 计算损失
                val_loss += loss.item()

                # 预测的分类值及其对应索引
                pred_value, pred_index = val_preds.max(1)
                # 预测索引与标签对应的bool标识
                pred_true_bool = val_label.eq(pred_index)
                # 一个batch中预测正确的数量
                pred_true_num = torch.sum(pred_true_bool)
                # num_train_true表示整个epoch预测正确的数量
                num_val_true += pred_true_num

        print(
            f'Epoch {epoch + 1}/{num_epochs} --- train loss {running_loss / train_steps} --- val loss {val_loss / val_steps}'
            f'---num_train_true {num_train_true} ---train_true_rate {num_train_true / len(train_dataset)} ---num_val_true {num_val_true}'
            f'---val_true_rate {num_val_true / len(val_dataset)}'
        )

        if (val_loss / val_steps) < bestmse:
            bestmse = val_loss / val_steps
            torch.save(net.state_dict(), save_path)

        writer.add_scalars('logs', {'train_loss': running_loss / train_steps, 'val_loss': val_loss / val_steps,
                                    'train_true_rate': num_train_true / len(train_dataset),
                                    'val_true_rate': num_val_true / len(val_dataset)}, epoch)
    writer.close()
    print('finish training')


# train(net, device, train_loader, criterion, optimizer, num_epochs, val_loader)


test_num = len(test_dataset)

predict_result = []

compare_result_label = []
compare_result_pred = []


def predict(test_num, save_path, test_loader, device, net, times):
    print("{} data for test".format(test_num))
    print("loading model...")
    test_loss = 0.0
    num_test_true = 0
    net.load_state_dict(torch.load(save_path))

    net.eval()
    net.to(device)
    test_steps = len(test_loader)


    f1_scores = 0.0

    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for step, (test_data, test_label) in enumerate(test_bar):
            # print(test_data)
            # print(test_label)
            test_data = test_data.float().to(device)
            test_label = test_label.long().to(device)
            test_preds = net(test_data)

            # test_preds, hs = net(test_data, hs)
            # test_preds = scaler.inverse_transform(test_preds)
            # test_label = scaler.inverse_transform(test_label)
            # test_preds = torch.from_numpy(test_preds)
            # test_label = torch.from_numpy(test_label)
            # hs = tuple([h.data for h in hs])

            loss = criterion(test_preds, test_label)

            test_loss += loss.item()

            # 预测的分类值及其对应索引
            pred_value, pred_index = test_preds.max(1)
            # 预测索引与标签对应的bool标识
            pred_true_bool = test_label.eq(pred_index)
            # 一个batch中预测正确的数量
            pred_true_num = torch.sum(pred_true_bool)
            # num_train_true表示整个epoch预测正确的数量
            num_test_true += pred_true_num
            test_label_np = test_label.cpu().numpy()
            pred_index_np = pred_index.cpu().numpy()
            # print(test_label_np)
            # print(pred_index_np)

            compare_result_label.extend(test_label_np)
            compare_result_pred.extend(pred_index_np)
            # if step < 5:
            #     plt.figure(figsize=(20, 16))
            #     plt.plot(test_label.cpu(), label='True 2')
            #     plt.plot(test_preds.cpu(), label='Pred 2')
            #     plt.legend(loc='best')
            #     plt.show()

            f1 = f1_score(test_label.cpu(), pred_index.cpu(), average='weighted')
            f1_scores += f1

        print('times', times)
        print('test_loss: %.4f' % (test_loss / test_steps))
        print('test_true_rate', {num_test_true / test_num})
        print('F1-measure', f1_scores / test_steps)

        result = num_test_true / test_num
        f1_mean = f1_scores / test_steps

        tmp = []
        tmp.append(result.item())
        tmp.append(f1_mean)
        predict_result.append(tmp)


# predict(test_num, save_path, test_loader, device, net)
if __name__ == "__main__":
    times = sys.argv[1]
    times = int(times)

    save_path = './train_14/save_path/time_%d' % times + '.pth'
    writer = SummaryWriter('./train_14/writer/time_%d' % times)
    train(net, device, train_loader, criterion, optimizer, num_epochs, val_loader, save_path, writer)
    predict(test_num, save_path, test_loader, device, net, times)

    print(predict_result)
    predict_result = pd.DataFrame(columns=None, data=predict_result)
    predict_result.to_csv('./train_14/result.csv', mode='a', index=None, header=None)
    print(len(compare_result_label))
    print(len(compare_result_pred))
    compare_result_label = pd.DataFrame(columns=list('a'), data=compare_result_label)
    compare_result_pred = pd.DataFrame(columns=list('a'), data=compare_result_pred)
    # print(compare_result_label)
    compare_result = pd.merge(compare_result_label, compare_result_pred, left_index=True, right_index=True)
    # print(compare_result)
    compare_result.to_csv('./train_14/compare/time_%d' % times + '.csv', mode='w', index=0, header=None)
