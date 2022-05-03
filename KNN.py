import heapq
import math
import torch
import pickle

# =====================距离求解======================
# L2 范数
import visualization


def euclidean_distance(feature1, feature2):
    return torch.sqrt(torch.sum(torch.pow((feature1 + feature2), 2)))


# L1 范数
def manhattan_distance(feature1, feature2):
    return torch.sum(torch.abs(feature1 - feature2))


# L 无穷范数
def chebyshev_distance(feature1, feature2):
    return torch.max(torch.abs(feature1 - feature2))

def bhattacharyya_distance(feature1,feature2):
    return -math.ln(torch.sum(torch.sqrt(torch.dot(feature1,feature2))))        # 可能存在负数，但如果按照距离比较的画不影响

# ==================== 核函数=========================
def single_mapping(feature):
    return feature


# ====================预处理函数=======================

def train_model(features, lables, k,type = "single"):
    model = KNN(k)
    if type == "single" :
        model.Train(features,lables)
    else:
        model.TrainLinearMapping(features,lables)
    return model

def save_model(model,filename):
    with open(f'model/{filename}.pickle', 'wb') as f:
        pickle.dump(model,f)

def load_model(filename):
    with open(filename,"rb") as f:
        model = pickle.load(f)
    return model

# 将特征向量转换为矩阵
# def vector2matrix(feature:torch.tensor):
#     return feature.reshape(1,-1)

class KNN:
    def __init__(self, k):
        self.kf = None
        self.df = None
        self.features = None
        self.labels = None
        self.L = None
        self.k = k

    class Item:
        def __init__(self, dis, label):
            self.dis = dis;
            self.label = label;

        def __lt__(self, other):
            if self.dis < other.dis:
                return True
            else:
                return False

    def Train(self, features, labels):
        self.features = features
        self.labels = labels

    def TrainLinearMapping(self, features:torch.tensor, labels:torch.tensor, epochs =10, learn_rate = 1e-6,r = 0.5):
        self.features = features
        self.labels = labels
        loss_x = [i for i in range(1,epochs+1)]
        loss_y = []
        L = torch.rand(features.shape[1])
        for epoch in range(1,epochs+1):
            loss = 0
            gridient = 0
            for i in range(len(features)):
                for j in range(len(features)):
                    if i==j or labels[i]!=labels[j] : continue
                    # pull
                    x_ij = features[i] - features[j]
                    pull_loss = float(sum(x_ij*x_ij*L))
                    pull_gridient = (1-r)*L*x_ij*x_ij

                    gridient += pull_gridient
                    loss += pull_loss
                    # push
                    for p in range(len(features)):
                        if p==i or p==j or labels[p]==labels[i]: continue
                        x_ip = features[i]-features[p]
                        push_loss = 1+pull_loss-float(sum(L*x_ip*x_ip))
                        if push_loss < 0: continue
                        push_gridient = r*(1+pull_gridient-L*x_ip*x_ip)

                        gridient += push_gridient
                        loss += push_loss

            print(f"training linear mapping parameter epoch:{epoch} / {epochs} loss={loss} learning rate={learn_rate}")
            loss_y.append(loss)
            L = L + gridient*learn_rate     #?????为什么反而采取加法
            learn_rate *= 0.95 ** (epoch + 1)
        print(f"Linear Mapping {L}")
        self.L = L
        visualization.ScatterChart(torch.tensor(loss_x),torch.tensor(loss_y))


    def DetectLinearMapping(self,feature):
        dis_list = []
        feature_kf = self.kf(feature)*self.L
        for i in range(0, len(self.features)):
            dis = self.df(self.kf(self.features[i])*self.L, feature_kf)
            heapq.heappush(dis_list, self.Item(dis, self.labels[i]))  # 默认小根堆

        label_count = {0: 0, 1: 0, 2: 0}
        label = 0
        for i in range(0, self.k):
            tmp = heapq.heappop(dis_list)
            label_count[int(tmp.label)] += 1
            if label_count[int(tmp.label)] > label_count[label]:
                label = int(tmp.label)

        # Todo 关闭学习
        # self.features.add(feature)
        # self.labels.add(label)
        return label


    def Detect(self, feature):
        dis_list = []
        feature_kf = self.kf(feature)
        for i in range(0, len(self.features)):
            dis = self.df(self.kf(self.features[i]), feature_kf)
            heapq.heappush(dis_list, self.Item(dis, self.labels[i]))  # 默认小根堆

        label_count = {0: 0, 1: 0, 2: 0}
        label = 0
        for i in range(0, self.k):
            tmp = heapq.heappop(dis_list)
            label_count[int(tmp.label)] += 1
            if label_count[int(tmp.label)] > label_count[label]:
                label = int(tmp.label)

        # Todo 关闭学习
        # self.features.add(feature)
        # self.labels.add(label)
        return label

    def DefineKF(self, kernal_function):
        self.kf = kernal_function

    def DefineDF(self,distance_function):
        self.df = distance_function
