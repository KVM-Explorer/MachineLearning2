import heapq
import math
import torch
import pickle

# =====================距离求解======================
# L2 范数
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



# ====================预处理函数=======================

def train_model(features, lables, k):
    model = KNN(k)
    model.Train(features,lables)
    return model

def save_model(model):
    with open('data/model.pickle', 'wb') as f:
        pickle.dump(model,f)

def load_model(filename):
    with open(filename,"rb") as f:
        model = pickle.load(f)
    return model

# 将特征向量转换为矩阵
def vector2matrix(feature:torch.tensor):
    return feature.reshape(1,-1)

class KNN:
    def __init__(self, k):
        self.kf = None
        self.df = None
        self.features = None
        self.labels = None
        self.S = None
        self.k = k

    class Item:
        def __init__(self, dis, label):
            self.dis = dis;
            self.label = label;

        def __iter__(self, other):
            if self.dis < other.dis:
                return True
            else:
                return False;

    def Train(self, features, labels):
        self.features = features
        self.labels = labels

    def TrainLinearMapping(self, features:torch.tensor, labels:torch.tensor, epochs =10, learn_rate = 0.01,r = 0.5):
        self.features = features
        self.labels = labels
        L = vector2matrix(torch.rand(features.shape[1]))
        for epoch in range(epochs):
            loss = 0
            gridient = 0
            for i in range(len(features)):
                for j in range(len(features)):
                    if i==j or labels[i]!=labels[j] : continue
                    # pull
                    x_ij = features[i] - features[j]
                    pull_loss = float(sum(x_ij*x_ij*L*L))
                    pull_gridient = (1-r)*L*L*x_ij

                    gridient += pull_gridient
                    loss += pull_loss
                    # push
                    for p in range(len(features)):
                        if p==i or p==j or labels[p]==labels[i]: continue
                        x_ip = features[i]-features[p]
                        push_loss = 1+pull_loss-float(sum(L*L*x_ip*x_ip))
                        if push_loss < 0: continue
                        push_gridient = r*(1+pull_gridient-L*L*x_ip)

                        gridient += push_gridient
                        loss += push_loss

            print(f"training linear mapping parameter epoch:{epoch} / {epochs} loss={loss} learning rate={learn_rate}")
            L = L - gridient*learn_rate
            learn_rate *= 0.95 ** (epoch + 1)
        print(f"Linear Mapping {L}")
        self.S = L*L


    def DetectLinearMapping(self,feature):


    def Detect(self, feature):
        dis_list = []
        feature_kf = self.kf(feature)
        for i in range(0, len(self.features)):
            dis = self.df(self.kf(self.features[i]), feature)  # Todo 计算核函数
            heapq.heappush(dis_list, self.Item(dis, self.labels[i]))  # 默认小根堆

        label_count = {0: 0, 1: 0, 2: 0}
        label = 0
        for i in range(0, self.k):
            tmp = heapq.heappop(dis_list)
            label_count[tmp.label] += 1
            if label_count[tmp.label] > label_count[label]:
                label = tmp.label

        self.features.add(feature)
        self.labels.add(label)

    def DefineKF(self, kernal_function):
        self.kf = kernal_function;
