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

class KNN:
    def __init__(self, k):
        self.kf = None
        self.df = None
        self.features = None
        self.labels = None
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
