import math

import torch

from utils import KNN
from utils import valid
from utils import visualization


def k_cross_validation(features, labels, block_size=3):
    block_num = len(features) // block_size
    k_epoch = 10
    # 枚举K值
    scores = []
    for k in range(1, k_epoch + 1):
        # 当前测试数据块
        average_score = 0
        for valid_block in range(block_num + 1):
            # 生成训练集 测试集
            train_data = []
            train_label = []
            test_data = []
            test_label = []
            for index in range(len(features)):
                if index / block_size != valid_block:
                    train_data.append(features[index])
                    train_label.append(labels[index])
                else:
                    test_data.append(features[index])
                    test_label.append(labels[index])

            # 训练并验证K值
            model = KNN.train_model(train_data, train_label, k)
            model.DefineDF(distance_function=KNN.manhattan_distance)
            model.DefineKF(kernal_function=KNN.single_mapping)
            accuracy = valid.validator(test_data, test_label, model=model)
            average_score += accuracy
        average_score = average_score / (block_num + 1)
        scores.append(average_score)

    X = [i for i in range(1, k_epoch + 1)]
    Y = scores
    visualization.ScatterChart(torch.tensor(X), torch.tensor(Y),
                               title="cross valid K",
                               x_label="k",
                               y_label="accuracy",
                               x_range=[0, 11],
                               y_range=[0, 100]
                               )
    print(f"block_size:{block_size} block_num:{block_num} best_score:{max(scores):.2f}")
    best_k = [1]
    for i in range(len(scores)):
        if scores[i] > scores[best_k[0] - 1]:
            best_k = [i + 1]
            continue
        if scores[i] == scores[best_k[0]]:
            best_k.append(i + 1)
    return best_k

# 定义同类样本的密度
def class_density(features):
    density = 0
    num = len(features)
    for i in range(len(features)):
        for j in range(i+1,len(features)):
            density += KNN.euclidean_distance(features[i],features[j])
    density = density / ((num*(num-1))/2)
    return 1/density
def sample_density(features,labels):
    class0_features = []
    class1_features = []
    class2_features = []

    for i in range(len(features)):
        if labels[i] == 0:
            class0_features.append(features[i])
        elif labels[i] == 1:
            class1_features.append(features[i])
        else:
            class2_features.append(features[i])


    class0_density = class_density(class0_features)
    class1_density = class_density(class1_features)
    class2_density = class_density(class2_features)
    print(f"class 0 density:{class0_density}")
    print(f"class 1 density:{class1_density}")
    print(f"class 2 density:{class2_density}")

def one_class_range(features):
    max_dis = 0
    for i in range(len(features)):
        for j in range(i+1,len(features)):
            dis = KNN.euclidean_distance(features[i],features[j])
            max_dis = max(max_dis,dis)
    print(f"one class range:{max_dis}")
    return max_dis