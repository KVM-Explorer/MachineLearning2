import torch

from utils import optimization

if __name__ == "__main__":
    features = torch.load("data/train/vectors.pt")  # 存储batch
    labels = torch.load("data/train/labels.pt")

    # 交叉验证最佳K值
    #best_k = optimization.k_cross_validation(features=features,labels=labels,block_size=12)
    #print(best_k)
    optimization.sample_density(features, labels)