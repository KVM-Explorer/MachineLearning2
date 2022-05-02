import KNN
import torch

import valid

if __name__ == '__main__':
    features = torch.load("data/train/vectors.pt") # 存储batch
    labels = torch.load("data/train/labels.pt")

    model = KNN.train_model(features, labels, 3)
    model.DefineKF(kernal_function=KNN.euclidean_distance)

    test_x = torch.load("data/test/vector.pt")
    test_y = torch.load("data/test/label.pt")

    valid.validator(test_x, test_y, model)




