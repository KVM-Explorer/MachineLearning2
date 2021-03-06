from utils import KNN, valid
import torch

if __name__ == '__main__':
    features = torch.load("data/train/vectors.pt") # 存储batch
    labels = torch.load("data/train/labels.pt")

    model = KNN.train_model(features, labels, 5)
    model.DefineDF(distance_function=KNN.chebyshev_distance)
    model.DefineKF(kernal_function=KNN.single_mapping)
    KNN.save_model(model=model, filename="KNNTestModel")
    # KNN.load_model("model/KNNTestModel.pickle")
    test_x = torch.load("data/test/vectors.pt")
    test_y = torch.load("data/test/labels.pt")

    valid.validator(test_x, test_y, model)





