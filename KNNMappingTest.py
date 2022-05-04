from utils import KNN, valid
import torch

if __name__ == '__main__':

    features = torch.load("data/train/vectors.pt") # 存储batch
    labels = torch.load("data/train/labels.pt")

    model = KNN.train_model(features, labels, 5  ,"mapping")
    model.DefineDF(distance_function=KNN.euclidean_distance)
    model.DefineKF(kernal_function=KNN.single_mapping)
    KNN.save_model(model=model,filename="KNNMappingModel")
    # model = KNN.load_model("model/KNNMappingModel.pickle")
    # model.k =10
    test_x = torch.load("data/test/vectors.pt")
    test_y = torch.load("data/test/labels.pt")

    valid.validator(test_x, test_y, model, "mapping")





