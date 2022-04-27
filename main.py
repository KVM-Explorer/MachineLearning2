import KNN
import torch
if __name__ == '__main__':
    feature1 = torch.tensor([3,4])
    feature2 = torch.tensor([0,0])
    print(KNN.euclidean_distance(feature1,feature2))