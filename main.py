import KNN
import torch

if __name__ == '__main__':
    feature1 = torch.tensor([3,4])
    feature2 = torch.tensor([0,0])
    print(KNN.euclidean_distance(feature1,feature2))

    vector = torch.tensor([[3,4,10,10],[10,10,100,100]],dtype=torch.float32) # 注意torch保持原有变量类型
    print(KNN.normalize(vector))