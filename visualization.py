import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

def ScatterChart(feature1:torch.tensor,feature2:torch.tensor):

    x,y=Variable(feature1),Variable(feature2)
    plt.xlabel("x: zuobiao")
    plt.ylabel("y: zuobiao")
    plt.xlim([-15,30])
    plt.ylim([-15,30])
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.figure()
    plt.show()

if __name__ == "__main__":
    feature1 = torch.tensor([1,2,3])
    feature2 = torch.tensor([1,2,3])
    ScatterChart(feature1=feature1,feature2=feature2)