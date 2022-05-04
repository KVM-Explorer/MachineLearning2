import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

def ScatterChart(feature1:torch.tensor,feature2:torch.tensor,
                 x_range:list = None,y_range:list = None,
                 x_label:str = "",y_label: str = "",
                 title:str = None):

    x,y=Variable(feature1),Variable(feature2)

    plt.xlabel(f"{x_label}")
    plt.ylabel(f"{y_label}")
    if title is not None:
        plt.title(title)
    if x_range is not None:
        plt.xlim(x_range)
    if y_range is not None:
        plt.ylim(y_range)

    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.figure()
    plt.show()

def LineChart(feature1:torch.tensor,feature2:torch.tensor,
                 x_range:list = None,y_range:list = None,
                 x_label:str = "",y_label: str = "",
                 title:str = None):
    x, y = Variable(feature1), Variable(feature2)

    plt.xlabel(f"{x_label}")
    plt.ylabel(f"{y_label}")
    if title is not None:
        plt.title(title)
    if x_range is not None:
        plt.xlim(x_range)
    if y_range is not None:
        plt.ylim(y_range)

    plt.plot(x.data.numpy(), y.data.numpy())
    plt.figure()
    plt.show()

if __name__ == "__main__":
    feature1 = torch.tensor([1,2,3])
    feature2 = torch.tensor([1,2,3])
    ScatterChart(feature1=feature1,feature2=feature2)