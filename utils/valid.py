def Accuracy(matrix):
    print("===============================")
    right = matrix[0][0]+matrix[1][1]+matrix[2][2]
    tot = 0
    for i in range(3):
        tot = tot + sum(matrix[i])
    right_rate = right/tot * 100.0
    print(f"全部样本准确率为{right_rate:.2f}%")
    return right_rate

# 真正预测正确占所有预测为当前标签的比率
def Precision(matrix):
    print("===============================")
    for i in range(3):
        tot = matrix[0][i]+matrix[1][i]+matrix[2][i]
        if tot != 0 :
            right_rate = matrix[i][i] / tot *100.0
            print(f"类别{i}的精确率为{right_rate:.2f}%")
        else :
            print(f"无预测为{i}类别的结果")


def Recall(matrix):
    print("===============================")
    for i in range(3):
        tot = sum(matrix[i])
        if tot != 0:
            right_rate = matrix[i][i] / tot *100.0
            print(f"类别{i}的召回率为{right_rate:.2f}%")
        else:
            print(f"无预测为{i}类别的结果")




def validator(test_x, test_y, model,type="single"):

    # 混淆矩阵
    confusion_matrix = [[0,0,0],
                        [0,0,0],
                        [0,0,0]]
    for i in range(0,len(test_x)):
        feature = test_x[i]
        label = test_y[i] # 0 1 2
        if type=="single" :
            ret = model.Detect(feature)
        else:
            ret = model.DetectLinearMapping(feature)
        if ret!=4:
            confusion_matrix[label][ret] = confusion_matrix[label][ret] + 1
        else:
            print("appear other class")
    accuracy = Accuracy(confusion_matrix)
    Precision(confusion_matrix)
    Recall(confusion_matrix)

    print(confusion_matrix)
    return accuracy