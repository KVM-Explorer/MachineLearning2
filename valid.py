import KNN
def Accuracy(matrix):
    print("===============================")
    right = matrix[0][0]+matrix[1][0]+matrix[2][0]
    tot = 0
    for i in range(3):
        tot = tot + sum(matrix[i])
    right_rate = right/tot *100.0
    print(f"全部样本准确率为{right_rate}%")

# 真正预测正确占所有预测为当前标签的比率
def Precision(matrix):
    print("===============================")
    for i in range(3):
        tot = matrix[0][i]+matrix[1][i]+matrix[2][i]
        right_rate = matrix[i][i] / tot *100.0
        print(f"类别{i}的精确率为{right_rate}%")

#
def Recall(matrix):
    print("===============================")
    for i in range(3):
        tot = sum(matrix[i])
        right_rate = matrix[i][i] / tot *100.0
        print(f"类别{i}的召回率为{right_rate}%")




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
        confusion_matrix[label][ret] = confusion_matrix[label][ret] + 1

    Accuracy(confusion_matrix)
    Precision(confusion_matrix)
    Recall(confusion_matrix)
