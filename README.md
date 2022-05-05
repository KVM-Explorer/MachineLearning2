# 机器学习KNN

## 简介

## 项目结构

文件结构data数据集、doc开发调试日志、model模型存储、utils存放代码组建包括KNN算法、线性变换、数据可视化和数据校验等多个模块、最外层的AdjustParameterTest为调参调用模块、KNNMappingTest为线性变换下的KNN、KNNTest为普通KNN算法、PreProcess为数据预处理和特征提取算法

## 项目配置

1. 使用pip

```
pip install -r requirements.txt
```

2. 采用pycharm打开requirement文件夹根据提示进行安装

## 数据预处理——PreProcess.py

调用PreProcess程序运行生成数据集

## 普通KNN算法——KNNTest.py

1. 首先加载数据集
2. 数据集训练模型
3. 为模型选择距离函数
4. 为模型选择核函数
5. 保存当前模型（也可以直接采用历史模型）
6. 加载校验数据并使用校验器进行校验

## 线性变换下的KNN算法——KNNMappingTest.py

同上，但是在训练环节采用了线性变换的训练算法，其余流程保持一致。

## 参数调优——AdjuestingParametersTest.py

- 交叉验证实验模块调用

- one class 阈值筛选
- 特征向量密度计算

## 其他

项目采用git对代码进行管理，提供代码回溯、追踪历史版本回退功能