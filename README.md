# Ocean: a hand-crafted toy level deep learning framework

## 1. 简介

本仓库是一个基于python的练习级别深度学习框架Ocean，主要用于熟悉深度学习算法的各个细节。

**Ocean框架具有以下特性：**

* 基于静态图
* 接近于PyTorch的模型编写风格
* 较为明晰、完备的前向、反向传播过程，适用于初学者学习与熟悉。
* 基于`CuPy`完成的cuda加速

**Ocean框架完成的功能：**

* 线性层`Linear`，具有可学习参数`W`与`B`。
* 卷积层`Conv2d`，采用`img2col`方法转换为`GEMM`实现，具有可学习的卷积核。
* 池化层`MaxPooling`、`AveragePooling`与`GlobalAveragePooling`。
* 批标准化层`BatchNorm2d`。
* 激活函数，包括`Sigmoid`、`Tanh`、`Softmax`等。
* 损失函数`MSE`与`CELoss`。
* 优化器`SGD`、`Adam`。

在`/demo`中有基于Ocean框架的示例。

同时，由于个人的力量有限，Ocean还有许多不足之处有待改进，未来如果有时间会尝试进行更大的改进。



## 2. 附录

1. [Mnist数据集地址](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

   



