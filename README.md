# Ocean: a hand-crafted toy level deep learning framework

本仓库是一个基于python的练习级别深度学习框架Ocean，主要用于熟悉深度学习算法的各个细节。

Ocean框架具有以下特性：

* 基于静态图
* 接近于PyTorch的模型编写风格
* 较为明晰的前向、反向传播过程，适用于初学者学习与熟悉。
* 基于`CuPy`完成的cuda加速

Ocean框架完成的功能：
* 线性层`Linear`，具有可学习参数`W`与`B`。
* 卷积层`Conv2d`，具有可学习的卷积核。
* 池化层`MaxPooling`、`AveragePooling`与`GlobalAveragePooling`。
* 激活函数，包括`Sigmoid`、`Tanh`、`Softmax`等。
* 损失函数`MSE`与`CELoss`。
* 优化器`SGD`。

同时，由于个人的力量有限，Ocean还有许多不足之处有待改进，未来如果有时间会尝试进行更大的改进。