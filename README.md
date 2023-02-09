# Ocean: a hand-crafted toy level deep learning framework

## 1. 简介

本仓库是一个基于python的练习级别深度学习框架Ocean，主要用于熟悉深度学习算法的各个细节。

**Ocean框架具有以下特性：**

* 基于**静态图**
* 接近于**PyTorch**的api风格
* **清晰、完备的前向、反向传播过程，适用于初学者学习与熟悉**
* 基于`CuPy`完成的cuda加速

**Ocean框架完成的功能：**

* 线性层`Linear`，具有可学习参数`W`与`B`。
* 卷积层`Conv2d`，采用`img2col`方法转换为`GEMM`实现，具有可学习的卷积核。
* 池化层`MaxPooling`、`AveragePooling`与`GlobalAveragePooling`。
* 可切换`train_mode`与`eval_mode`的`BatchNorm2d`与`Dropout`。
* 激活函数，包括`Sigmoid`、`Tanh`、`Softmax`等。
* 损失函数`MSE`与`CELoss`。
* 优化器`SGD`、`Adam`。

**Ocean框架具有非常易读且易于使用的api风格：**

```python
class MLP(nn.NodeAdder):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim=784, output_dim=200),
            nn.Sigmoid(),
            nn.Linear(input_dim=200, output_dim=10),
            nn.Softmax()
        )

    def forward(self, X):
        return self.fc(X)
    
x = Variable()
m = MLP()
pred = m(x)
y = Variable()
loss = nn.MSE()
error = loss(pred=pred, target=y)
optim = SGD(graph=m.model_graph, loss=error, lr=0.1)

for i in range(E):
    # train 
    m.set_train_mode()
    for batch_data, batch_label in tqdm(train_loader, desc=f'epoch {i}'):
        optim.zero_gradient()
        for data, label in zip(batch_data, batch_label):
            x.set_value(data)
            y.set_value(label)
            optim.calculate_grad()
    # test
    m.set_eval_mode()
    for batch_data, batch_label in tqdm(test_loader):
        for data, label in zip(batch_data, batch_label):
            x.set_value(data)
            y.set_value(label)
            error.forward()
```

在`/demo`中有基于Ocean框架的更多示例。

同时，由于个人的力量有限，Ocean还有许多不足之处有待改进，未来如果有时间会尝试进行更大的改进。



## 2. 附录

1. [Mnist数据集地址](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

   



