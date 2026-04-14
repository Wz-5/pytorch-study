# model.py
import numpy as np


class FullyConnectedLayer:
    def __init__(self, input_dim, output_dim, weight_init="he"):
        """
        全连接层: Z = XW + b

        参数
        ----
        input_dim : 输入维度
        output_dim: 输出维度
        weight_init: 权重初始化方式
                     可选 "he" / "xavier" / "small"
        """
        if weight_init == "he":
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        elif weight_init == "xavier":
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(1.0 / input_dim)
        else:  # "small"
            self.W = np.random.randn(input_dim, output_dim) * 0.01

        self.b = np.zeros((1, output_dim))

        # 前向缓存
        self.X = None
        self.Z = None

        # 参数梯度
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # 用于分析梯度消失/爆炸
        self.last_grad_norm_w = 0.0   # 当前层参数梯度范数 ||dW||
        self.last_grad_norm_x = 0.0   # 传给前一层的梯度范数 ||dX||

    def forward(self, X):
        """
        前向传播
        Z = XW + b
        """
        self.X = X
        self.Z = np.dot(X, self.W) + self.b
        return self.Z

    def backward(self, dZ, l2_lambda=0.0):
        """
        反向传播

        这里 dZ 默认已经是“对 batch 平均后的梯度”，
        因此这里不再额外除以 m，只在 L2 项里保留 lambda/m。

        若损失函数为平均交叉熵，则:
        dZ = (P - Y) / m

        则有:
        dW = X^T dZ + (lambda/m) W
        db = sum(dZ, axis=0)
        dX = dZ W^T
        """
        m = self.X.shape[0]

        self.dW = np.dot(self.X.T, dZ) + (l2_lambda / m) * self.W
        self.db = np.sum(dZ, axis=0, keepdims=True)
        dX = np.dot(dZ, self.W.T)

        # 记录梯度范数，便于比较不同激活函数的梯度传播情况
        self.last_grad_norm_w = float(np.linalg.norm(self.dW))
        self.last_grad_norm_x = float(np.linalg.norm(dX))

        return dX

    def update(self, lr):
        """
        参数更新
        """
        self.W -= lr * self.dW
        self.b -= lr * self.db


class ReLU:
    def __init__(self):
        self.Z = None

    def forward(self, Z):
        self.Z = Z
        return np.maximum(0.0, Z)

    def backward(self, dA):
        dZ = dA.copy()
        dZ[self.Z <= 0] = 0.0
        return dZ


class Sigmoid:
    def __init__(self):
        self.A = None

    def forward(self, Z):
        Z = np.clip(Z, -50, 50)
        self.A = 1.0 / (1.0 + np.exp(-Z))
        return self.A

    def backward(self, dA):
        return dA * self.A * (1.0 - self.A)


class Tanh:
    def __init__(self):
        self.A = None

    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dA):
        return dA * (1.0 - self.A ** 2)


class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.Z = None
        self.alpha = alpha

    def forward(self, Z):
        self.Z = Z
        return np.where(Z > 0, Z, self.alpha * Z)

    def backward(self, dA):
        dZ = np.ones_like(self.Z)
        dZ[self.Z < 0] = self.alpha
        return dA * dZ


class SoftmaxCrossEntropy:
    def __init__(self):
        self.probs = None
        self.y_true = None

    def forward(self, logits, y_true):
        """
        Softmax + 平均交叉熵

        logits: (m, C)
        y_true: (m, C) one-hot
        """
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        self.y_true = y_true

        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(self.probs + 1e-12)) / m
        return loss

    def backward(self):
        """
        Softmax + CrossEntropy 联合求导:
        dZ = (P - Y) / m
        """
        m = self.y_true.shape[0]
        return (self.probs - self.y_true) / m


class MLP:
    def __init__(self, input_dim, hidden_dims, output_dim,
                 activation="relu", weight_init="he"):
        """
        多层感知机

        例:
        input_dim=784
        hidden_dims=[256, 128]
        output_dim=10
        """
        self.layers = []
        self.activations = []

        dims = [input_dim] + hidden_dims + [output_dim]

        # 构造线性层
        for i in range(len(dims) - 1):
            self.layers.append(
                FullyConnectedLayer(dims[i], dims[i + 1], weight_init=weight_init)
            )

        # 构造隐藏层激活函数
        for _ in range(len(hidden_dims)):
            if activation == "relu":
                self.activations.append(ReLU())
            elif activation == "sigmoid":
                self.activations.append(Sigmoid())
            elif activation == "tanh":
                self.activations.append(Tanh())
            elif activation == "leakyrelu":
                self.activations.append(LeakyReLU())
            else:
                raise ValueError(
                    "Unsupported activation. Choose from "
                    "['relu', 'sigmoid', 'tanh', 'leakyrelu']"
                )

        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, X):
        """
        前向传播
        隐藏层: Linear -> Activation
        输出层: Linear
        """
        out = X

        for i in range(len(self.activations)):
            out = self.layers[i].forward(out)
            out = self.activations[i].forward(out)

        logits = self.layers[-1].forward(out)
        return logits

    def compute_loss(self, logits, y_true, l2_lambda=0.0):
        """
        总损失 = 交叉熵 + L2正则项
        """
        data_loss = self.loss_fn.forward(logits, y_true)

        m = y_true.shape[0]
        l2_loss = 0.0
        for layer in self.layers:
            l2_loss += np.sum(layer.W ** 2)
        l2_loss *= (l2_lambda / (2.0 * m))

        return data_loss + l2_loss

    def backward(self, l2_lambda=0.0):
        """
        反向传播
        """
        # 输出层梯度
        dZ = self.loss_fn.backward()

        # 最后一层线性层反传
        dA = self.layers[-1].backward(dZ, l2_lambda=l2_lambda)

        # 隐藏层按从后往前反传
        for i in reversed(range(len(self.activations))):
            dZ = self.activations[i].backward(dA)
            dA = self.layers[i].backward(dZ, l2_lambda=l2_lambda)

    def update(self, lr):
        """
        梯度下降更新
        """
        for layer in self.layers:
            layer.update(lr)

    def predict_proba(self, X):
        logits = self.forward(X)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def get_gradient_norms(self, kind="weight"):
        """
        返回每一层的梯度范数列表，用于分析梯度消失/爆炸

        参数
        ----
        kind:
            "weight" -> 返回每层 ||dW||
            "input"  -> 返回每层传回去的 ||dX||
        """
        if kind == "weight":
            return [layer.last_grad_norm_w for layer in self.layers]
        elif kind == "input":
            return [layer.last_grad_norm_x for layer in self.layers]
        else:
            raise ValueError("kind must be 'weight' or 'input'")