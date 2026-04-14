# model.py
import numpy as np


class FullyConnectedLayer:
    def __init__(self, input_dim, output_dim, weight_init="he"):
        """
        全连接层（线性层）

        前向传播公式：
            Z = XW + b

        参数：
            input_dim  : 输入维度
            output_dim : 输出维度
            weight_init: 权重初始化方式
                         可选 "he" / "xavier" / "small"

        不同初始化方式说明：
        1. he:
           更适合 ReLU / LeakyReLU
           Var(W) ≈ 2 / input_dim

        2. xavier:
           更适合 Tanh / Sigmoid
           Var(W) ≈ 1 / input_dim

        3. small:
           简单小随机数初始化，适合做对照实验
        """
        if weight_init == "he":
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        elif weight_init == "xavier":
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(1.0 / input_dim)
        else:  # "small"
            self.W = np.random.randn(input_dim, output_dim) * 0.01

        # 偏置初始化为 0
        self.b = np.zeros((1, output_dim))

        # 前向传播缓存
        # 反向传播时需要用到输入 X 和线性输出 Z
        self.X = None
        self.Z = None

        # 参数梯度
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # 用于观察梯度传播情况（分析梯度消失/爆炸）
        self.last_grad_norm_w = 0.0
        self.last_grad_norm_x = 0.0

    def forward(self, X):
        """
        前向传播

        输入：
            X.shape = (m, input_dim)

        输出：
            Z.shape = (m, output_dim)

        数学公式：
            Z = XW + b
        """
        self.X = X
        self.Z = np.dot(X, self.W) + self.b
        return self.Z

    def backward(self, dZ, l2_lambda=0.0):
        """
        反向传播

        已知：
            dZ = ∂L / ∂Z

        求：
            dW = ∂L / ∂W
            db = ∂L / ∂b
            dX = ∂L / ∂X

        前向传播公式：
            Z = XW + b

        梯度公式：
            dW = X^T dZ + (λ/m)W
            db = sum(dZ, axis=0)
            dX = dZ W^T

        
        """
        m = self.X.shape[0]

        
        # 1. 计算权重梯度 dW
        
        # self.X.T.shape = (input_dim, m)
        # dZ.shape       = (m, output_dim)
        # 因此：
        #   X^T dZ.shape = (input_dim, output_dim)
        # 与 W 的形状一致
        #
        # 前一项 X^T dZ ：来自数据损失
        # 后一项 (λ/m)W ：来自 L2 正则化
        self.dW = np.dot(self.X.T, dZ) + (l2_lambda / m) * self.W

        
        # 2. 计算偏置梯度 db
        
        # dZ.shape = (m, output_dim)
        # db.shape = (1, output_dim)
        self.db = np.sum(dZ, axis=0, keepdims=True)

        
        # 3. 计算传递给前一层的梯度 dX
        
        # 当前层前向：
        #   Z = XW + b
        # 因此反向传播：
        #   dX = dZ W^T
        #
        # dZ.shape   = (m, output_dim)
        # W.T.shape  = (output_dim, input_dim)
        # dX.shape   = (m, input_dim)
        dX = np.dot(dZ, self.W.T)

        
        # 4. 记录梯度范数，便于分析梯度消失/爆炸
        
        self.last_grad_norm_w = float(np.linalg.norm(self.dW))
        self.last_grad_norm_x = float(np.linalg.norm(dX))

        return dX

    def update(self, lr):
        """
        使用梯度下降更新参数

        更新公式：
            W = W - lr * dW
            b = b - lr * db
        """
        self.W -= lr * self.dW
        self.b -= lr * self.db


class ReLU:
    def __init__(self):
        """
        ReLU 激活函数

        前向：
            A = max(0, Z)
        """
        self.Z = None

    def forward(self, Z):
        """
        前向传播
        """
        self.Z = Z
        return np.maximum(0.0, Z)

    def backward(self, dA):
        """
        反向传播

        已知：
            dA = ∂L / ∂A

        ReLU 导数：
            dA/dZ = 1,  当 Z > 0
                    0,  当 Z <= 0

        因此：
            dZ = dA ⊙ 1(Z > 0)
        """
        dZ = dA.copy()
        dZ[self.Z <= 0] = 0.0
        return dZ


class Sigmoid:
    def __init__(self):
        """
        Sigmoid 激活函数

        前向：
            A = 1 / (1 + e^(-Z))
        """
        self.A = None

    def forward(self, Z):
        """
        前向传播

        
        """
        Z = np.clip(Z, -50, 50)
        self.A = 1.0 / (1.0 + np.exp(-Z))
        return self.A

    def backward(self, dA):
        """
        反向传播

        Sigmoid 导数：
            dA/dZ = A(1 - A)

        因此：
            dZ = dA ⊙ A(1 - A)

        
        """
        return dA * self.A * (1.0 - self.A)


class Tanh:
    def __init__(self):
        """
        Tanh 激活函数

        前向：
            A = tanh(Z)
        """
        self.A = None

    def forward(self, Z):
        """
        前向传播
        """
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dA):
        """
        反向传播

        Tanh 导数：
            dA/dZ = 1 - tanh²(Z) = 1 - A²

        因此：
            dZ = dA ⊙ (1 - A²)

        
        """
        return dA * (1.0 - self.A ** 2)


class LeakyReLU:
    def __init__(self, alpha=0.01):
        """
        LeakyReLU 激活函数

        前向：
            A = Z,      当 Z > 0
            A = αZ,     当 Z <= 0

        
        """
        self.Z = None
        self.alpha = alpha

    def forward(self, Z):
        """
        前向传播
        """
        self.Z = Z
        return np.where(Z > 0, Z, self.alpha * Z)

    def backward(self, dA):
        """
        反向传播

        LeakyReLU 导数：
            dA/dZ = 1,    当 Z > 0
                    α,    当 Z <= 0

        因此：
            dZ = dA ⊙ g'(Z)

        
        """
        dZ = np.ones_like(self.Z)
        dZ[self.Z < 0] = self.alpha
        return dA * dZ


class SoftmaxCrossEntropy:
    def __init__(self):
        """
        Softmax + CrossEntropy 损失函数
        """
        self.probs = None
        self.y_true = None

    def forward(self, logits, y_true):
        """
        前向传播

        输入：
            logits.shape = (m, num_classes)
            y_true.shape = (m, num_classes)   # one-hot 标签

        Softmax：
            P_i = exp(z_i) / sum_j exp(z_j)

        交叉熵损失：
            L = -(1/m) Σ_i Σ_k y_ik log(p_ik)

        
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
        反向传播

        Softmax + CrossEntropy 联合求导结果可直接化简为：
            dZ = ∂L / ∂logits = (P - Y) / m

      
        """
        m = self.y_true.shape[0]
        return (self.probs - self.y_true) / m


class MLP:
    def __init__(self, input_dim, hidden_dims, output_dim,
                 activation="relu", weight_init="he"):
        """
        多层感知机（MLP）

        网络结构：
            输入层 -> [全连接 + 激活] × 若干层 -> 输出层全连接

        参数：
            input_dim  : 输入维度，例如 MNIST 为 784
            hidden_dims: 隐藏层维度列表，例如 [256, 128]
            output_dim : 输出维度，例如 10 类分类任务
            activation : 隐藏层激活函数
                         可选 "relu" / "sigmoid" / "tanh" / "leakyrelu"
            weight_init: 权重初始化方式
        """
        self.layers = []
        self.activations = []

        # 将所有层维度拼起来
        # 例如 [784] + [256, 128] + [10] -> [784, 256, 128, 10]
        dims = [input_dim] + hidden_dims + [output_dim]

        
        # 构造所有全连接层
        
        for i in range(len(dims) - 1):
            self.layers.append(
                FullyConnectedLayer(dims[i], dims[i + 1], weight_init=weight_init)
            )

        
        
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

        # 输出层使用 softmax + 交叉熵损失
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, X):
        """
        

        过程：
            对每个隐藏层：
                先过全连接层，再过激活函数
            最后一层：
                只过全连接层，输出 logits

        返回：
            logits
        """
        out = X

        # 隐藏层：Linear -> Activation
        for i in range(len(self.activations)):
            out = self.layers[i].forward(out)
            out = self.activations[i].forward(out)

        # 输出层：只做线性变换，不做激活
        logits = self.layers[-1].forward(out)
        return logits

    def compute_loss(self, logits, y_true, l2_lambda=0.0):
        """
        计算总损失

        总损失 = 数据损失 + L2 正则项

        数据损失：
            交叉熵损失

        正则项：
            L_reg = (λ / 2m) Σ ||W||²
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
        整个网络的反向传播

        

        
        数学过程：
            dZ_L = loss.backward()
            dA_{L-1} = FC_L.backward(dZ_L)

            对每个隐藏层从后往前：
                dZ_l = Activation_l.backward(dA_l)
                dA_{l-1} = FC_l.backward(dZ_l)
        """
       
        # 1. 输出层损失函数反向传播
        
        # 对 softmax + cross entropy：
        #   dZ = (P - Y) / m
        dZ = self.loss_fn.backward()

        
        # 2. 最后一层线性层反向传播
       
        # 输出层没有额外激活函数，直接过最后一个全连接层
        dA = self.layers[-1].backward(dZ, l2_lambda=l2_lambda)

        
        # 3. 按从后往前的顺序处理所有隐藏层
        
        for i in reversed(range(len(self.activations))):
            # 先过激活函数层
            # 输入：dA = ∂L/∂A
            # 输出：dZ = ∂L/∂Z
            dZ = self.activations[i].backward(dA)

            # 再过对应的线性层
            # 输入：dZ
            # 输出：传给更前一层的 dA
            dA = self.layers[i].backward(dZ, l2_lambda=l2_lambda)

    def update(self, lr):
        """
        更新所有层的参数
        """
        for layer in self.layers:
            layer.update(lr)

    def predict_proba(self, X):
        """
        输出类别概率
        """
        logits = self.forward(X)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def predict(self, X):
        """
        输出类别预测结果
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def get_gradient_norms(self, kind="weight"):
        """
        返回每一层的梯度范数列表

        参数：
            kind = "weight" -> 返回每层 ||dW||
            kind = "input"  -> 返回每层 ||dX||

        用途：
            可用于分析不同激活函数下的梯度传播情况，
            观察是否存在梯度消失或梯度爆炸
        """
        if kind == "weight":
            return [layer.last_grad_norm_w for layer in self.layers]
        elif kind == "input":
            return [layer.last_grad_norm_x for layer in self.layers]
        else:
            raise ValueError("kind must be 'weight' or 'input'")