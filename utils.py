# utils.py
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def ensure_dir(path):
    """
    确保目录存在
    """
    os.makedirs(path, exist_ok=True)


def one_hot_encode(y, num_classes=10):
    y = y.astype(int)
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot


def load_mnist_data(test_size=10000, val_size=0.2, random_state=42):
    """
    加载 MNIST，并划分 train / val / test
    """
    print("Loading MNIST from OpenML...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)

    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int64)

    # 先划分出测试集
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 再从 trainval 中划分验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size,
        random_state=random_state,
        stratify=y_trainval
    )

    y_train_oh = one_hot_encode(y_train)
    y_val_oh = one_hot_encode(y_val)
    y_test_oh = one_hot_encode(y_test)

    return X_train, y_train, y_train_oh, X_val, y_val, y_val_oh, X_test, y_test, y_test_oh


def get_batches(X, y, batch_size=64, shuffle=True):
    """
    生成 mini-batch
    """
    n = X.shape[0]
    indices = np.arange(n)

    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, n, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]


def compute_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def plot_curves(train_losses, val_losses, train_accs, val_accs, save_dir="results", prefix=""):
    """
    单次实验的 loss / acc 曲线
    """
    ensure_dir(save_dir)
    epochs = np.arange(1, len(train_losses) + 1)

    prefix = f"{prefix}_" if prefix else ""

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}loss_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}acc_curve.png"), dpi=200)
    plt.close()


def plot_conf_matrix(y_true, y_pred, save_path, title="Confusion Matrix"):
    """
    绘制混淆矩阵
    """
    save_dir = os.path.dirname(save_path)
    if save_dir:
        ensure_dir(save_dir)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_wrong_cases(X, y_true, y_pred, save_path, num_show=16, title="Wrongly Classified Examples"):
    """
    可视化分类错误案例
    """
    save_dir = os.path.dirname(save_path)
    if save_dir:
        ensure_dir(save_dir)

    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) == 0:
        print(f"No wrong cases found for {title}.")
        return

    num_show = min(num_show, len(wrong_idx))
    chosen_idx = wrong_idx[:num_show]

    rows = int(np.ceil(num_show / 4))
    cols = 4

    plt.figure(figsize=(10, 2.5 * rows))
    for i, idx in enumerate(chosen_idx):
        plt.subplot(rows, cols, i + 1)

        img = X[idx]
        if img.ndim == 1:
            img = img.reshape(28, 28)

        plt.imshow(img, cmap="gray")
        plt.title(f"T:{y_true[idx]}  P:{y_pred[idx]}")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def mean_grad_norms_from_batches(batch_grad_norms):
    """
    对一个 epoch 内所有 batch 的梯度范数做平均

    参数
    ----
    batch_grad_norms: list
        形如:
        [
            [layer1_norm, layer2_norm, layer3_norm],
            [layer1_norm, layer2_norm, layer3_norm],
            ...
        ]

    返回
    ----
    np.ndarray, shape=(num_layers,)
    """
    arr = np.array(batch_grad_norms, dtype=np.float64)
    return np.mean(arr, axis=0)


def plot_activation_gradient_comparison(
    activation_grad_history,
    save_dir="results",
    log_scale=True
):
    """
    比较不同激活函数对梯度消失的影响，并直接生成图像

    参数
    ----
    activation_grad_history: dict
        键: 激活函数名称，如 "relu", "sigmoid", "tanh", "leakyrelu"
        值: shape=(num_epochs, num_layers) 的数组，或等价 list
            每一行表示“某个 epoch 内，各层平均梯度范数”

        例如:
        {
            "relu": np.array([[...layer norms...], [...], ...]),
            "sigmoid": np.array([[...], [...], ...]),
            "tanh": np.array([[...], [...], ...])
        }

    save_dir: 图片保存目录
    log_scale: 是否使用对数坐标，推荐 True，更容易观察梯度消失
    """
    ensure_dir(save_dir)

    # 统一转成 ndarray
    processed = {}
    for act_name, history in activation_grad_history.items():
        history = np.array(history, dtype=np.float64)
        if history.ndim != 2:
            raise ValueError(
                f"activation_grad_history['{act_name}'] must have shape (num_epochs, num_layers)"
            )
        processed[act_name] = history

    # ===== 图1：各层平均梯度范数比较 =====
    plt.figure(figsize=(8, 5))
    for act_name, history in processed.items():
        mean_by_layer = np.mean(history, axis=0)  # 对 epoch 求平均
        layers = np.arange(1, len(mean_by_layer) + 1)
        plt.plot(layers, mean_by_layer, marker="o", label=act_name)

    plt.xlabel("Layer Index")
    plt.ylabel("Average Gradient Norm")
    plt.title("Gradient Norm by Layer for Different Activations")
    if log_scale:
        plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "activation_gradient_by_layer.png"), dpi=200)
    plt.close()

    # ===== 图2：第一层梯度范数随 epoch 变化 =====
    plt.figure(figsize=(8, 5))
    for act_name, history in processed.items():
        epochs = np.arange(1, history.shape[0] + 1)
        first_layer_grad = history[:, 0]
        plt.plot(epochs, first_layer_grad, marker="o", label=act_name)

    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm of First Layer")
    plt.title("First-Layer Gradient Norm Across Epochs")
    if log_scale:
        plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "activation_first_layer_gradient.png"), dpi=200)
    plt.close()


def plot_activation_loss_comparison(activation_results, save_dir="results"):
    """
    可选：比较不同激活函数的验证损失曲线

    参数
    ----
    activation_results: dict
        例如:
        {
            "relu": {"val_losses": [...]},
            "sigmoid": {"val_losses": [...]},
            "tanh": {"val_losses": [...]}
        }
    """
    ensure_dir(save_dir)

    plt.figure(figsize=(8, 5))
    for act_name, result in activation_results.items():
        val_losses = np.array(result["val_losses"], dtype=np.float64)
        epochs = np.arange(1, len(val_losses) + 1)
        plt.plot(epochs, val_losses, marker="o", label=act_name)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss for Different Activations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "activation_val_loss_compare.png"), dpi=200)
    plt.close()


def plot_activation_acc_comparison(activation_results, save_dir="results"):
    """
    可选：比较不同激活函数的验证准确率曲线
    """
    ensure_dir(save_dir)

    plt.figure(figsize=(8, 5))
    for act_name, result in activation_results.items():
        val_accs = np.array(result["val_accs"], dtype=np.float64)
        epochs = np.arange(1, len(val_accs) + 1)
        plt.plot(epochs, val_accs, marker="o", label=act_name)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy for Different Activations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "activation_val_acc_compare.png"), dpi=200)
    plt.close()