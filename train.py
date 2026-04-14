# train.py
import os
import numpy as np
import matplotlib.pyplot as plt

from model import MLP
from utils import (
    ensure_dir,
    load_mnist_data,
    compute_accuracy,
    plot_curves,
    plot_conf_matrix,
    plot_wrong_cases,
)


def evaluate(model, X, y, y_oh, l2_lambda=0.0, batch_size=256):
    """
    在验证集/测试集上评估模型
    返回:
        avg_loss, acc, y_pred
    """
    total_loss = 0.0
    preds = []
    num_batches = 0
    n = X.shape[0]

    for start in range(0, n, batch_size):
        end = start + batch_size
        X_batch = X[start:end]
        y_batch_oh = y_oh[start:end]

        logits = model.forward(X_batch)
        loss = model.compute_loss(logits, y_batch_oh, l2_lambda=l2_lambda)
        total_loss += loss

        y_pred_batch = model.predict(X_batch)
        preds.append(y_pred_batch)
        num_batches += 1

    y_pred = np.concatenate(preds)
    acc = compute_accuracy(y_pred, y)
    avg_loss = total_loss / num_batches
    return avg_loss, acc, y_pred


def save_model_parameters(model, save_path):
    """
    保存模型参数到 npz 文件
    """
    save_dir = os.path.dirname(save_path)
    if save_dir:
        ensure_dir(save_dir)

    params = {}
    for i, layer in enumerate(model.layers):
        params[f"W{i}"] = layer.W
        params[f"b{i}"] = layer.b

    np.savez(save_path, **params)


def load_model_parameters(model, save_path):
    """
    从 npz 文件恢复模型参数
    """
    data = np.load(save_path)
    for i, layer in enumerate(model.layers):
        layer.W = data[f"W{i}"]
        layer.b = data[f"b{i}"]


def save_summary_txt(summary_text, save_path="results/hparam_summary.txt"):
    save_dir = os.path.dirname(save_path)
    if save_dir:
        ensure_dir(save_dir)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(summary_text)


def train_one_experiment(
    exp_name,
    display_name,
    config,
    X_train, y_train, y_train_oh,
    X_val, y_val, y_val_oh,
    patience=5,
    min_delta=1e-4
):
    """
    训练单组实验
    """
    model = MLP(
        input_dim=config["input_dim"],
        hidden_dims=config["hidden_dims"],
        output_dim=config["output_dim"],
        activation=config["activation"],
        weight_init=config["weight_init"]
    )

    history = {
        "train_losses": [],
        "val_losses": [],
        "train_accs": [],
        "val_accs": [],
    }

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = -1
    wait = 0

    best_model_path = os.path.join("results", f"best_model_{exp_name}.npz")

    n_train = X_train.shape[0]
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]
    l2_lambda = config["l2_lambda"]

    print(f"\n===== {display_name} =====")
    print(
        f"hidden_dims={config['hidden_dims']}, "
        f"activation={config['activation']}, "
        f"lr={lr}, batch_size={batch_size}, l2={l2_lambda}, "
        f"weight_init={config['weight_init']}"
    )

    for epoch in range(epochs):
        indices = np.random.permutation(n_train)

        epoch_loss = 0.0
        num_batches = 0
        train_preds = []
        train_true = []

        for start in range(0, n_train, batch_size):
            end = start + batch_size
            idx = indices[start:end]

            X_batch = X_train[idx]
            y_batch = y_train[idx]
            y_batch_oh = y_train_oh[idx]

            # 前向传播
            logits = model.forward(X_batch)

            # 损失
            loss = model.compute_loss(logits, y_batch_oh, l2_lambda=l2_lambda)
            epoch_loss += loss

            # 当前 batch 预测结果
            batch_pred = np.argmax(model.loss_fn.probs, axis=1)
            train_preds.append(batch_pred)
            train_true.append(y_batch)

            # 反向传播与参数更新
            model.backward(l2_lambda=l2_lambda)
            model.update(lr)

            num_batches += 1

        # 训练集指标
        train_loss = epoch_loss / num_batches
        train_pred = np.concatenate(train_preds)
        train_true_epoch = np.concatenate(train_true)
        train_acc = compute_accuracy(train_pred, train_true_epoch)

        # 验证集指标
        val_loss, val_acc, _ = evaluate(
            model,
            X_val, y_val, y_val_oh,
            l2_lambda=l2_lambda,
            batch_size=256
        )

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["train_accs"].append(train_acc)
        history["val_accs"].append(val_acc)

        print(
            f"[{display_name}] Epoch {epoch+1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # 以验证损失作为 early stopping 标准，同时记录最佳验证准确率
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1
            wait = 0
            save_model_parameters(model, best_model_path)
        else:
            wait += 1

        if wait >= patience:
            print(f"[{display_name}] Early stopping at epoch {epoch+1}.")
            break

    # 恢复最佳模型参数
    load_model_parameters(model, best_model_path)

    result = {
        "exp_name": exp_name,
        "display_name": display_name,
        "config": config,
        "model": model,
        "history": history,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_model_path": best_model_path,
    }
    return result


def plot_three_mode_val_acc_comparison(results, save_path):
    """
    在一张图中比较三组参数设计模式的验证集准确率曲线
    """
    save_dir = os.path.dirname(save_path)
    if save_dir:
        ensure_dir(save_dir)

    plt.figure(figsize=(9, 5))

    for result in results:
        val_accs = result["history"]["val_accs"]
        epochs = np.arange(1, len(val_accs) + 1)
        label = f"{result['display_name']} | best={result['best_val_acc']:.4f}"
        plt.plot(epochs, val_accs, marker="o", label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Comparison of Three Hyperparameter Designs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_three_mode_val_loss_comparison(results, save_path):
    """
    在一张图中比较三组参数设计模式的验证集损失曲线
    """
    save_dir = os.path.dirname(save_path)
    if save_dir:
        ensure_dir(save_dir)

    plt.figure(figsize=(9, 5))

    for result in results:
        val_losses = result["history"]["val_losses"]
        epochs = np.arange(1, len(val_losses) + 1)
        plt.plot(epochs, val_losses, marker="o", label=result["display_name"])

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Comparison of Three Hyperparameter Designs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_overfitting_control_comparison(result_no_l2, result_with_l2, save_path):
    """
    用 Baseline 同参数下 no_L2 / with_L2 的训练-验证损失对照，
    体现 L2 正则化对过拟合的控制效果
    """
    save_dir = os.path.dirname(save_path)
    if save_dir:
        ensure_dir(save_dir)

    epochs_no = np.arange(1, len(result_no_l2["history"]["train_losses"]) + 1)
    epochs_l2 = np.arange(1, len(result_with_l2["history"]["train_losses"]) + 1)

    plt.figure(figsize=(10, 5))

    plt.plot(
        epochs_no,
        result_no_l2["history"]["train_losses"],
        linestyle="--",
        label="Train Loss (no L2)"
    )
    plt.plot(
        epochs_no,
        result_no_l2["history"]["val_losses"],
        label="Val Loss (no L2)"
    )

    plt.plot(
        epochs_l2,
        result_with_l2["history"]["train_losses"],
        linestyle="--",
        label="Train Loss (with L2)"
    )
    plt.plot(
        epochs_l2,
        result_with_l2["history"]["val_losses"],
        label="Val Loss (with L2)"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss Curves for Overfitting Control")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    np.random.seed(42)
    ensure_dir("results")

    # =========================
    # 1. 加载数据
    # =========================
    X_train, y_train, y_train_oh, X_val, y_val, y_val_oh, X_test, y_test, y_test_oh = load_mnist_data()

    # =========================
    # 2. 按题目表格设置三组实验
    # =========================
    experiments = [
        {
            "exp_name": "baseline",
            "display_name": "Baseline",
            "config": {
                "input_dim": 784,
                "hidden_dims": [256, 128],
                "output_dim": 10,
                "activation": "relu",
                "weight_init": "he",
                "lr": 0.01,
                "batch_size": 64,
                "epochs": 25,
                "l2_lambda": 1e-4,
            }
        },
        {
            "exp_name": "group1",
            "display_name": "组1",
            "config": {
                "input_dim": 784,
                "hidden_dims": [512, 256],
                "output_dim": 10,
                "activation": "leakyrelu",
                "weight_init": "he",
                "lr": 0.005,
                "batch_size": 128,
                "epochs": 25,
                "l2_lambda": 1e-3,
            }
        },
        {
            "exp_name": "group2",
            "display_name": "组2",
            "config": {
                "input_dim": 784,
                "hidden_dims": [128],
                "output_dim": 10,
                "activation": "tanh",
                "weight_init": "xavier",
                "lr": 0.02,
                "batch_size": 32,
                "epochs": 25,
                "l2_lambda": 0.0,
            }
        },
    ]

    # =========================
    # 3. 依次训练三组实验
    # =========================
    all_results = []

    for exp in experiments:
        result = train_one_experiment(
            exp_name=exp["exp_name"],
            display_name=exp["display_name"],
            config=exp["config"],
            X_train=X_train, y_train=y_train, y_train_oh=y_train_oh,
            X_val=X_val, y_val=y_val, y_val_oh=y_val_oh,
            patience=5,
            min_delta=1e-4
        )

        # 保存单组训练/验证曲线
        plot_curves(
            result["history"]["train_losses"],
            result["history"]["val_losses"],
            result["history"]["train_accs"],
            result["history"]["val_accs"],
            save_dir="results",
            prefix=exp["exp_name"]
        )

        # 测试集评估
        test_loss, test_acc, _ = evaluate(
            result["model"],
            X_test, y_test, y_test_oh,
            l2_lambda=exp["config"]["l2_lambda"]
        )

        result["test_loss"] = test_loss
        result["test_acc"] = test_acc
        all_results.append(result)

    # =========================
    # 4. 在一张图中比较三种参数设计模式
    # =========================
    plot_three_mode_val_acc_comparison(
        all_results,
        save_path="results/three_mode_val_acc_compare.png"
    )

    plot_three_mode_val_loss_comparison(
        all_results,
        save_path="results/three_mode_val_loss_compare.png"
    )

    # =========================
    # 5. 过拟合控制：Baseline 同参数下 no_L2 vs with_L2
    # =========================
    baseline_no_l2_config = {
        "input_dim": 784,
        "hidden_dims": [256, 128],
        "output_dim": 10,
        "activation": "relu",
        "weight_init": "he",
        "lr": 0.01,
        "batch_size": 64,
        "epochs": 25,
        "l2_lambda": 0.0,
    }

    baseline_with_l2_config = {
        "input_dim": 784,
        "hidden_dims": [256, 128],
        "output_dim": 10,
        "activation": "relu",
        "weight_init": "he",
        "lr": 0.01,
        "batch_size": 64,
        "epochs": 25,
        "l2_lambda": 1e-4,
    }

    result_no_l2 = train_one_experiment(
        exp_name="baseline_no_l2",
        display_name="Baseline-no_L2",
        config=baseline_no_l2_config,
        X_train=X_train, y_train=y_train, y_train_oh=y_train_oh,
        X_val=X_val, y_val=y_val, y_val_oh=y_val_oh,
        patience=5,
        min_delta=1e-4
    )

    result_with_l2 = train_one_experiment(
        exp_name="baseline_with_l2",
        display_name="Baseline-with_L2",
        config=baseline_with_l2_config,
        X_train=X_train, y_train=y_train, y_train_oh=y_train_oh,
        X_val=X_val, y_val=y_val, y_val_oh=y_val_oh,
        patience=5,
        min_delta=1e-4
    )

    # 单独保存两组的 train/val 曲线
    plot_curves(
        result_no_l2["history"]["train_losses"],
        result_no_l2["history"]["val_losses"],
        result_no_l2["history"]["train_accs"],
        result_no_l2["history"]["val_accs"],
        save_dir="results",
        prefix="baseline_no_l2"
    )

    plot_curves(
        result_with_l2["history"]["train_losses"],
        result_with_l2["history"]["val_losses"],
        result_with_l2["history"]["train_accs"],
        result_with_l2["history"]["val_accs"],
        save_dir="results",
        prefix="baseline_with_l2"
    )

    # 一张图体现过拟合控制
    plot_overfitting_control_comparison(
        result_no_l2,
        result_with_l2,
        save_path="results/overfitting_control_loss_compare.png"
    )

    # =========================
    # 6. 选三组参数设计模式中“验证集最佳”的模型，做混淆矩阵和错误案例可视化
    # =========================
    best_result = max(all_results, key=lambda x: x["best_val_acc"])
    best_l2 = best_result["config"]["l2_lambda"]

    best_test_loss, best_test_acc, best_y_pred = evaluate(
        best_result["model"],
        X_test, y_test, y_test_oh,
        l2_lambda=best_l2
    )

    plot_conf_matrix(
        y_true=y_test,
        y_pred=best_y_pred,
        save_path="results/confusion_matrix_best.png",
        title=f"Confusion Matrix ({best_result['display_name']})"
    )

    plot_wrong_cases(
        X=X_test,
        y_true=y_test,
        y_pred=best_y_pred,
        save_path="results/wrong_cases_best.png",
        num_show=16,
        title=f"Wrong Cases ({best_result['display_name']})"
    )

    # =========================
    # 7. 输出实验汇总
    # =========================
    summary = []
    summary.append("===== Hyperparameter Tuning Experiments =====\n")
    summary.append("实验组 | 隐藏层结构 | 激活函数 | 学习率 | Batch Size | L2系数 | 最佳验证集准确率 | 测试集准确率")
    summary.append("-" * 120)

    for result in all_results:
        cfg = result["config"]
        summary.append(
            f"{result['display_name']} | "
            f"{cfg['hidden_dims']} | "
            f"{cfg['activation']} | "
            f"{cfg['lr']} | "
            f"{cfg['batch_size']} | "
            f"{cfg['l2_lambda']} | "
            f"{result['best_val_acc']:.4f} | "
            f"{result['test_acc']:.4f}"
        )

    summary.append("\n=== Overfitting Control (Baseline no_L2 vs with_L2) ===")
    summary.append(
        f"Baseline-no_L2   | best_val_acc={result_no_l2['best_val_acc']:.4f} | best_val_loss={result_no_l2['best_val_loss']:.4f}"
    )
    summary.append(
        f"Baseline-with_L2 | best_val_acc={result_with_l2['best_val_acc']:.4f} | best_val_loss={result_with_l2['best_val_loss']:.4f}"
    )

    summary.append("\n=== Confusion Matrix & Wrong Cases ===")
    summary.append(f"Best hyperparameter design: {best_result['display_name']}")
    summary.append(f"Best model test loss      : {best_test_loss:.4f}")
    summary.append(f"Best model test accuracy  : {best_test_acc:.4f}")

    summary.append("\nSaved Figures:")
    summary.append("- results/three_mode_val_acc_compare.png")
    summary.append("- results/three_mode_val_loss_compare.png")
    summary.append("- results/baseline_loss_curve.png")
    summary.append("- results/baseline_acc_curve.png")
    summary.append("- results/group1_loss_curve.png")
    summary.append("- results/group1_acc_curve.png")
    summary.append("- results/group2_loss_curve.png")
    summary.append("- results/group2_acc_curve.png")
    summary.append("- results/baseline_no_l2_loss_curve.png")
    summary.append("- results/baseline_no_l2_acc_curve.png")
    summary.append("- results/baseline_with_l2_loss_curve.png")
    summary.append("- results/baseline_with_l2_acc_curve.png")
    summary.append("- results/overfitting_control_loss_compare.png")
    summary.append("- results/confusion_matrix_best.png")
    summary.append("- results/wrong_cases_best.png")

    summary_text = "\n".join(summary)
    print("\n" + summary_text)
    save_summary_txt(summary_text, save_path="results/hparam_summary.txt")


if __name__ == "__main__":
    main()