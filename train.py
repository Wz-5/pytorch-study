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
    mean_grad_norms_from_batches,
    plot_activation_gradient_comparison,
    plot_activation_loss_comparison,
    plot_activation_acc_comparison,
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
    保存模型参数到 npz
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
    从 npz 加载模型参数
    """
    data = np.load(save_path)
    for i, layer in enumerate(model.layers):
        layer.W = data[f"W{i}"]
        layer.b = data[f"b{i}"]


def save_summary_txt(summary_text, save_path="results/summary.txt"):
    ensure_dir(os.path.dirname(save_path) if os.path.dirname(save_path) else ".")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(summary_text)


def plot_loss_comparison(result_a, result_b, save_path, title):
    """
    两组实验的 loss 对比图
    """
    ensure_dir(os.path.dirname(save_path) if os.path.dirname(save_path) else ".")

    train_a = result_a["history"]["train_losses"]
    val_a = result_a["history"]["val_losses"]
    train_b = result_b["history"]["train_losses"]
    val_b = result_b["history"]["val_losses"]

    epochs_a = np.arange(1, len(train_a) + 1)
    epochs_b = np.arange(1, len(train_b) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs_a, train_a, label=f"Train Loss ({result_a['exp_name']})", linestyle="--")
    plt.plot(epochs_a, val_a, label=f"Val Loss ({result_a['exp_name']})")
    plt.plot(epochs_b, train_b, label=f"Train Loss ({result_b['exp_name']})", linestyle="--")
    plt.plot(epochs_b, val_b, label=f"Val Loss ({result_b['exp_name']})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_acc_comparison(result_a, result_b, save_path, title):
    """
    两组实验的 acc 对比图
    """
    ensure_dir(os.path.dirname(save_path) if os.path.dirname(save_path) else ".")

    train_a = result_a["history"]["train_accs"]
    val_a = result_a["history"]["val_accs"]
    train_b = result_b["history"]["train_accs"]
    val_b = result_b["history"]["val_accs"]

    epochs_a = np.arange(1, len(train_a) + 1)
    epochs_b = np.arange(1, len(train_b) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs_a, train_a, label=f"Train Acc ({result_a['exp_name']})", linestyle="--")
    plt.plot(epochs_a, val_a, label=f"Val Acc ({result_a['exp_name']})")
    plt.plot(epochs_b, train_b, label=f"Train Acc ({result_b['exp_name']})", linestyle="--")
    plt.plot(epochs_b, val_b, label=f"Val Acc ({result_b['exp_name']})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def train_one_experiment(
    exp_name,
    X_train, y_train, y_train_oh,
    X_val, y_val, y_val_oh,
    config,
    activation=None,
    weight_init=None,
    l2_lambda=0.0,
    patience=5,
    min_delta=1e-4,
    track_gradients=False,
):
    """
    训练单组实验

    参数
    ----
    activation: 若为 None，则使用 config["activation"]
    weight_init: 若为 None，则使用 config["weight_init"]
    track_gradients: 是否记录每个 epoch 的各层平均梯度范数
    """
    act = activation if activation is not None else config["activation"]
    init = weight_init if weight_init is not None else config["weight_init"]

    model = MLP(
        input_dim=config["input_dim"],
        hidden_dims=config["hidden_dims"],
        output_dim=config["output_dim"],
        activation=act,
        weight_init=init,
    )

    history = {
        "train_losses": [],
        "val_losses": [],
        "train_accs": [],
        "val_accs": [],
    }

    gradient_history = []

    best_val_loss = float("inf")
    best_epoch = -1
    wait = 0

    best_model_path = os.path.join("results", f"best_model_{exp_name}.npz")

    n_train = X_train.shape[0]
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]

    print(f"\n===== Start: {exp_name} =====")
    print(
        f"activation={act}, weight_init={init}, "
        f"hidden_dims={config['hidden_dims']}, lr={lr}, "
        f"batch_size={batch_size}, epochs={epochs}, l2={l2_lambda}"
    )

    for epoch in range(epochs):
        indices = np.random.permutation(n_train)

        epoch_loss = 0.0
        num_batches = 0

        train_preds = []
        train_true = []

        batch_grad_norms = []

        for start in range(0, n_train, batch_size):
            end = start + batch_size
            idx = indices[start:end]

            X_batch = X_train[idx]
            y_batch = y_train[idx]
            y_batch_oh = y_train_oh[idx]

            # forward
            logits = model.forward(X_batch)

            # loss
            loss = model.compute_loss(logits, y_batch_oh, l2_lambda=l2_lambda)
            epoch_loss += loss

            # 当前 batch 预测
            batch_pred = np.argmax(model.loss_fn.probs, axis=1)
            train_preds.append(batch_pred)
            train_true.append(y_batch)

            # backward
            model.backward(l2_lambda=l2_lambda)

            # 记录梯度范数（分析梯度消失/爆炸）
            if track_gradients:
                batch_grad_norms.append(model.get_gradient_norms(kind="weight"))

            # update
            model.update(lr)

            num_batches += 1

        train_loss = epoch_loss / num_batches
        train_pred = np.concatenate(train_preds)
        train_true_epoch = np.concatenate(train_true)
        train_acc = compute_accuracy(train_pred, train_true_epoch)

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

        if track_gradients:
            epoch_grad_norm = mean_grad_norms_from_batches(batch_grad_norms)
            gradient_history.append(epoch_grad_norm)

        print(
            f"[{exp_name}] Epoch {epoch+1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Early Stopping + 保存最佳模型
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            wait = 0
            save_model_parameters(model, best_model_path)
        else:
            wait += 1

        if wait >= patience:
            print(f"[{exp_name}] Early stopping triggered at epoch {epoch+1}.")
            break

    # 恢复最佳模型参数
    load_model_parameters(model, best_model_path)

    result = {
        "exp_name": exp_name,
        "model": model,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_model_path": best_model_path,
        "activation": act,
        "weight_init": init,
        "l2_lambda": l2_lambda,
        "gradient_history": np.array(gradient_history) if track_gradients else None,
    }
    return result


def main():
    np.random.seed(42)
    ensure_dir("results")

    # =========================
    # 1) 加载数据
    # =========================
    X_train, y_train, y_train_oh, X_val, y_val, y_val_oh, X_test, y_test, y_test_oh = load_mnist_data()

    # =========================
    # 2) 正则化对比：no_L2 vs with_L2
    # =========================
    reg_config = {
        "input_dim": 784,
        "hidden_dims": [256, 128],
        "output_dim": 10,
        "activation": "relu",
        "weight_init": "he",
        "lr": 0.08,
        "batch_size": 128,
        "epochs": 30,
    }

    result_no_l2 = train_one_experiment(
        exp_name="no_l2",
        X_train=X_train, y_train=y_train, y_train_oh=y_train_oh,
        X_val=X_val, y_val=y_val, y_val_oh=y_val_oh,
        config=reg_config,
        activation="relu",
        weight_init="he",
        l2_lambda=0.0,
        patience=5,
        min_delta=1e-4,
        track_gradients=False,
    )

    result_with_l2 = train_one_experiment(
        exp_name="with_l2",
        X_train=X_train, y_train=y_train, y_train_oh=y_train_oh,
        X_val=X_val, y_val=y_val, y_val_oh=y_val_oh,
        config=reg_config,
        activation="relu",
        weight_init="he",
        l2_lambda=1e-4,
        patience=5,
        min_delta=1e-4,
        track_gradients=False,
    )

    # 单组曲线
    plot_curves(
        result_no_l2["history"]["train_losses"],
        result_no_l2["history"]["val_losses"],
        result_no_l2["history"]["train_accs"],
        result_no_l2["history"]["val_accs"],
        save_dir="results",
        prefix="no_l2"
    )

    plot_curves(
        result_with_l2["history"]["train_losses"],
        result_with_l2["history"]["val_losses"],
        result_with_l2["history"]["train_accs"],
        result_with_l2["history"]["val_accs"],
        save_dir="results",
        prefix="with_l2"
    )

    # 对比曲线
    plot_loss_comparison(
        result_no_l2,
        result_with_l2,
        save_path="results/loss_compare_l2.png",
        title="Loss Curves: no_L2 vs with_L2"
    )

    plot_acc_comparison(
        result_no_l2,
        result_with_l2,
        save_path="results/acc_compare_l2.png",
        title="Accuracy Curves: no_L2 vs with_L2"
    )

    # 测试集评估
    test_loss_no_l2, test_acc_no_l2, y_pred_no_l2 = evaluate(
        result_no_l2["model"], X_test, y_test, y_test_oh, l2_lambda=0.0
    )
    test_loss_with_l2, test_acc_with_l2, y_pred_with_l2 = evaluate(
        result_with_l2["model"], X_test, y_test, y_test_oh, l2_lambda=1e-4
    )

    # 混淆矩阵
    plot_conf_matrix(
        y_true=y_test,
        y_pred=y_pred_no_l2,
        save_path="results/confusion_matrix_no_l2.png",
        title="Confusion Matrix (no L2)"
    )

    plot_conf_matrix(
        y_true=y_test,
        y_pred=y_pred_with_l2,
        save_path="results/confusion_matrix_with_l2.png",
        title="Confusion Matrix (with L2)"
    )

    # 错分样本
    plot_wrong_cases(
        X=X_test,
        y_true=y_test,
        y_pred=y_pred_no_l2,
        save_path="results/wrong_cases_no_l2.png",
        num_show=16,
        title="Wrong Cases (no L2)"
    )

    plot_wrong_cases(
        X=X_test,
        y_true=y_test,
        y_pred=y_pred_with_l2,
        save_path="results/wrong_cases_with_l2.png",
        num_show=16,
        title="Wrong Cases (with L2)"
    )

    # =========================
    # 3) 激活函数对梯度消失的影响比较
    # =========================
    # 为了更明显观察梯度传播差异，这里使用更深一点的 MLP
    # 并统一使用 Xavier 初始化，以便突出“激活函数本身”的影响
    act_config = {
        "input_dim": 784,
        "hidden_dims": [256, 128, 64],
        "output_dim": 10,
        "activation": "relu",      # 这里只是占位，实际会在循环中覆盖
        "weight_init": "xavier",   # 统一初始化，便于公平比较
        "lr": 0.05,
        "batch_size": 128,
        "epochs": 12,
    }

    activation_list = ["sigmoid", "tanh", "relu", "leakyrelu"]
    activation_results = {}
    activation_grad_history = {}
    activation_test_metrics = {}

    for act_name in activation_list:
        result_act = train_one_experiment(
            exp_name=f"act_{act_name}",
            X_train=X_train, y_train=y_train, y_train_oh=y_train_oh,
            X_val=X_val, y_val=y_val, y_val_oh=y_val_oh,
            config=act_config,
            activation=act_name,
            weight_init="xavier",
            l2_lambda=0.0,
            patience=4,
            min_delta=1e-4,
            track_gradients=True,
        )

        activation_results[act_name] = {
            "val_losses": result_act["history"]["val_losses"],
            "val_accs": result_act["history"]["val_accs"],
        }
        activation_grad_history[act_name] = result_act["gradient_history"]

        test_loss_act, test_acc_act, _ = evaluate(
            result_act["model"], X_test, y_test, y_test_oh, l2_lambda=0.0
        )
        activation_test_metrics[act_name] = {
            "test_loss": test_loss_act,
            "test_acc": test_acc_act,
            "best_epoch": result_act["best_epoch"],
            "best_val_loss": result_act["best_val_loss"],
        }

        # 单组曲线也保存下来
        plot_curves(
            result_act["history"]["train_losses"],
            result_act["history"]["val_losses"],
            result_act["history"]["train_accs"],
            result_act["history"]["val_accs"],
            save_dir="results",
            prefix=f"act_{act_name}"
        )

    # 画激活函数对比图
    plot_activation_gradient_comparison(
        activation_grad_history,
        save_dir="results",
        log_scale=True
    )

    plot_activation_loss_comparison(
        activation_results,
        save_dir="results"
    )

    plot_activation_acc_comparison(
        activation_results,
        save_dir="results"
    )

    # =========================
    # 4) 结果汇总 summary.txt
    # =========================
    summary = []
    summary.append("===== Experiment Summary =====\n")

    summary.append("=== 1. Regularization Comparison ===")
    summary.append("Experiment: no_L2")
    summary.append(f"Best Epoch      : {result_no_l2['best_epoch']}")
    summary.append(f"Best Val Loss   : {result_no_l2['best_val_loss']:.6f}")
    summary.append(f"Test Loss       : {test_loss_no_l2:.6f}")
    summary.append(f"Test Accuracy   : {test_acc_no_l2:.6f}")
    summary.append(f"Best Model Path : {result_no_l2['best_model_path']}\n")

    summary.append("Experiment: with_L2")
    summary.append(f"Best Epoch      : {result_with_l2['best_epoch']}")
    summary.append(f"Best Val Loss   : {result_with_l2['best_val_loss']:.6f}")
    summary.append(f"Test Loss       : {test_loss_with_l2:.6f}")
    summary.append(f"Test Accuracy   : {test_acc_with_l2:.6f}")
    summary.append(f"Best Model Path : {result_with_l2['best_model_path']}\n")

    summary.append("Saved Figures:")
    summary.append("- results/no_l2_loss_curve.png")
    summary.append("- results/no_l2_acc_curve.png")
    summary.append("- results/with_l2_loss_curve.png")
    summary.append("- results/with_l2_acc_curve.png")
    summary.append("- results/loss_compare_l2.png")
    summary.append("- results/acc_compare_l2.png")
    summary.append("- results/confusion_matrix_no_l2.png")
    summary.append("- results/confusion_matrix_with_l2.png")
    summary.append("- results/wrong_cases_no_l2.png")
    summary.append("- results/wrong_cases_with_l2.png\n")

    summary.append("=== 2. Activation Comparison (Gradient Vanishing) ===")
    summary.append("To highlight the influence of activation functions on gradient propagation,")
    summary.append("all activations use the same network depth and Xavier initialization.\n")

    for act_name in activation_list:
        metrics = activation_test_metrics[act_name]
        grad_hist = activation_grad_history[act_name]

        if grad_hist is not None and len(grad_hist) > 0:
            avg_grad_by_layer = np.mean(grad_hist, axis=0)
            grad_str = np.array2string(avg_grad_by_layer, precision=6, separator=", ")
        else:
            grad_str = "None"

        summary.append(f"Activation: {act_name}")
        summary.append(f"Best Epoch         : {metrics['best_epoch']}")
        summary.append(f"Best Val Loss      : {metrics['best_val_loss']:.6f}")
        summary.append(f"Test Loss          : {metrics['test_loss']:.6f}")
        summary.append(f"Test Accuracy      : {metrics['test_acc']:.6f}")
        summary.append(f"Avg Grad By Layer  : {grad_str}")
        summary.append("")

    summary.append("Saved Figures:")
    summary.append("- results/activation_gradient_by_layer.png")
    summary.append("- results/activation_first_layer_gradient.png")
    summary.append("- results/activation_val_loss_compare.png")
    summary.append("- results/activation_val_acc_compare.png")
    for act_name in activation_list:
        summary.append(f"- results/act_{act_name}_loss_curve.png")
        summary.append(f"- results/act_{act_name}_acc_curve.png")

    summary_text = "\n".join(summary)
    print("\n" + summary_text)
    save_summary_txt(summary_text, save_path="results/summary.txt")


if __name__ == "__main__":
    main()