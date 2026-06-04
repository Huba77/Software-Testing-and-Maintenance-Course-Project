
"""
可视化分析模块：z-space分布、重构曲线对比、ROC/F-Score曲线
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve, precision_recall_curve


def plot_ablation_comparison(all_scores, all_losses, timestamps):
    """
    绘制消融实验对比图：
    - 各变体的loss曲线
    - 各变体的anomaly score曲线对比
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 1. Loss曲线对比
    ax1 = axes[0]
    colors = {
        "VAE Baseline": "#1f77b4",
        "M-ELBO": "#ff7f0e",
        "+Injection": "#2ca02c",
        "+MCMC": "#d62728",
        "Full Donut": "#9467bd"
    }
    
    for name, losses in all_losses.items():
        if name == "Full Donut":
            continue
        ax1.plot(losses, label=name, color=colors.get(name, "gray"), linewidth=1.5)
    
    ax1.set_title("Training Loss Comparison (Ablation Study)", fontsize=14)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    
    # 2. Anomaly Score曲线对比
    ax2 = axes[1]
    for name, scores in all_scores.items():
        if name == "Full Donut":
            continue
        ax2.plot(scores, label=name, color=colors.get(name, "gray"), linewidth=1.0, alpha=0.8)
    
    ax2.set_title("Anomaly Score Comparison (Ablation Study)", fontsize=14)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Anomaly Score")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("ablation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved ablation_comparison.png")


def plot_roc_comparison(y_true, all_scores):
    """
    绘制各变体的ROC曲线对比。
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {
        "VAE Baseline": "#1f77b4",
        "M-ELBO": "#ff7f0e",
        "+Injection": "#2ca02c",
        "+MCMC": "#d62728",
        "Full Donut": "#9467bd"
    }
    
    # ROC曲线
    ax1 = axes[0]
    for name, scores in all_scores.items():
        if name == "Full Donut":
            continue
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, scores)
        from sklearn.metrics import auc
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", 
                 color=colors.get(name, "gray"), linewidth=1.5)
    
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax1.set_title("ROC Curve Comparison", fontsize=14)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall曲线
    ax2 = axes[1]
    for name, scores in all_scores.items():
        if name == "Full Donut":
            continue
        if len(np.unique(y_true)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ax2.plot(recall, precision, label=name,
                 color=colors.get(name, "gray"), linewidth=1.5)
    
    ax2.set_title("Precision-Recall Curve Comparison", fontsize=14)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("roc_pr_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved roc_pr_comparison.png")


def plot_latent_space(model, data_loader, save_path="latent_space.png"):
    """
    绘制 latent space (z-space) 分布，类似论文 Figure 10/12 风格。
    
    需要 LATENT_DIM <= 3 才能直接可视化，否则使用PCA/TSNE降维。
    
    Args:
        model: 训练好的VAE模型
        data_loader: 数据加载器
        save_path: 保存路径
    """
    model.eval()
    
    all_z = []
    all_recon_error = []
    
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0]
            recon_x, mu, logvar = model(x)
            z = model.reparameterize(mu, logvar)
            
            recon_error = ((x - recon_x) ** 2).mean(dim=1)
            
            all_z.append(z.numpy())
            all_recon_error.append(recon_error.numpy())
    
    all_z = np.concatenate(all_z, axis=0)
    all_recon_error = np.concatenate(all_recon_error, axis=0)
    
    latent_dim = all_z.shape[1]
    
    if latent_dim == 2:
        # 2D直接绘制
        fig, ax = plt.subplots(figsize=(8, 8))
        scatter = ax.scatter(
            all_z[:, 0], all_z[:, 1],
            c=all_recon_error,
            cmap="RdYlBu_r",
            s=10,
            alpha=0.6
        )
        ax.set_title("Latent Space Distribution (z-space)\nColor = Reconstruction Error", fontsize=14)
        ax.set_xlabel("z[0]")
        ax.set_ylabel("z[1]")
        plt.colorbar(scatter, ax=ax, label="Reconstruction Error")
        plt.grid(True, alpha=0.3)
    
    elif latent_dim == 3:
        # 3D绘制
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            all_z[:, 0], all_z[:, 1], all_z[:, 2],
            c=all_recon_error,
            cmap="RdYlBu_r",
            s=10,
            alpha=0.6
        )
        ax.set_title("Latent Space Distribution (z-space)\nColor = Reconstruction Error", fontsize=14)
        ax.set_xlabel("z[0]")
        ax.set_ylabel("z[1]")
        ax.set_zlabel("z[2]")
        plt.colorbar(scatter, ax=ax, label="Reconstruction Error")
    
    else:
        # 使用PCA降维到2D
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(all_z)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        scatter = ax.scatter(
            z_2d[:, 0], z_2d[:, 1],
            c=all_recon_error,
            cmap="RdYlBu_r",
            s=10,
            alpha=0.6
        )
        ax.set_title(f"Latent Space Distribution (PCA 2D, {latent_dim}D→2D)\nColor = Reconstruction Error", fontsize=14)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        plt.colorbar(scatter, ax=ax, label="Reconstruction Error")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved latent space plot to {save_path}")


def plot_reconstruction_comparison(model, x_data, num_samples=5, save_path="reconstruction.png"):
    """
    绘制原始数据 vs 重构数据的对比图。
    
    Args:
        model: 训练好的VAE模型
        x_data: 输入数据 [N, window_size]
        num_samples: 展示的样本数
        save_path: 保存路径
    """
    model.eval()
    
    with torch.no_grad():
        if isinstance(x_data, np.ndarray):
            x_data = torch.tensor(x_data)
        recon_x, _, _ = model(x_data)
        recon_x = recon_x.numpy()
        x_data = x_data.numpy()
    
    # 选择重构误差最大和最小的样本
    errors = ((x_data - recon_x) ** 2).mean(axis=1)
    
    # 选：误差最小的2个、中等的1个、最大的2个
    sorted_idx = np.argsort(errors)
    selected = [
        sorted_idx[0],
        sorted_idx[1],
        sorted_idx[len(errors)//2],
        sorted_idx[-2],
        sorted_idx[-1]
    ]
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(selected):
        ax = axes[i]
        ax.plot(x_data[idx], label="Original", color="blue", linewidth=1.5)
        ax.plot(recon_x[idx], label="Reconstruction", color="red", linewidth=1.5, alpha=0.8)
        ax.fill_between(range(len(x_data[idx])), x_data[idx], recon_x[idx], 
                        alpha=0.2, color="gray", label="Error")
        ax.set_title(f"Sample {idx} (MSE={errors[idx]:.4f})")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Original vs Reconstruction Comparison", fontsize=14, y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved reconstruction comparison to {save_path}")


def plot_detection_result(timestamps, values, scores, y_true=None, threshold=None, 
                          save_path="detection_result.png"):
    """
    绘制完整的检测结果图：原始数据 + 异常分数 + 阈值线 + 异常区域标注。
    
    Args:
        timestamps: 时间戳
        values: 原始数值
        scores: 异常分数
        y_true: 真实标签（可选）
        threshold: 阈值（可选）
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    
    # 原始数据
    ax1 = axes[0]
    ax1.plot(range(len(values)), values, color="blue", linewidth=0.8, label="Value")
    
    if y_true is not None:
        # 标注真实异常区域
        anomaly_mask = y_true.astype(bool)
        if np.any(anomaly_mask):
            ax1.fill_between(range(len(values)), values.min(), values.max(),
                            where=anomaly_mask, alpha=0.2, color="red", label="True Anomaly")
    
    ax1.set_ylabel("Value")
    ax1.set_title("Original Time Series", fontsize=14)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    
    # 异常分数
    ax2 = axes[1]
    ax2.plot(range(len(scores)), scores, color="orange", linewidth=0.8, label="Anomaly Score")
    
    if threshold is not None:
        ax2.axhline(y=threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold={threshold:.4f}")
        pred_anomaly = scores >= threshold
        if np.any(pred_anomaly):
            ax2.fill_between(range(len(scores)), 0, scores.max(),
                            where=pred_anomaly, alpha=0.2, color="red", label="Predicted Anomaly")
    
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Anomaly Score")
    ax2.set_title("Anomaly Detection Result", fontsize=14)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved detection result to {save_path}")


def plot_fscore_vs_threshold(y_true, scores, save_path="fscore_threshold.png"):
    """
    绘制F-Score随阈值变化的曲线。
    """
    from sklearn.metrics import precision_recall_curve, f1_score
    
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, f1_scores[:-1], color="green", linewidth=2, label="F1-Score")
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    ax.axvline(x=best_threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"Best Threshold={best_threshold:.4f}")
    ax.scatter([best_threshold], [f1_scores[best_idx]], color="red", s=100, zorder=5)
    
    ax.set_title("F1-Score vs Threshold", fontsize=14)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1-Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved F-Score threshold plot to {save_path}")
