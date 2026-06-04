
"""
消融实验框架：对比 (1)纯VAE Baseline (2)M-ELBO单用 (3)+Injection (4)+MCMC (5)完整Donut
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import json

from config import *
from data import *
from model import DonutVAE
from loss import donut_loss, donut_loss_melbo
from metrics import compute_all_metrics
from visualize import plot_ablation_comparison, plot_roc_comparison


# ========== 消融变体定义 ==========

class VAE_Baseline(nn.Module):
    """纯VAE Baseline：标准ELBO，无missing injection，无MCMC"""
    def __init__(self, input_dim=120, hidden_dim=100, latent_dim=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def standard_vae_loss(recon_x, x, mu, logvar):
    """标准VAE ELBO Loss（无M-ELBO mask）"""
    recon_loss = ((x - recon_x) ** 2).mean()
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.01 * kl_loss, recon_loss, kl_loss


def mcmc_imputation(model, x_window, num_samples=1024, mcmc_steps=10):
    """
    MCMC Imputation：检测阶段固定观测点，迭代填补缺失点。
    
    简化实现：
    1. 对输入窗口，识别可能的"缺失/异常"位置（重构误差大的点）
    2. 对这些位置进行MCMC采样迭代
    3. 返回多次采样后的平均重构
    
    Args:
        model: VAE模型
        x_window: 单个窗口 [window_size]
        num_samples: 采样次数L
        mcmc_steps: MCMC迭代步数M
    
    Returns:
        平均重构结果
    """
    model.eval()
    with torch.no_grad():
        x = x_window.unsqueeze(0)  # [1, window_size]
        
        # 先做一次前向传播，获取初始重构
        recon_init, mu_init, logvar_init = model(x)
        
        # 计算每个点的重构误差，识别高误差位置（视为"缺失"候选）
        recon_error = (x - recon_init).abs().squeeze()  # [window_size]
        
        # 使用动态阈值：误差大于中位数的2倍视为可疑点
        threshold = recon_error.median() * 2
        suspicious_mask = recon_error > threshold  # bool [window_size]
        
        # MCMC迭代
        current_x = x.clone()
        recons = []
        
        for _ in range(num_samples):
            # 每次采样：从先验采样z，解码
            z = model.reparameterize(mu_init, logvar_init)
            recon = model.decode(z)
            
            # MCMC steps: 迭代 refine 可疑位置
            for _ in range(mcmc_steps):
                # 固定观测点（非可疑位置），对可疑位置用模型预测更新
                refined = current_x.clone()
                refined[:, suspicious_mask] = recon[:, suspicious_mask]
                
                # 用refined输入重新编码解码
                mu, logvar = model.encode(refined)
                z = model.reparameterize(mu, logvar)
                recon = model.decode(z)
            
            recons.append(recon)
        
        # 平均所有采样结果
        avg_recon = torch.stack(recons).mean(dim=0)
        return avg_recon.squeeze()


def train_variant(variant_name, model_class, loss_fn, use_injection, train_csv, epochs=None):
    """
    训练一个消融变体。
    
    Args:
        variant_name: 变体名称
        model_class: 模型类
        loss_fn: 损失函数
        use_injection: 是否使用missing injection
        train_csv: 训练数据路径
        epochs: 训练轮数（默认使用config中的EPOCHS）
    
    Returns:
        model_path, loss_history
    """
    if epochs is None:
        epochs = EPOCHS
    
    print(f"\n{'='*50}")
    print(f"Training variant: {variant_name}")
    print(f"{'='*50}")
    
    timestamps, values = load_series(train_csv)
    values, scaler = standardize(values)
    windows = create_windows(values, WINDOW_SIZE)
    
    if use_injection:
        windows = inject_missing(windows, rate=MISSING_INJECTION_RATE)
    
    x_tensor = torch.tensor(windows)
    dataset = TensorDataset(x_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(
        input_dim=WINDOW_SIZE,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.75)
    
    loss_history = []
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            x = batch[0].to(device)
            recon_x, mu, logvar = model(x)
            loss, recon_loss, kl_loss = loss_fn(recon_x, x, mu, logvar)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] Loss={avg_loss:.6f}")
    
    model_path = f"ablation_{variant_name.replace(' ', '_').replace('+', 'plus')}.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_
    }, model_path)
    
    print(f"  Saved to {model_path}")
    return model_path, loss_history


def detect_variant(variant_name, model_class, model_path, test_csv, use_mcmc=False):
    """
    对一个变体进行检测。
    
    Args:
        variant_name: 变体名称
        model_class: 模型类
        model_path: 模型权重路径
        test_csv: 测试数据路径
        use_mcmc: 是否使用MCMC imputation
    
    Returns:
        scores, timestamps_out
    """
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    timestamps, values = load_series(test_csv)
    mean = checkpoint["scaler_mean"][0]
    scale = checkpoint["scaler_scale"][0]
    values = (values - mean) / scale
    
    windows = create_windows(values, WINDOW_SIZE)
    x_tensor = torch.tensor(windows)
    
    model = model_class(
        input_dim=WINDOW_SIZE,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    scores = []
    
    with torch.no_grad():
        if use_mcmc:
            # 对每个窗口使用MCMC
            for i in range(len(x_tensor)):
                avg_recon = mcmc_imputation(model, x_tensor[i])
                mse = ((x_tensor[i] - avg_recon) ** 2).mean()
                scores.append(mse.item())
        else:
            recon_x, _, _ = model(x_tensor)
            mse = ((x_tensor - recon_x) ** 2).mean(dim=1)
            scores = mse.numpy().tolist()
    
    timestamps_out = timestamps[WINDOW_SIZE:]
    return np.array(scores), timestamps_out


def run_ablation(train_csv, test_csv, label_csv=None, epochs=None):
    """
    运行完整的消融实验。
    
    对比5个变体：
    1. VAE Baseline（标准ELBO，无injection，无MCMC）
    2. M-ELBO（修改的ELBO with mask，无injection，无MCMC）
    3. +Injection（M-ELBO + Missing Injection，无MCMC）
    4. +MCMC（M-ELBO + Injection + MCMC Imputation）
    5. 完整Donut（与变体4相同）
    
    Args:
        train_csv: 训练数据
        test_csv: 测试数据
        label_csv: Ground truth标签（可选，用于计算指标）
        epochs: 训练轮数
    
    Returns:
        results: dict，包含所有变体的结果和对比
    """
    results = {}
    all_scores = {}
    all_losses = {}
    
    # 变体1: VAE Baseline
    path1, loss1 = train_variant(
        "VAE Baseline", VAE_Baseline, standard_vae_loss,
        use_injection=False, train_csv=train_csv, epochs=epochs
    )
    scores1, ts1 = detect_variant("VAE Baseline", VAE_Baseline, path1, test_csv, use_mcmc=False)
    all_scores["VAE Baseline"] = scores1
    all_losses["VAE Baseline"] = loss1
    
    # 变体2: M-ELBO
    path2, loss2 = train_variant(
        "M-ELBO", DonutVAE, donut_loss_melbo,
        use_injection=False, train_csv=train_csv, epochs=epochs
    )
    scores2, ts2 = detect_variant("M-ELBO", DonutVAE, path2, test_csv, use_mcmc=False)
    all_scores["M-ELBO"] = scores2
    all_losses["M-ELBO"] = loss2
    
    # 变体3: +Injection
    path3, loss3 = train_variant(
        "+Injection", DonutVAE, donut_loss_melbo,
        use_injection=True, train_csv=train_csv, epochs=epochs
    )
    scores3, ts3 = detect_variant("+Injection", DonutVAE, path3, test_csv, use_mcmc=False)
    all_scores["+Injection"] = scores3
    all_losses["+Injection"] = loss3
    
    # 变体4: +MCMC
    path4, loss4 = train_variant(
        "+MCMC", DonutVAE, donut_loss_melbo,
        use_injection=True, train_csv=train_csv, epochs=epochs
    )
    scores4, ts4 = detect_variant("+MCMC", DonutVAE, path4, test_csv, use_mcmc=True)
    all_scores["+MCMC"] = scores4
    all_losses["+MCMC"] = loss4
    
    # 变体5: 完整Donut（与+MCMC相同实现）
    all_scores["Full Donut"] = scores4.copy()
    all_losses["Full Donut"] = loss4.copy()
    
    # 如果有ground truth，计算指标
    metrics_by_variant = {}
    if label_csv is not None:
        gt_type, gt_data = load_ground_truth(label_csv)
        if gt_type == "intervals":
            y_true = align_ground_truth(ts4, gt_data, gt_type="intervals")
        else:
            y_true = align_ground_truth(ts4, gt_data, gt_type="pointwise")
        
        for name, scores in all_scores.items():
            if name == "Full Donut":
                continue  # 与+MCMC相同
            metrics = compute_all_metrics(y_true, scores, timestamps=ts4)
            metrics_by_variant[name] = metrics
        metrics_by_variant["Full Donut"] = metrics_by_variant["+MCMC"]
    
    results = {
        "scores": all_scores,
        "losses": all_losses,
        "timestamps": ts4,
        "metrics": metrics_by_variant
    }
    
    # 保存结果
    with open("ablation_results.json", "w", encoding="utf-8") as f:
        # 只保存可序列化的部分
        serializable = {}
        for k, v in metrics_by_variant.items():
            serializable[k] = {
                "auc": v["auc"],
                "best_fscore": v["best_fscore"],
                "threshold": float(v["threshold"]),
                "segment_adjusted": v["segment_adjusted"],
                "alert_delay": {
                    "mean_delay_points": v["alert_delay"]["mean_delay_points"],
                    "num_anomaly_segments": v["alert_delay"]["num_anomaly_segments"]
                }
            }
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print("Ablation Results Summary")
    print(f"{'='*50}")
    for name, m in metrics_by_variant.items():
        print(f"\n{name}:")
        print(f"  AUC: {m['auc']:.4f}" if m['auc'] else "  AUC: N/A")
        print(f"  Best F1: {m['best_fscore']['best_f1']:.4f}")
        print(f"  Best Precision: {m['best_fscore']['best_precision']:.4f}")
        print(f"  Best Recall: {m['best_fscore']['best_recall']:.4f}")
        print(f"  Adjusted F1: {m['segment_adjusted']['adjusted']['f1']:.4f}")
        print(f"  Mean Alert Delay: {m['alert_delay']['mean_delay_points']:.2f} points")
    
    # 生成对比图
    plot_ablation_comparison(all_scores, all_losses, ts4)
    if label_csv:
        plot_roc_comparison(y_true, all_scores)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--label_csv", default=None, help="Ground truth标签文件路径")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数（默认使用config）")
    
    args = parser.parse_args()
    
    run_ablation(args.train_csv, args.test_csv, args.label_csv, args.epochs)
