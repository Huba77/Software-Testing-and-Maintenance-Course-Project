
"""
评估脚本：完整的评估 pipeline

功能：
1. 加载模型和测试数据
2. 计算异常分数
3. 加载Ground Truth并对齐
4. 计算全部评估指标（AUC, Best F1, Precision, Recall, Alert Delay）
5. 生成评估报告和可视化图表
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch

from config import *
from data import *
from model import DonutVAE
from metrics import compute_all_metrics, compute_metrics_with_adjustment
from visualize import (
    plot_detection_result,
    plot_fscore_vs_threshold,
    plot_latent_space,
    plot_reconstruction_comparison
)


def evaluate(test_csv, model_path, label_csv=None, output_dir="."):
    """
    执行完整评估。
    
    Args:
        test_csv: 测试数据CSV路径
        model_path: 模型权重路径
        label_csv: Ground Truth标签文件路径（可选）
        output_dir: 输出目录
    
    Returns:
        results: 评估结果字典
    """
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    print(f"Loading test data from {test_csv}...")
    timestamps, values = load_series(test_csv)
    
    # 标准化
    mean = checkpoint["scaler_mean"][0]
    scale = checkpoint["scaler_scale"][0]
    values_scaled = (values - mean) / scale
    
    # 创建窗口
    windows = create_windows(values_scaled, WINDOW_SIZE)
    x_tensor = torch.tensor(windows)
    
    # 加载模型
    model = DonutVAE(
        input_dim=WINDOW_SIZE,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # 计算异常分数
    print("Computing anomaly scores...")
    scores = []
    with torch.no_grad():
        recon_x, _, _ = model(x_tensor)
        mse = ((x_tensor - recon_x) ** 2).mean(dim=1)
        scores = mse.numpy()
    
    timestamps_out = timestamps[WINDOW_SIZE:]
    values_out = values[WINDOW_SIZE:]
    
    # 保存异常分数
    result_df = pd.DataFrame({
        "timestamp": timestamps_out,
        "value": values_out,
        "anomaly_score": scores
    })
    result_df.to_csv(f"{output_dir}/anomaly_scores.csv", index=False)
    print(f"Saved anomaly scores to {output_dir}/anomaly_scores.csv")
    
    # 基础统计
    stats = {
        "num_points": len(scores),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "max_score": float(np.max(scores)),
        "min_score": float(np.min(scores)),
        "percentile_95": float(np.percentile(scores, 95)),
        "percentile_99": float(np.percentile(scores, 99))
    }
    
    print(f"\nScore Statistics:")
    print(f"  Mean: {stats['mean_score']:.4f}")
    print(f"  Std:  {stats['std_score']:.4f}")
    print(f"  Max:  {stats['max_score']:.4f}")
    print(f"  95th percentile: {stats['percentile_95']:.4f}")
    
    results = {
        "statistics": stats,
        "metrics": None
    }
    
    # 如果有Ground Truth，计算完整指标
    y_true = None
    if label_csv is not None:
        print(f"\nLoading ground truth from {label_csv}...")
        gt_type, gt_data = load_ground_truth(label_csv)
        
        if gt_type == "intervals":
            y_true = align_ground_truth(timestamps_out, gt_data, gt_type="intervals")
        else:
            y_true = align_ground_truth(timestamps_out, gt_data, gt_type="pointwise")
        
        num_anomalies = int(np.sum(y_true))
        print(f"  Total points: {len(y_true)}")
        print(f"  Anomaly points: {num_anomalies} ({100*num_anomalies/len(y_true):.2f}%)")
        
        # 计算所有指标
        print("\nComputing metrics...")
        metrics = compute_all_metrics(y_true, scores, timestamps=timestamps_out)
        results["metrics"] = metrics
        
        # 打印结果
        print(f"\n{'='*50}")
        print("Evaluation Results")
        print(f"{'='*50}")
        print(f"AUC: {metrics['auc']:.4f}" if metrics['auc'] else "AUC: N/A")
        print(f"\nBest F-Score (threshold search):")
        print(f"  F1:       {metrics['best_fscore']['best_f1']:.4f}")
        print(f"  Precision:{metrics['best_fscore']['best_precision']:.4f}")
        print(f"  Recall:   {metrics['best_fscore']['best_recall']:.4f}")
        print(f"  Threshold:{metrics['best_fscore']['best_threshold']:.4f}")
        print(f"\nSegment-adjusted Metrics:")
        print(f"  Raw F1:       {metrics['segment_adjusted']['raw']['f1']:.4f}")
        print(f"  Adjusted F1:  {metrics['segment_adjusted']['adjusted']['f1']:.4f}")
        print(f"\nAlert Delay:")
        print(f"  Mean delay: {metrics['alert_delay']['mean_delay_points']:.2f} points")
        print(f"  #Segments:  {metrics['alert_delay']['num_anomaly_segments']}")
        
        # 保存详细指标
        serializable_metrics = {
            "auc": metrics["auc"],
            "best_fscore": metrics["best_fscore"],
            "threshold": float(metrics["threshold"]),
            "segment_adjusted": metrics["segment_adjusted"],
            "alert_delay": {
                "mean_delay_points": metrics["alert_delay"]["mean_delay_points"],
                "num_anomaly_segments": metrics["alert_delay"]["num_anomaly_segments"]
            }
        }
        
        with open(f"{output_dir}/evaluation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
        print(f"\nSaved metrics to {output_dir}/evaluation_metrics.json")
        
        # 生成可视化
        print("\nGenerating visualizations...")
        
        # 1. 检测结果图
        plot_detection_result(
            timestamps_out, values_out, scores,
            y_true=y_true,
            threshold=metrics["best_fscore"]["best_threshold"],
            save_path=f"{output_dir}/detection_result.png"
        )
        
        # 2. F-Score vs Threshold
        plot_fscore_vs_threshold(
            y_true, scores,
            save_path=f"{output_dir}/fscore_threshold.png"
        )
        
        # 3. Latent space visualization
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(x_tensor)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        plot_latent_space(
            model, loader,
            save_path=f"{output_dir}/latent_space.png"
        )
        
        # 4. Reconstruction comparison
        plot_reconstruction_comparison(
            model, x_tensor.numpy()[:100],
            save_path=f"{output_dir}/reconstruction.png"
        )
    
    else:
        # 无Ground Truth时，只生成基础图
        print("\nNo ground truth provided. Generating basic plots...")
        plot_detection_result(
            timestamps_out, values_out, scores,
            threshold=stats["percentile_95"],
            save_path=f"{output_dir}/detection_result.png"
        )
    
    # 保存异常分数曲线（原有功能兼容）
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    plt.plot(scores, color="orange", linewidth=0.8)
    plt.title("Anomaly Scores")
    plt.xlabel("Time Step")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/anomaly_scores.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved anomaly score plot to {output_dir}/anomaly_scores.png")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Donut Model Evaluation")
    parser.add_argument("--test_csv", required=True, help="测试数据CSV路径")
    parser.add_argument("--model_path", default=MODEL_PATH, help="模型权重路径")
    parser.add_argument("--label_csv", default=None, help="Ground Truth标签文件路径")
    parser.add_argument("--output_dir", default=".", help="输出目录")
    
    args = parser.parse_args()
    
    evaluate(args.test_csv, args.model_path, args.label_csv, args.output_dir)
