
"""
评估指标模块：Best F-Score, AUC, Alert Delay, Segment-based 调整策略
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, f1_score,
    precision_score, recall_score,
    confusion_matrix
)


def compute_auc(y_true, scores):
    """
    计算 ROC AUC。
    
    Args:
        y_true: 真实标签 (0=正常, 1=异常)
        scores: 异常分数（越高越异常）
    
    Returns:
        auc: AUC值
    """
    if len(np.unique(y_true)) < 2:
        return None  # 无法计算AUC
    return roc_auc_score(y_true, scores)


def compute_best_fscore(y_true, scores, pos_label=1):
    """
    通过遍历阈值找到最佳 F1-Score。
    
    Args:
        y_true: 真实标签
        scores: 异常分数
        pos_label: 正类标签（异常=1）
    
    Returns:
        dict: {
            'best_f1': 最佳F1值,
            'best_threshold': 最佳阈值,
            'best_precision': 对应precision,
            'best_recall': 对应recall
        }
    """
    # 使用precision_recall_curve获取所有阈值下的precision和recall
    precision, recall, thresholds = precision_recall_curve(y_true, scores, pos_label=pos_label)
    
    # 计算每个阈值的F1
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    
    # 注意：thresholds比precision/recall少一个元素
    if best_idx < len(thresholds):
        best_threshold = thresholds[best_idx]
    else:
        best_threshold = thresholds[-1] if len(thresholds) > 0 else 0.0
    
    return {
        'best_f1': float(best_f1),
        'best_threshold': float(best_threshold),
        'best_precision': float(best_precision),
        'best_recall': float(best_recall)
    }


def segment_based_adjustment(y_pred, window_size=10):
    """
    Segment-based 调整策略（论文Fig.6）。
    
    规则：如果一个异常段（连续预测为异常的点）中，
    只要有一个点真正命中了ground truth异常区间，
    则整个异常段的所有点都视为正确检测。
    
    这是异常检测中常用的评估策略，降低了对精确时间对齐的要求。
    
    Args:
        y_pred: 预测标签 (0/1)
        window_size: 用于判断"段"的窗口大小
    
    Returns:
        adjusted_pred: 调整后的预测标签
    """
    adjusted = y_pred.copy()
    n = len(adjusted)
    
    # 找到所有异常段
    in_segment = False
    segment_start = 0
    segments = []
    
    for i in range(n):
        if adjusted[i] == 1 and not in_segment:
            in_segment = True
            segment_start = i
        elif adjusted[i] == 0 and in_segment:
            in_segment = False
            segments.append((segment_start, i))
    
    if in_segment:
        segments.append((segment_start, n))
    
    # 对每个异常段，如果段内有任何True Positive，则整个段都算对
    # 注意：这里需要结合y_true来判断，所以实际应该在compute_metrics_with_adjustment中处理
    return segments


def compute_metrics_with_adjustment(y_true, y_pred):
    """
    使用segment-based adjustment计算指标。
    
    调整策略：
    1. 找到所有预测为异常的连续段
    2. 如果某段中至少有一个点真正是异常，则整个段视为正确检测
    3. 重新计算precision, recall, f1
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签（已用某阈值二值化）
    
    Returns:
        dict: 调整前后的指标对比
    """
    # 原始指标
    raw_precision = precision_score(y_true, y_pred, zero_division=0)
    raw_recall = recall_score(y_true, y_pred, zero_division=0)
    raw_f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Segment-based adjustment
    adjusted_pred = y_pred.copy()
    n = len(adjusted_pred)
    
    # 找到所有预测异常段
    segments = []
    in_seg = False
    seg_start = 0
    for i in range(n):
        if y_pred[i] == 1 and not in_seg:
            in_seg = True
            seg_start = i
        elif y_pred[i] == 0 and in_seg:
            in_seg = False
            segments.append((seg_start, i))
    if in_seg:
        segments.append((seg_start, n))
    
    # 对每个段：如果段内有任何True Positive，则整个段设为1（已满足），否则设为0
    for start, end in segments:
        segment_true = y_true[start:end]
        if np.sum(segment_true) == 0:
            # 整个段都是False Positive，全部清零
            adjusted_pred[start:end] = 0
    
    # 找到所有真实异常段
    true_segments = []
    in_seg = False
    seg_start = 0
    for i in range(n):
        if y_true[i] == 1 and not in_seg:
            in_seg = True
            seg_start = i
        elif y_true[i] == 0 and in_seg:
            in_seg = False
            true_segments.append((seg_start, i))
    if in_seg:
        true_segments.append((seg_start, n))
    
    # 如果某个真实异常段被任何预测异常段覆盖，则该真实段全部视为被检测到
    for t_start, t_end in true_segments:
        detected = False
        for p_start, p_end in segments:
            # 有重叠即视为检测到
            if not (p_end <= t_start or p_start >= t_end):
                detected = True
                break
        if detected:
            adjusted_pred[t_start:t_end] = 1
    
    adj_precision = precision_score(y_true, adjusted_pred, zero_division=0)
    adj_recall = recall_score(y_true, adjusted_pred, zero_division=0)
    adj_f1 = f1_score(y_true, adjusted_pred, zero_division=0)
    
    return {
        'raw': {
            'precision': float(raw_precision),
            'recall': float(raw_recall),
            'f1': float(raw_f1)
        },
        'adjusted': {
            'precision': float(adj_precision),
            'recall': float(adj_recall),
            'f1': float(adj_f1)
        }
    }


def compute_alert_delay(y_true, y_pred, timestamps=None):
    """
    计算 Alert Delay（警报延迟）。
    
    对每个真实异常段，计算从异常开始到首次被检测到的时间差。
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        timestamps: 时间戳数组（可选，用于计算实际时间延迟）
                   为None时返回点数延迟
    
    Returns:
        dict: {
            'mean_delay_points': 平均延迟（点数）,
            'mean_delay_seconds': 平均延迟（秒，如果提供了timestamps）,
            'delays': 每个异常段的延迟列表
        }
    """
    # 找到所有真实异常段
    n = len(y_true)
    true_segments = []
    in_seg = False
    seg_start = 0
    for i in range(n):
        if y_true[i] == 1 and not in_seg:
            in_seg = True
            seg_start = i
        elif y_true[i] == 0 and in_seg:
            in_seg = False
            true_segments.append((seg_start, i))
    if in_seg:
        true_segments.append((seg_start, n))
    
    delays_points = []
    delays_seconds = []
    
    for start, end in true_segments:
        # 在该段内找第一个被预测为异常的点
        detected_indices = np.where(y_pred[start:end] == 1)[0]
        if len(detected_indices) > 0:
            delay = detected_indices[0]  # 相对于段开始的延迟
        else:
            delay = end - start  # 未被检测到，延迟为整个段长度
        
        delays_points.append(int(delay))
        
        if timestamps is not None:
            # 计算实际时间差
            ts = pd.to_datetime(timestamps)
            delay_sec = (ts[start + delay] - ts[start]).total_seconds()
            delays_seconds.append(float(delay_sec))
    
    result = {
        'mean_delay_points': float(np.mean(delays_points)) if delays_points else 0.0,
        'delays_points': delays_points,
        'num_anomaly_segments': len(true_segments)
    }
    
    if timestamps is not None:
        result['mean_delay_seconds'] = float(np.mean(delays_seconds)) if delays_seconds else 0.0
        result['delays_seconds'] = delays_seconds
    
    return result


def compute_all_metrics(y_true, scores, timestamps=None):
    """
    计算所有评估指标。
    
    Args:
        y_true: 真实标签
        scores: 异常分数
        timestamps: 时间戳（可选，用于计算Alert Delay）
    
    Returns:
        dict: 所有指标
    """
    # 1. AUC
    auc = compute_auc(y_true, scores)
    
    # 2. Best F-Score（通过遍历阈值）
    best_f = compute_best_fscore(y_true, scores)
    
    # 3. 使用最佳阈值做二值化预测
    threshold = best_f['best_threshold']
    y_pred = (scores >= threshold).astype(int)
    
    # 4. Segment-based adjustment 指标
    adjusted = compute_metrics_with_adjustment(y_true, y_pred)
    
    # 5. Alert Delay
    alert_delay = compute_alert_delay(y_true, y_pred, timestamps)
    
    return {
        'auc': auc,
        'best_fscore': best_f,
        'threshold': threshold,
        'segment_adjusted': adjusted,
        'alert_delay': alert_delay
    }
