
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_series(csv_path):
    df = pd.read_csv(csv_path)
    values = df["value"].values.astype(np.float32)
    timestamps = df["timestamp"].values
    return timestamps, values

def standardize(train_values, test_values=None):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(
        train_values.reshape(-1, 1)
    ).flatten()

    if test_values is None:
        return train_scaled, scaler

    test_scaled = scaler.transform(
        test_values.reshape(-1, 1)
    ).flatten()

    return train_scaled, test_scaled, scaler

def create_windows(values, window_size):
    windows = []

    for i in range(window_size, len(values)):
        windows.append(values[i-window_size:i])

    return np.array(windows, dtype=np.float32)

def inject_missing(x, rate=0.01):
    x = x.copy()

    mask = np.random.rand(*x.shape) < rate
    x[mask] = 0.0

    return x


# ========== 成员B新增：数据接口增强 ==========

def load_series_with_missing(csv_path, missing_marker=None):
    """
    读取时间序列，支持缺失值标记。
    
    Args:
        csv_path: CSV文件路径
        missing_marker: 缺失值标记，如 None, np.nan, -999 等
                       为 None 时只返回正常数据
    
    Returns:
        timestamps, values, missing_mask
        missing_mask: bool数组，True表示该位置是缺失值
    """
    df = pd.read_csv(csv_path)
    values = df["value"].values.astype(np.float32)
    timestamps = df["timestamp"].values
    
    missing_mask = np.zeros(len(values), dtype=bool)
    
    if missing_marker is not None:
        if pd.isna(missing_marker):
            missing_mask = pd.isna(df["value"])
        else:
            missing_mask = (df["value"] == missing_marker)
    
    return timestamps, values, missing_mask


def load_ground_truth(label_csv_path):
    """
    读取Ground Truth标签文件。
    
    支持的格式：
    1. CSV格式（每行一个异常区间）：
       start_time,end_time
       2024-01-01 00:05:00,2024-01-01 00:07:00
    
    2. CSV格式（逐点标签）：
       timestamp,label
       2024-01-01 00:00:00,0
       2024-01-01 00:05:00,1
    
    Returns:
        如果是区间格式：list of (start_time_str, end_time_str) tuples
        如果是逐点格式：(timestamps, labels) tuple
    """
    df = pd.read_csv(label_csv_path)
    columns = [c.lower().strip() for c in df.columns]
    
    if "start_time" in columns or "start" in columns:
        # 区间格式
        start_col = "start_time" if "start_time" in columns else "start"
        end_col = "end_time" if "end_time" in columns else "end"
        intervals = []
        for _, row in df.iterrows():
            intervals.append((str(row[start_col]), str(row[end_col])))
        return "intervals", intervals
    
    elif "label" in columns:
        # 逐点标签格式
        timestamps = df["timestamp"].values
        labels = df["label"].values.astype(int)
        return "pointwise", (timestamps, labels)
    
    else:
        raise ValueError(
            f"Ground truth文件格式无法识别。列名: {list(df.columns)}。"
            f"需要包含 'start_time,end_time' 或 'timestamp,label'"
        )


def align_ground_truth(timestamps, labels_or_intervals, gt_type="intervals"):
    """
    将Ground Truth对齐到模型输出的时间戳序列。
    
    Args:
        timestamps: 模型输出的时间戳数组（与anomaly_scores对应）
        labels_or_intervals: 
            gt_type="intervals" 时：list of (start, end) tuples
            gt_type="pointwise" 时：(gt_timestamps, gt_labels) tuple
        gt_type: "intervals" 或 "pointwise"
    
    Returns:
        binary_labels: 与timestamps等长的0/1数组，1表示异常
    """
    binary_labels = np.zeros(len(timestamps), dtype=int)
    
    # 将时间戳转为pandas datetime用于比较
    ts_pd = pd.to_datetime(timestamps)
    
    if gt_type == "intervals":
        intervals = labels_or_intervals
        for start_str, end_str in intervals:
            start = pd.to_datetime(start_str)
            end = pd.to_datetime(end_str)
            mask = (ts_pd >= start) & (ts_pd <= end)
            binary_labels[mask] = 1
    
    elif gt_type == "pointwise":
        gt_timestamps, gt_labels = labels_or_intervals
        gt_pd = pd.to_datetime(gt_timestamps)
        
        # 对于每个输出时间戳，找最近的ground truth标签
        for i, ts in enumerate(ts_pd):
            # 找时间差最小的
            diffs = np.abs(gt_pd - ts)
            nearest_idx = diffs.argmin()
            # 如果时间差在1分钟以内，采用该标签
            if diffs.iloc[nearest_idx] <= pd.Timedelta(minutes=1):
                binary_labels[i] = gt_labels[nearest_idx]
    
    return binary_labels
