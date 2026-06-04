# Donut 论文复现 — 成员B 完成文档

> **分工**：成员A 已完成核心算法（VAE、训练、基础检测）。本文档为成员B（数据对接 + 评估优化 + 理论分析）的交付说明。

---

## 1. 当前项目完整目录结构

```text
donut_project/
│
├── 核心代码（成员A + B）
│   ├── config.py              # 超参数配置
│   ├── data.py                # 数据读取 + 成员B新增：缺失值标记、Ground Truth对齐
│   ├── model.py               # Donut VAE 模型（成员A）
│   ├── loss.py                # 损失函数 + 成员B新增：M-ELBO with mask
│   ├── train.py               # 训练脚本（成员A）
│   ├── detect.py              # 基础检测脚本（成员A）
│   ├── evaluate.py            # 【新增】完整评估 pipeline
│   ├── metrics.py             # 【新增】评估指标（AUC/Best F1/Alert Delay/Segment-based）
│   ├── ablation.py            # 【新增】消融实验框架
│   ├── visualize.py           # 【新增】可视化分析
│   └── theory.md              # 【新增】论文理论整理（KDE/Time Gradient/Segment-based）
│
├── 输入数据（需数据组提供）
│   ├── train.csv              # 训练数据（时间序列）
│   ├── test.csv               # 测试数据（时间序列）
│   └── labels.csv             # 【需收集】Ground Truth 标签
│
├── 输出结果
│   ├── donut_model.pth        # 训练好的模型
│   ├── anomaly_scores.csv     # 异常分数输出
│   ├── evaluation_metrics.json# 【新增】评估指标JSON
│   ├── loss_curve.png         # 训练loss曲线
│   ├── anomaly_scores.png     # 异常分数曲线
│   ├── detection_result.png   # 【新增】检测结果综合图（原始数据+分数+阈值+异常标注）
│   ├── fscore_threshold.png   # 【新增】F1-Score vs 阈值曲线
│   ├── latent_space.png       # 【新增】Latent Space 3D分布图
│   ├── reconstruction.png     # 【新增】原始vs重构对比图
│   ├── ablation_comparison.png# 【新增】消融实验对比图
│   └── roc_pr_comparison.png  # 【新增】ROC/PR曲线对比图
│
└── 说明文档
    ├── README.md              # 项目简介
    ├── 说明.md                 # 成员A的交接说明
    └── 说明_成员B.md           # 本文档
```

---

## 2. 环境安装

```bash
pip install torch pandas numpy scikit-learn matplotlib
```

依赖包清单：

| 包名 | 用途 |
|------|------|
| torch | VAE模型训练 |
| pandas | CSV数据处理 |
| numpy | 数值计算 |
| scikit-learn | 评估指标（AUC/F1等） |
| matplotlib | 可视化绘图 |

---

## 3. 启动方式

### 3.1 基础流程（成员A原有功能）

**训练模型**：
```bash
python train.py --train_csv train.csv
```
输出：`donut_model.pth`、`loss_curve.png`

**基础检测**：
```bash
python detect.py --test_csv test.csv
```
输出：`anomaly_scores.csv`、`anomaly_scores.png`

---

### 3.2 完整评估（成员B新增）

**带Ground Truth的完整评估**：
```bash
python evaluate.py --test_csv test.csv --label_csv labels.csv --output_dir .
```

参数说明：
- `--test_csv`: 测试数据CSV路径（必需）
- `--model_path`: 模型路径（默认 `donut_model.pth`）
- `--label_csv`: Ground Truth标签文件（可选，提供后计算完整指标）
- `--output_dir`: 输出目录（默认当前目录）

**输出效果**：
- 终端打印：AUC、Best F1、Precision、Recall、Alert Delay、Segment-adjusted F1
- `evaluation_metrics.json`：结构化指标数据
- `detection_result.png`：原始数据 + 异常分数 + 阈值线 + 异常区域标注
- `fscore_threshold.png`：F1-Score随阈值变化曲线
- `latent_space.png`：Latent Space 3D分布（颜色=重构误差）
- `reconstruction.png`：原始数据 vs 重构数据对比

---

### 3.3 消融实验（成员B新增）

**运行5组消融对比**：
```bash
python ablation.py --train_csv train.csv --test_csv test.csv --label_csv labels.csv --epochs 50
```

对比的5个变体：
1. **VAE Baseline**：标准ELBO，无injection，无MCMC
2. **M-ELBO**：修改的ELBO with mask，无injection，无MCMC
3. **+Injection**：M-ELBO + Missing Data Injection
4. **+MCMC**：M-ELBO + Injection + MCMC Imputation
5. **Full Donut**：完整Donut（与+MCMC相同）

**输出效果**：
- 终端打印各变体的AUC、Best F1、Precision、Recall、Adjusted F1、Alert Delay
- `ablation_results.json`：结构化结果
- `ablation_comparison.png`：各变体loss曲线 + anomaly score曲线对比
- `roc_pr_comparison.png`：ROC曲线 + Precision-Recall曲线对比

---

## 4. 效果说明

### 4.1 评估指标含义

| 指标 | 含义 | 论文对应 |
|------|------|----------|
| AUC | ROC曲线下面积，衡量整体排序能力 | 论文Table 1 |
| Best F1 | 遍历阈值后的最优F1-Score | 论文主要指标 |
| Precision | 预测为异常中真正异常的比例 | — |
| Recall | 真正异常中被检测出的比例 | — |
| Adjusted F1 | Segment-based调整后的F1 | 论文Fig.6策略 |
| Alert Delay | 异常开始到首次检测到的平均延迟 | 论文Fig.8 |

### 4.2 Segment-based 调整策略

KPI异常通常是**连续区间**而非孤立点。本实现采用论文Fig.6策略：
- 预测调整：一个预测异常段若命中任何真实异常，则整个段视为正确
- 真实标签调整：一个真实异常段若被任何预测段覆盖，则视为被检测到
- 效果：Precision略降、Recall提高、F1更符合实际运维需求

### 4.3 MCMC Imputation（简化实现）

检测阶段对可疑位置进行迭代采样填补：
1. 通过重构误差识别可疑点（误差 > 2×中位数）
2. 对这些位置进行M步MCMC迭代refine
3. 返回L次采样的平均重构结果
4. 效果：降低检测随机性，提高稳定性

---

## 5. 数据需求清单（发给数据收集组员）

### 5.1 必需数据

#### ① 训练数据 `train.csv`

```csv
timestamp,value
2024-01-01 00:00:00,100
2024-01-01 00:01:00,101
...
```

- `timestamp`: 时间戳，格式不限（pandas可解析即可）
- `value`: KPI数值（float）
- 要求：**只包含正常数据**（无异常、无缺失最佳）

#### ② 测试数据 `test.csv`

格式同 `train.csv`，但需**包含异常区间**。

#### ③ Ground Truth 标签 `labels.csv`（二选一格式）

**格式A — 异常区间（推荐）**：
```csv
start_time,end_time
2024-01-01 00:05:00,2024-01-01 00:07:00
2024-01-01 00:12:00,2024-01-01 00:15:00
```

**格式B — 逐点标签**：
```csv
timestamp,label
2024-01-01 00:00:00,0
2024-01-01 00:05:00,1
...
```
- `label`: 0=正常，1=异常

### 5.2 数据收集要求

| 项目 | 要求 | 用途 |
|------|------|------|
| 训练数据长度 | 建议 ≥ 1000 条 | 模型充分学习正常模式 |
| 测试数据长度 | 建议 ≥ 500 条 | 包含多个异常区间 |
| 异常比例 | 5%~20% | 符合真实KPI场景 |
| 时间间隔 | 尽量均匀（如每分钟） | 滑动窗口对齐 |
| 异常类型 | 包含尖峰、跌落、趋势变化 | 验证模型泛化性 |
| 标注精度 | 区间标注精确到分钟级 | 评估对齐准确 |

### 5.3 可选补充数据

| 数据 | 说明 |
|------|------|
| 缺失值标记 | 如果数据中有已知缺失点，可用特殊值标记 |
| 多KPI数据 | 不同业务线的KPI，用于验证通用性 |
| 公开数据集 | 如 Yahoo S5、NAB 等（已有标注） |

### 5.4 推荐公开数据集（如无法自行收集）

| 数据集 | 来源 | 特点 |
|--------|------|------|
| Yahoo S5 | Yahoo Lab | 包含真实+合成异常，有标注 |
| NAB | Numenta | 真实世界数据，有标注 |
| KPI-Anomaly | AIOPS竞赛 | 大规模KPI数据 |

---

## 6. 关键配置文件

`config.py` 中的参数可根据数据调整：

```python
WINDOW_SIZE = 120      # 滑动窗口大小（建议覆盖1~2个异常周期）
LATENT_DIM = 3         # 隐变量维度（可视化建议改为2）
HIDDEN_DIM = 100       # 隐藏层维度
BATCH_SIZE = 256       # 批次大小
EPOCHS = 50            # 训练轮数
LEARNING_RATE = 1e-3   # 学习率
MISSING_INJECTION_RATE = 0.01  # 缺失注入比例
```

---

## 7. PPT 图表对应关系

| PPT内容 | 生成命令 | 输出文件 |
|---------|----------|----------|
| 训练Loss曲线 | `python train.py` | `loss_curve.png` |
| 异常分数曲线 | `python detect.py` | `anomaly_scores.png` |
| 检测结果综合图 | `python evaluate.py` | `detection_result.png` |
| F1-Score阈值图 | `python evaluate.py` | `fscore_threshold.png` |
| Latent Space分布 | `python evaluate.py` | `latent_space.png` |
| 重构对比图 | `python evaluate.py` | `reconstruction.png` |
| 消融实验对比 | `python ablation.py` | `ablation_comparison.png` |
| ROC/PR曲线 | `python ablation.py` | `roc_pr_comparison.png` |

---

## 8. 理论支撑文档

详见 `theory.md`，包含：
- **KDE解释**：VAE在latent space中的密度估计原理
- **Time Gradient效应**：异常分数渐变现象的原因和应对
- **Segment-based策略**：为什么需要段级评估
- **M-ELBO数学原理**：mask如何屏蔽异常点
- **MCMC Imputation**：检测阶段的迭代填补原理
- **PPT技术要点**：给听众的关键信息总结

---

## 9. 快速开始示例

```bash
# 1. 安装依赖
pip install torch pandas numpy scikit-learn matplotlib

# 2. 训练模型
python train.py --train_csv train.csv

# 3. 完整评估（需要labels.csv）
python evaluate.py --test_csv test.csv --label_csv labels.csv

# 4. 消融实验
python ablation.py --train_csv train.csv --test_csv test.csv --label_csv labels.csv --epochs 50
```

---

## 10. 故障排查

| 问题 | 原因 | 解决 |
|------|------|------|
| `No module named 'torch'` | torch未安装 | `pip install torch` |
| `AUC: N/A` | 测试数据无异常或全异常 | 检查labels.csv标注 |
| `latent_space.png`为空白 | 数据量太少 | 使用真实大数据集 |
| 消融实验运行慢 | MCMC采样次数多 | 减少`num_samples`参数 |
| 分数全为0 | 窗口大小>数据长度 | 减小WINDOW_SIZE或增大数据 |
