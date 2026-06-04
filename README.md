
# Donut KPI Anomaly Detection (Simplified Reproduction)

## Install

```bash
pip install torch pandas numpy scikit-learn matplotlib
```

## Dataset Format

CSV:

```csv
timestamp,value
2024-01-01 00:00:00,123
```

## Train

```bash
python train.py --train_csv train.csv
```

## Detect

```bash
python detect.py --test_csv test.csv
```

Output:

```
anomaly_scores.csv
```
