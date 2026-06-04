import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd
import torch

from config import *
from data import *
from model import DonutVAE

parser = argparse.ArgumentParser()
parser.add_argument("--test_csv", required=True)

args = parser.parse_args()

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

timestamps, values = load_series(args.test_csv)

mean = checkpoint["scaler_mean"][0]
scale = checkpoint["scaler_scale"][0]

values = (values - mean) / scale

windows = create_windows(values, WINDOW_SIZE)

x_tensor = torch.tensor(windows)

model = DonutVAE(
    input_dim=WINDOW_SIZE,
    hidden_dim=HIDDEN_DIM,
    latent_dim=LATENT_DIM
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

scores = []

with torch.no_grad():

    recon_x, _, _ = model(x_tensor)

    mse = ((x_tensor - recon_x) ** 2).mean(dim=1)

    scores = mse.numpy()

result_df = pd.DataFrame({
    "timestamp": timestamps[WINDOW_SIZE:],
    "anomaly_score": scores
})

result_df.to_csv("anomaly_scores.csv", index=False)

print(result_df.head())

print("\nSaved to anomaly_scores.csv")

plt.figure(figsize=(12,5))

plt.plot(scores)

plt.title("Anomaly Scores")

plt.xlabel("Time")

plt.ylabel("Score")

plt.grid(True)

plt.savefig("anomaly_scores.png")

print("Saved anomaly score plot to anomaly_scores.png")