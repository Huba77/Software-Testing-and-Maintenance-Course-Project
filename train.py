import matplotlib.pyplot as plt
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from config import *
from data import *
from model import DonutVAE
from loss import donut_loss

parser = argparse.ArgumentParser()
parser.add_argument("--train_csv", required=True)

args = parser.parse_args()

print("Loading dataset...")

timestamps, values = load_series(args.train_csv)

values, scaler = standardize(values)

windows = create_windows(values, WINDOW_SIZE)

windows = inject_missing(
    windows,
    rate=MISSING_INJECTION_RATE
)

x_tensor = torch.tensor(windows)

dataset = TensorDataset(x_tensor)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DonutVAE(
    input_dim=WINDOW_SIZE,
    hidden_dim=HIDDEN_DIM,
    latent_dim=LATENT_DIM
).to(device)

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

scheduler = StepLR(optimizer, step_size=10, gamma=0.75)

print("Start training...")

loss_history = []

model.train()

for epoch in range(EPOCHS):

    total_loss = 0

    for batch in loader:

        x = batch[0].to(device)

        recon_x, mu, logvar = model(x)

        loss, recon_loss, kl_loss = donut_loss(
            recon_x,
            x,
            mu,
            logvar
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    avg_loss = total_loss / len(loader)

    loss_history.append(avg_loss)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss={avg_loss:.6f}"
    )

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_
    },
    MODEL_PATH
)

print(f"Model saved to {MODEL_PATH}")

plt.figure(figsize=(8,5))

plt.plot(loss_history)

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid(True)

plt.savefig("loss_curve.png")

print("Saved loss curve to loss_curve.png")
