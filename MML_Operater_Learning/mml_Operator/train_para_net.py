import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from evaluate_flops import para_net_flops

# PARA-Net architecture
class ParaNet(nn.Module):
    def __init__(self, u_dim, x_dim=1, width=64, depth=3):
        super().__init__()
        input_dim = u_dim + x_dim
        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, u, x_grid):
        B, N = u.shape
        x_input = x_grid.unsqueeze(0).repeat(B, 1)     # [B, N]
        u_expanded = u.unsqueeze(1).repeat(1, N, 1)    # [B, N, N]
        x_expanded = x_input.unsqueeze(2)              # [B, N, 1]
        input_cat = torch.cat([u_expanded, x_expanded], dim=-1)  # [B, N, N+1]
        return self.net(input_cat).squeeze(-1)         # [B, N]

# Load data
def load_data():
    x = np.load("data/train/x.npy")
    u_train = np.load("data/train/u.npy")
    v_train = np.load("data/train/v.npy")
    u_test = np.load("data/test/u.npy")
    v_test = np.load("data/test/v.npy")
    return x, u_train, v_train, u_test, v_test

# Training
def train_paranet(width, save_dir):
    x, u_train, v_train, u_test, v_test = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    u_train = torch.tensor(u_train, dtype=torch.float32).to(device)
    v_train = torch.tensor(v_train, dtype=torch.float32).to(device)
    u_test = torch.tensor(u_test, dtype=torch.float32).to(device)
    v_test = torch.tensor(v_test, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(u_train, v_train), batch_size=128, shuffle=True)

    model = ParaNet(u_dim=u_train.shape[1], width=width).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(300):
        model.train()
        for u_batch, v_batch in train_loader:
            pred = model(u_batch, x_tensor)
            loss = loss_fn(pred, v_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_test = model(u_test, x_tensor)
        test_loss = loss_fn(pred_test, v_test).item()

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f"paranet_w{width}.pth"))

    # Compute FLOPs
    Np = u_train.shape[1]
    du = u_train.shape[1]
    dy = 1  # x is 1D
    do = 1  # output is scalar per point
    flops = para_net_flops(Np, du, dy, do, width)

    with open(os.path.join(save_dir, f"paranet_w{width}_results.txt"), "w") as f:
        f.write(f"{flops},{test_loss:.8f}\n")

    print(f"[PARA-Net width={width}] Test Error = {test_loss:.2e}, FLOPs = {flops:.2e}")

# Sweep width
if __name__ == "__main__":
    save_dir = "models/para_net/sweep_results"
    for width in [16, 32, 64, 128, 256]:
        train_paranet(width, save_dir)
