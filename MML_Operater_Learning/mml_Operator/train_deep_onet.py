import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
from evaluate_flops import deep_onet_flops

# DeepONet architecture
class BranchNet(nn.Module):
    def __init__(self, input_dim, width=64, depth=3, out_dim=100):
        super().__init__()
        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, u):
        return self.net(u)

class TrunkNet(nn.Module):
    def __init__(self, input_dim=1, width=64, depth=3, out_dim=100):
        super().__init__()
        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DeepONet(nn.Module):
    def __init__(self, branch, trunk):
        super().__init__()
        self.branch = branch
        self.trunk = trunk

    def forward(self, u, x):
        B, N = u.shape
        trunk_out = self.trunk(x.unsqueeze(1))      # [N, d]
        branch_out = self.branch(u)                  # [B, d]
        return torch.matmul(branch_out, trunk_out.transpose(0, 1))  # [B, N]

# Load data
def load_data():
    x = np.load("data/train/x.npy")
    u_train = np.load("data/train/u.npy")
    v_train = np.load("data/train/v.npy")
    u_test = np.load("data/test/u.npy")
    v_test = np.load("data/test/v.npy")
    return x, u_train, v_train, u_test, v_test

# Training
def train_deeponet(latent_dim, save_dir):
    x, u_train, v_train, u_test, v_test = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    u_train = torch.tensor(u_train, dtype=torch.float32).to(device)
    v_train = torch.tensor(v_train, dtype=torch.float32).to(device)
    u_test = torch.tensor(u_test, dtype=torch.float32).to(device)
    v_test = torch.tensor(v_test, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(u_train, v_train), batch_size=128, shuffle=True)

    branch = BranchNet(input_dim=u_train.shape[1], out_dim=latent_dim)
    trunk = TrunkNet(input_dim=1, out_dim=latent_dim)
    model = DeepONet(branch, trunk).to(device)

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
        pred = model(u_test, x_tensor)
        test_loss = loss_fn(pred, v_test).item()

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f"deeponet_w{latent_dim}.pth"))

    # Estimate FLOPs
    Np = u_train.shape[1]
    du = dv = latent_dim
    w = 64
    flops = deep_onet_flops(Np, du, dv, w)

    with open(os.path.join(save_dir, f"deeponet_w{latent_dim}_results.txt"), "w") as f:
        f.write(f"{flops},{test_loss:.8f}\n")

    print(f"[width={latent_dim}] Test Error = {test_loss:.2e}, FLOPs = {flops:.2e}")

# widths
if __name__ == "__main__":
    save_dir = "models/deep_onet/sweep_results"
    for latent_dim in [16, 32, 64, 128, 256]:
        train_deeponet(latent_dim, save_dir)
