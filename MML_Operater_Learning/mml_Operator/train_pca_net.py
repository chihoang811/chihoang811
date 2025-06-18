import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
import os
from evaluate_flops import pca_net_flops


# PCA_Net architecture
class PCANet(nn.Module):
    def __init__(self, input_dim, output_dim, width=64, depth=3):
        super().__init__()
        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers += [nn.Linear(width, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Load data
def load_data():
    x = np.load("data/train/x.npy")
    u_train = np.load("data/train/u.npy")
    v_train = np.load("data/train/v.npy")
    u_test = np.load("data/test/u.npy")
    v_test = np.load("data/test/v.npy")
    return x, u_train, v_train, u_test, v_test

# Training
def train_pca_net(pca_dim, width, save_dir):
    x, u_train, v_train, u_test, v_test = load_data()

    # PCA
    pca_u = PCA(n_components=pca_dim).fit(u_train)
    pca_v = PCA(n_components=pca_dim).fit(v_train)

    u_train_pca = pca_u.transform(u_train)
    v_train_pca = pca_v.transform(v_train)
    u_test_pca = pca_u.transform(u_test)
    v_test_pca = pca_v.transform(v_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = torch.tensor(u_train_pca, dtype=torch.float32).to(device)
    Y_train = torch.tensor(v_train_pca, dtype=torch.float32).to(device)
    X_test = torch.tensor(u_test_pca, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=128, shuffle=True)

    model = PCANet(input_dim=pca_dim, output_dim=pca_dim, width=width).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(300):
        model.train()
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_pca = model(X_test).cpu().numpy()
        v_pred = pca_v.inverse_transform(pred_pca)
        test_mse = np.mean((v_pred - v_test) ** 2)

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f"pca_net_d{pca_dim}.pth"))

    # FLOPs estimation
    Np = u_train.shape[1]
    du = dv = pca_dim

    flops = pca_net_flops(Np, du, dv, width)

    torch.save(model.state_dict(), os.path.join(save_dir, f"pca_net_d{pca_dim}_w{width}.pth"))
    with open(os.path.join(save_dir, f"pca_net_d{pca_dim}_results.txt"), "w") as f:
        f.write(f"{flops},{test_mse:.8f}\n")

    print(f"[PCA dim={pca_dim}] Test Error = {test_mse:.2e}, FLOPs = {flops:.2e}")


if __name__ == "__main__":
    save_dir = "models/pca_net/sweep_results"
    for pca_dim in [2,4]:
        for width in [16, 32, 64, 128, 256]:
            train_pca_net(pca_dim, width, save_dir)
