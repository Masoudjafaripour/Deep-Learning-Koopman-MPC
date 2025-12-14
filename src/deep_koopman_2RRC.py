# deep_koopman_clean.py
# A clean, minimal Deep-Koopman example:
# - Nonlinear controlled dynamics (2D + 1D control)
# - Collect (x_k, u_k, x_{k+1}) data
# - Learn lifting (encoder) + linear Koopman dynamics in lifted space + decoder
# - Compare rollouts (true vs Koopman)

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ----------------------------
# 1) Reproducibility + device
# ----------------------------
SEED = 2141444
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# 2) Nonlinear dynamics (discrete)
# ----------------------------
DT = 0.01

@torch.no_grad()
def f_step(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    x: (B, 2)
    u: (B, 1)
    returns x_next: (B, 2)

    Simple nonlinear controlled system:
      x1+ = x1 + dt(-x1^3 + x2 + u)
      x2+ = x2 + dt(-0.5 x2 + x1)
    """
    x1, x2 = x[:, 0], x[:, 1]
    u0 = u[:, 0]
    x1_next = x1 + DT * (-x1**3 + x2 + u0)
    x2_next = x2 + DT * (-0.5 * x2 + x1)
    return torch.stack([x1_next, x2_next], dim=1)


# ----------------------------
# 3) Data collection (like X,Y,U in MATLAB)
# ----------------------------
@torch.no_grad()
def collect_data(n_traj=1000, n_steps=50, x_range=1.0, u_range=1.0):
    """
    Generates a dataset by rolling out random controls:
      X: (N, 2), U: (N, 1), Y: (N, 2)
    where Y = f_step(X, U)
    """
    x = (torch.rand(n_traj, 2, device=device) * 2 - 1) * x_range

    X_list, U_list, Y_list = [], [], []
    for _ in range(n_steps):
        u = (torch.rand(n_traj, 1, device=device) * 2 - 1) * u_range
        y = f_step(x, u)

        X_list.append(x)
        U_list.append(u)
        Y_list.append(y)

        x = y

    X = torch.cat(X_list, dim=0)
    U = torch.cat(U_list, dim=0)
    Y = torch.cat(Y_list, dim=0)
    return X, U, Y


# ----------------------------
# 4) Deep Koopman model
# ----------------------------
class DeepKoopman(nn.Module):
    """
    Learn:
      z = encoder(x)
      z_next = A z + B u          (linear Koopman dynamics in lifted space)
      x_hat_next = decoder(z_next)

    Also enforce consistency:
      z_next should match encoder(y)
    """
    def __init__(self, state_dim=2, control_dim=1, lift_dim=16, hidden=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, lift_dim),
        )

        # Linear operators (no bias to keep it strictly linear)
        self.A = nn.Linear(lift_dim, lift_dim, bias=False)
        self.B = nn.Linear(control_dim, lift_dim, bias=False)

        self.decoder = nn.Linear(lift_dim, state_dim)

    def lift(self, x):
        return self.encoder(x)

    def koopman_step(self, z, u):
        return self.A(z) + self.B(u)

    def forward(self, x, u):
        z = self.lift(x)
        z_next = self.koopman_step(z, u)
        x_next_hat = self.decoder(z_next)
        return x_next_hat, z, z_next


# ----------------------------
# 5) Training loop
# ----------------------------
def train_deep_koopman(
    X, U, Y,
    lift_dim=16,
    hidden=64,
    lr=1e-3,
    epochs=2000,
    batch_size=4096,
    koopman_weight=0.1,
):
    model = DeepKoopman(lift_dim=lift_dim, hidden=hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    N = X.shape[0]
    indices = torch.arange(N, device=device)

    for ep in range(1, epochs + 1):
        # minibatch
        batch_idx = indices[torch.randint(0, N, (batch_size,), device=device)]
        xb, ub, yb = X[batch_idx], U[batch_idx], Y[batch_idx]

        opt.zero_grad()

        x_next_hat, z, z_next = model(xb, ub)
        z_next_true = model.lift(yb)

        loss_pred = mse(x_next_hat, yb)
        loss_koop = mse(z_next, z_next_true)
        loss = loss_pred + koopman_weight * loss_koop

        loss.backward()
        opt.step()

        if ep % 200 == 0 or ep == 1:
            print(
                f"epoch {ep:4d} | "
                f"loss {loss.item():.3e} | "
                f"pred {loss_pred.item():.3e} | "
                f"koop {loss_koop.item():.3e}"
            )

    return model


# ----------------------------
# 6) Rollout comparison
# ----------------------------
@torch.no_grad()
def rollout_compare(model, x0, T=200, u_const=0.5):
    """
    Compare true rollout vs Koopman rollout under constant control.
    Returns:
      xs_true: (T+1, 2)
      xs_koop: (T+1, 2)
    """
    x_true = x0.clone().to(device)
    z = model.lift(x_true)

    xs_true = [x_true]
    xs_koop = [x_true]

    u = torch.full((1, 1), float(u_const), device=device)

    for _ in range(T):
        # true dynamics
        x_true = f_step(x_true, u)
        xs_true.append(x_true)

        # koopman dynamics
        z = model.koopman_step(z, u)
        x_hat = model.decoder(z)
        xs_koop.append(x_hat)

    xs_true = torch.cat(xs_true, dim=0).cpu().numpy()
    xs_koop = torch.cat(xs_koop, dim=0).cpu().numpy()
    return xs_true, xs_koop


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

import matplotlib.pyplot as plt

def plot_rollout(xs_true, xs_koop, dt):
    T = xs_true.shape[0]
    t = np.arange(T) * dt

    plt.figure(figsize=(12, 4))

    # x1
    plt.subplot(1, 2, 1)
    plt.plot(t, xs_true[:, 0], linewidth=2, label="True")
    plt.plot(t, xs_koop[:, 0], "--", linewidth=2, label="Koopman")
    plt.xlabel("Time [s]")
    plt.ylabel("x1")
    plt.title("State x1")
    plt.legend()
    plt.grid(True)

    # x2
    plt.subplot(1, 2, 2)
    plt.plot(t, xs_true[:, 1], linewidth=2, label="True")
    plt.plot(t, xs_koop[:, 1], "--", linewidth=2, label="Koopman")
    plt.xlabel("Time [s]")
    plt.ylabel("x2")
    plt.title("State x2")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_phase(xs_true, xs_koop):
    plt.figure(figsize=(5, 5))
    plt.plot(xs_true[:, 0], xs_true[:, 1], linewidth=2, label="True")
    plt.plot(xs_koop[:, 0], xs_koop[:, 1], "--", linewidth=2, label="Koopman")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Phase Space")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

# ----------------------------
# 7) Main
# ----------------------------
def main():
    # Data (similar spirit to MATLAB: many trajs, multiple sim steps)
    X, U, Y = collect_data(n_traj=1000, n_steps=50, x_range=1.0, u_range=1.0)
    print(f"dataset: X {tuple(X.shape)} | U {tuple(U.shape)} | Y {tuple(Y.shape)}")

    # Train
    model = train_deep_koopman(
        X, U, Y,
        lift_dim=16,
        hidden=64,
        lr=1e-3,
        epochs=2000,
        batch_size=4096,
        koopman_weight=0.1,
    )

    # Rollout test
    x0 = torch.tensor([[0.5, 0.1]], dtype=torch.float32, device=device)
    xs_true, xs_koop = rollout_compare(model, x0, T=200, u_const=0.5)

    # Report errors per state dim
    rmse_x1 = rmse(xs_true[:, 0], xs_koop[:, 0])
    rmse_x2 = rmse(xs_true[:, 1], xs_koop[:, 1])
    print(f"RMSE x1: {rmse_x1:.4e}")
    print(f"RMSE x2: {rmse_x2:.4e}")
    print(f"RMSE avg: {(rmse_x1 + rmse_x2)/2:.4e}")

    # Optional: quick text preview
    print("\nFirst 5 timesteps (true vs koopman):")
    for i in range(5):
        t = i * DT
        print(
            f"t={t:5.2f}  "
            f"true=({xs_true[i,0]: .4f},{xs_true[i,1]: .4f})  "
            f"koop=({xs_koop[i,0]: .4f},{xs_koop[i,1]: .4f})"
        )

    # Plots
    plot_rollout(xs_true, xs_koop, DT)
    plot_phase(xs_true, xs_koop)


if __name__ == "__main__":
    main()

