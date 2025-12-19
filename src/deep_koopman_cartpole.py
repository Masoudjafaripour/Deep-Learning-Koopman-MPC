# deep_koopman_cartpole.py
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.set_default_dtype(torch.float32)
device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# 1) Nonlinear controlled dynamics: CartPole
# state x = [pos, vel, theta, theta_dot]
# control u = [force]
# -----------------------------
def cartpole_step(x, u, dt=0.02):
    """
    x: (B,4), u: (B,1)
    Returns x_next: (B,4)
    """
    # Parameters (standard-ish)
    g = 9.8
    mc = 1.0       # cart mass
    mp = 0.1       # pole mass
    l = 0.5        # half pole length
    total_mass = mc + mp
    polemass_length = mp * l

    pos, vel, th, thd = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    force = u[:, 0]

    # keep theta within reasonable range (optional)
    # th = (th + math.pi) % (2*math.pi) - math.pi

    costh = torch.cos(th)
    sinth = torch.sin(th)

    # dynamics from classic cartpole equations (continuous-time)
    temp = (force + polemass_length * thd**2 * sinth) / total_mass
    thdd = (g * sinth - costh * temp) / (l * (4.0/3.0 - mp * costh**2 / total_mass))
    xdd = temp - polemass_length * thdd * costh / total_mass

    # Euler integration
    pos_next = pos + dt * vel
    vel_next = vel + dt * xdd
    th_next  = th  + dt * thd
    thd_next = thd + dt * thdd

    return torch.stack([pos_next, vel_next, th_next, thd_next], dim=1)


# -----------------------------
# 2) Dataset collection (like your X,Y,U)
# -----------------------------
@torch.no_grad()
def collect_data(n_traj=2000, horizon=50, dt=0.02):
    """
    Returns:
      X: (N,4), U: (N,1), Y: (N,4)  where Y is x_{k+1}
    """
    # random initial states (small angles/velocities typical)
    x = torch.zeros(n_traj, 4)
    x[:, 0] = (torch.rand(n_traj) * 2 - 1) * 0.5   # pos in [-0.5,0.5]
    x[:, 1] = (torch.rand(n_traj) * 2 - 1) * 0.5   # vel
    x[:, 2] = (torch.rand(n_traj) * 2 - 1) * 0.5   # theta in [-0.5,0.5] rad
    x[:, 3] = (torch.rand(n_traj) * 2 - 1) * 1.0   # theta_dot

    Xs, Us, Ys = [], [], []
    for _ in range(horizon):
        u = (torch.rand(n_traj, 1) * 2 - 1) * 10.0  # force in [-10,10]
        x_next = cartpole_step(x, u, dt=dt)

        Xs.append(x.clone())
        Us.append(u.clone())
        Ys.append(x_next.clone())
        x = x_next

    X = torch.cat(Xs, dim=0)
    U = torch.cat(Us, dim=0)
    Y = torch.cat(Ys, dim=0)
    return X, U, Y


# -----------------------------
# 3) Deep Koopman model:
# z = encoder(x)  (learned basis)
# z_next = A z + B u  (linear Koopman in lifted space)
# x_hat_next = decoder(z_next)
# plus consistency loss: encoder(y) ~ z_next
# -----------------------------
class DeepKoopman(nn.Module):
    def __init__(self, state_dim=4, control_dim=1, lift_dim=64):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.lift_dim = lift_dim

        # Encoder = learned basis functions
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, lift_dim),
        )

        # Linear dynamics in lifted space (no bias to keep it "operator-like")
        self.A = nn.Linear(lift_dim, lift_dim, bias=False)
        self.B = nn.Linear(control_dim, lift_dim, bias=False)

        # Decoder back to state
        self.decoder = nn.Sequential(
            nn.Linear(lift_dim, 128),
            nn.Tanh(),
            nn.Linear(128, state_dim),
        )

    def lift(self, x):
        return self.encoder(x)

    def koopman_step(self, z, u):
        return self.A(z) + self.B(u)

    def forward(self, x, u):
        z = self.lift(x)
        z_next = self.koopman_step(z, u)
        x_next_hat = self.decoder(z_next)
        return x_next_hat, z, z_next


# -----------------------------
# 4) Training loop
# -----------------------------
def train_deep_koopman(
    epochs=2000,
    batch_size=4096,
    lr=3e-4,
    lift_dim=64,
    lambda_koop=0.2,
    lambda_rec=0.05,
):
    X, U, Y = collect_data(n_traj=2000, horizon=60)
    X, U, Y = X.to(device), U.to(device), Y.to(device)

    model = DeepKoopman(lift_dim=lift_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    n = X.shape[0]
    for ep in range(1, epochs + 1):
        idx = torch.randint(0, n, (batch_size,), device=device)
        x = X[idx]
        u = U[idx]
        y = Y[idx]

        x_hat, z, z_next = model(x, u)
        z_y = model.lift(y)

        # 1-step prediction loss in state space
        loss_pred = nn.MSELoss()(x_hat, y)

        # Koopman consistency: encoder(y) should match linear z_next
        loss_koop = nn.MSELoss()(z_next, z_y)

        # Optional: reconstruction regularizer x ≈ decoder(encoder(x))
        x_rec = model.decoder(model.lift(x))
        loss_rec = nn.MSELoss()(x_rec, x)

        loss = loss_pred + lambda_koop * loss_koop + lambda_rec * loss_rec

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if ep % 200 == 0:
            print(
                f"ep {ep:4d} | loss {loss.item():.3e} | "
                f"pred {loss_pred.item():.3e} | koop {loss_koop.item():.3e} | rec {loss_rec.item():.3e}"
            )

    return model


# -----------------------------
# 5) Rollout comparison: true vs Koopman
# -----------------------------
@torch.no_grad()
def rollout_compare(model, T=200, dt=0.02):
    model.eval()
    # initial condition
    x_true = torch.tensor([[0.0, 0.0, 0.3, 0.0]], device=device)
    z = model.lift(x_true)

    xs_true = [x_true.cpu()]
    xs_koop = [x_true.cpu()]

    # fixed control (or you can make it time-varying)
    for k in range(T):
        u = torch.tensor([[5.0 * math.sin(0.05 * k)]], device=device)  # smooth forcing

        # true system
        x_true = cartpole_step(x_true, u, dt=dt)
        xs_true.append(x_true.cpu())

        # koopman (lifted linear)
        z = model.koopman_step(z, u)
        x_hat = model.decoder(z)
        xs_koop.append(x_hat.cpu())

    xs_true = torch.cat(xs_true, dim=0)
    xs_koop = torch.cat(xs_koop, dim=0)

    rmse = torch.sqrt(torch.mean((xs_true - xs_koop) ** 2, dim=0))
    print("RMSE per state [pos, vel, theta, theta_dot]:", rmse.numpy())
    print("RMSE avg:", rmse.mean().item())

    return xs_true, xs_koop


if __name__ == "__main__":
    model = train_deep_koopman(
        epochs=2000,
        batch_size=4096,
        lr=3e-4,
        lift_dim=64,
        lambda_koop=0.2,
        lambda_rec=0.05,
    )
    xs_true, xs_koop = rollout_compare(model, T=300)
    import matplotlib.pyplot as plt
    T = xs_true.shape[0]
    t = np.arange(T) * 0.02
    plt.figure(figsize=(12, 8))
    labels = ['Position', 'Velocity', 'Angle', 'Angular Velocity']
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.plot(t, xs_true[:, i], label='True')
        plt.plot(t, xs_koop[:, i], label='Koopman', linestyle='--')
        plt.title(labels[i])
        plt.xlabel('Time [s]')
        plt.legend()
    plt.tight_layout()
    plt.show()  
