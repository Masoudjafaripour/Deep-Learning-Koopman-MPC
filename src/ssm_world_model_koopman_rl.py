# mamba_koopman_rl_cartpole.py
# Full demo: Selective-SSM ("Mamba-ish") + Koopman latent linearity + RL (A2C) on POMDP CartPole
# Runs on CPU/GPU. Produces plots.

import math
import random
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange, tqdm


# -----------------------
# Repro
# -----------------------
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)


# =======================
# 1) POMDP CartPole environment (continuous force)
# =======================

@dataclass
class EnvParams:
    dt: float = 0.02
    g: float = 9.8
    mc: float = 1.0
    mp: float = 0.1
    l: float = 0.5  # half-length
    force_limit: float = 10.0
    # Observation noise
    obs_noise_std: float = 0.02
    # Colored disturbance (AR(1)) that adds to force
    dist_rho: float = 0.95
    dist_std: float = 1.0
    # Termination thresholds
    x_limit: float = 2.4
    theta_limit: float = 12.0 * math.pi / 180.0  # 12 degrees


class POMDPCartPole:
    """
    True state x = [pos, vel, theta, theta_dot] (4D)
    Observation o = [pos, sin(theta), cos(theta)] + noise (3D)
    Hidden velocities => POMDP
    """
    def __init__(self, params: EnvParams):
        self.p = params
        self.reset()

    def reset(self):
        # small random init
        self.x = np.array([
            np.random.uniform(-0.05, 0.05),
            np.random.uniform(-0.05, 0.05),
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-0.05, 0.05)
        ], dtype=np.float32)
        self.d = 0.0  # colored disturbance state
        self.t = 0
        return self.obs()

    def obs(self):
        pos, vel, th, thd = self.x
        o = np.array([pos, math.sin(th), math.cos(th)], dtype=np.float32)
        o += np.random.randn(3).astype(np.float32) * self.p.obs_noise_std
        return o

    def step(self, u):
        # clip action
        u = float(np.clip(u, -self.p.force_limit, self.p.force_limit))

        # colored disturbance
        self.d = self.p.dist_rho * self.d + np.random.randn() * self.p.dist_std
        u_eff = u + float(self.d)

        # unpack
        dt = self.p.dt
        g = self.p.g
        mc = self.p.mc
        mp = self.p.mp
        l = self.p.l

        x, xdot, th, thdot = self.x
        total_mass = mc + mp
        polemass_length = mp * l
        costh = math.cos(th)
        sinth = math.sin(th)

        temp = (u_eff + polemass_length * thdot * thdot * sinth) / total_mass
        thacc = (g * sinth - costh * temp) / (l * (4.0/3.0 - mp * costh * costh / total_mass))
        xacc = temp - polemass_length * thacc * costh / total_mass

        # Euler integrate
        x = x + dt * xdot
        xdot = xdot + dt * xacc
        th = th + dt * thdot
        thdot = thdot + dt * thacc

        self.x = np.array([x, xdot, th, thdot], dtype=np.float32)
        self.t += 1

        done = (abs(x) > self.p.x_limit) or (abs(th) > self.p.theta_limit)
        # reward: keep upright + centered
        reward = 1.0 - 0.01*(x**2) - 0.05*(th**2) - 0.001*(u**2)
        if done:
            reward -= 5.0

        return self.obs(), reward, done, {"state": self.x.copy(), "dist": self.d}


# =======================
# 2) "Mamba-ish" selective SSM block (pure PyTorch)
# =======================

class MiniSelectiveSSM(nn.Module):
    """
    A lightweight selective SSM inspired by Mamba-style selection:
      s_{t+1} = a(x_t) ⊙ s_t + b(x_t)
      y_t     = c(x_t) ⊙ s_t + d(x_t)

    where a(x_t) in (0,1) acts like a learned discretization / gating (Δ)
    and b,c,d depend on input x_t (selection).
    """
    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # projections from token to parameters
        self.to_a = nn.Linear(d_model, d_state)   # gate
        self.to_b = nn.Linear(d_model, d_state)   # input inject into state
        self.to_c = nn.Linear(d_model, d_state)   # readout gate
        self.to_d = nn.Linear(d_model, d_model)   # skip / direct

        # map state->model dim
        self.state_to_y = nn.Linear(d_state, d_model)

    def forward(self, x):
        """
        x: (B,T,d_model)
        returns y: (B,T,d_model)
        """
        B, T, D = x.shape
        s = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)

        ys = []
        for t in range(T):
            xt = x[:, t, :]
            a = torch.sigmoid(self.to_a(xt))          # (B,d_state) in (0,1)
            b = self.to_b(xt)                         # (B,d_state)
            c = torch.tanh(self.to_c(xt))             # (B,d_state)
            d = self.to_d(xt)                         # (B,d_model)

            s = a * s + (1.0 - a) * b                 # gated update
            y_state = self.state_to_y(c * s)          # (B,d_model)
            y = y_state + d                            # skip connection
            ys.append(y.unsqueeze(1))

        return torch.cat(ys, dim=1)  # (B,T,d_model)


# =======================
# 3) World model: Selective SSM filter -> latent h_t, Koopman linear latent dynamics
# =======================

class MambaKoopmanWorldModel(nn.Module):
    """
    Inputs are sequences: (o_{t-L+1:t}, u_{t-L+1:t}) -> latent sequence h
    Koopman constraint: h_{t+1} ~ A h_t + B u_t
    Predict next observation: o_{t+1} ~ decoder(A h_t + B u_t)
    """
    def __init__(self, obs_dim=3, act_dim=1, d_model=128, d_state=32, latent_dim=64):
        super().__init__()
        self.obs_enc = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.act_enc = nn.Sequential(
            nn.Linear(act_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.ssm = MiniSelectiveSSM(d_model=d_model, d_state=d_state)
        self.to_latent = nn.Linear(d_model, latent_dim)

        # Koopman linear controlled latent dynamics
        self.A = nn.Linear(latent_dim, latent_dim, bias=False)
        self.B = nn.Linear(act_dim, latent_dim, bias=False)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, obs_dim),
        )

    def forward(self, o_seq, u_seq):
        """
        o_seq: (B,T,obs_dim)
        u_seq: (B,T,act_dim)  -- action aligned with o_t (action applied at time t)
        returns:
          h: (B,T,latent_dim)
          h_lin: (B,T-1,latent_dim) = A h_t + B u_t
          o_hat_next: (B,T-1,obs_dim)
        """
        xo = self.obs_enc(o_seq)
        xu = self.act_enc(u_seq)
        x = xo + xu

        y = self.ssm(x)
        h = self.to_latent(y)  # (B,T,latent)

        h_t = h[:, :-1, :]
        u_t = u_seq[:, :-1, :]
        h_lin = self.A(h_t) + self.B(u_t)
        o_hat_next = self.dec(h_lin)
        return h, h_lin, o_hat_next

    def koop_loss(self, h, h_lin):
        # enforce h_{t+1} ~= h_lin
        return ((h[:, 1:, :] - h_lin) ** 2).mean()

    def pred_loss(self, o_seq, o_hat_next):
        return ((o_seq[:, 1:, :] - o_hat_next) ** 2).mean()


@torch.no_grad()
def encode_latent_from_history(model, o_hist, u_hist):
    """
    o_hist: (1,L,obs_dim), u_hist: (1,L,act_dim)
    return h_t: (1,latent_dim) last latent
    """
    model.eval()
    h, _, _ = model(o_hist, u_hist)
    return h[:, -1, :]  # last latent


# =======================
# 4) Replay buffer that stores sequences
# =======================

class SeqReplay:
    def __init__(self, max_episodes=2000):
        self.episodes = []
        self.max_episodes = max_episodes

    def add_episode(self, obs, acts):
        # obs: (T+1,obs_dim), acts: (T,act_dim)
        self.episodes.append((obs, acts))
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)

    def sample_batch(self, batch_size=64, seq_len=32):
        """
        returns o_seq, u_seq with shapes:
          o_seq: (B,seq_len,obs_dim)
          u_seq: (B,seq_len,act_dim)
        """
        o_batch = []
        u_batch = []
        for _ in range(batch_size):
            obs, acts = random.choice(self.episodes)
            # need at least seq_len+1 obs and seq_len acts
            T = acts.shape[0]
            if T < seq_len:
                # fallback: pad by sampling another
                continue
            start = random.randint(0, T - seq_len)
            o_seq = obs[start:start+seq_len]           # aligns with acts
            u_seq = acts[start:start+seq_len]
            o_batch.append(o_seq)
            u_batch.append(u_seq)

        o_batch = torch.tensor(np.stack(o_batch, axis=0), device=device)
        u_batch = torch.tensor(np.stack(u_batch, axis=0), device=device)
        return o_batch, u_batch


# =======================
# 5) Policy + Value for A2C
# =======================

class ActorCritic(nn.Module):
    def __init__(self, latent_dim=64, act_limit=10.0):
        super().__init__()
        self.act_limit = act_limit
        self.pi = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.v = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # log std for Gaussian policy (scalar)
        self.log_std = nn.Parameter(torch.tensor([-0.2]))

    def forward(self, h):
        mu = self.pi(h)  # (B,1)
        v = self.v(h)    # (B,1)
        std = torch.exp(self.log_std).clamp(0.05, 5.0)
        return mu, std, v

    def sample_action(self, h):
        mu, std, v = self.forward(h)
        eps = torch.randn_like(mu)
        a = mu + std * eps
        a = torch.tanh(a) * self.act_limit
        # log prob with tanh squashing approx (good enough for this demo)
        # (if you want exact, use tanh-normal correction)
        logp = -0.5 * (((mu - a/self.act_limit).detach())**2 / (std**2) + 2*self.log_std + math.log(2*math.pi))
        logp = logp.sum(dim=-1, keepdim=True)
        return a, logp, v


# =======================
# 6) Data collection
# =======================

def rollout_episode(env, policy=None, model=None, hist_len=32, max_steps=500):
    """
    Collect an episode. If policy is None -> random actions.
    If policy given, uses latent from model + history (POMDP belief).
    Returns:
      obs_arr: (T+1,3)
      act_arr: (T,1)
      rew_arr: (T,)
      info_states: (T+1,4) true state logs
    """
    obs_list = []
    act_list = []
    rew_list = []
    state_list = []

    o = env.reset()
    obs_list.append(o.copy())
    state_list.append(env.x.copy())

    # history buffers (start with zeros)
    o_hist = [np.zeros(3, dtype=np.float32) for _ in range(hist_len)]
    u_hist = [np.zeros(1, dtype=np.float32) for _ in range(hist_len)]

    for t in range(max_steps):
        if policy is None or model is None:
            a = np.random.uniform(-env.p.force_limit, env.p.force_limit)
        else:
            # update history with current obs (before choosing action)
            o_hist.pop(0); o_hist.append(o.astype(np.float32))
            o_hist_t = torch.tensor(np.stack(o_hist)[None, ...], device=device)
            u_hist_t = torch.tensor(np.stack(u_hist)[None, ...], device=device)

            h = encode_latent_from_history(model, o_hist_t, u_hist_t)  # (1,latent)
            with torch.no_grad():
                mu, std, v = policy.forward(h)
                a_t = mu  # deterministic for rollout stability
                a = float(torch.tanh(a_t)[0, 0].cpu().item() * env.p.force_limit)

        o2, r, done, info = env.step(a)

        # append action after using it
        act_list.append([a])
        rew_list.append(r)

        obs_list.append(o2.copy())
        state_list.append(info["state"].copy())

        # update action history (after action)
        u_hist.pop(0); u_hist.append(np.array([a], dtype=np.float32))

        o = o2
        if done:
            break

    return (np.array(obs_list, dtype=np.float32),
            np.array(act_list, dtype=np.float32),
            np.array(rew_list, dtype=np.float32),
            np.array(state_list, dtype=np.float32))


# =======================
# 7) Train world model
# =======================

def train_world_model(model, replay, epochs=800, batch_size=64, seq_len=32,
                      lam_koop=0.5, lr=3e-4):
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)

    loss_hist = []
    pred_hist = []
    koop_hist = []

    # for ep in range(1, epochs + 1):
    for ep in trange(1, epochs + 1, desc="World Model Training"):
        o_seq, u_seq = replay.sample_batch(batch_size=batch_size, seq_len=seq_len)
        h, h_lin, o_hat_next = model(o_seq, u_seq)

        loss_pred = model.pred_loss(o_seq, o_hat_next)
        loss_koop = model.koop_loss(h, h_lin)
        loss = loss_pred + lam_koop * loss_koop

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        loss_hist.append(loss.item())
        pred_hist.append(loss_pred.item())
        koop_hist.append(loss_koop.item())

    return loss_hist, pred_hist, koop_hist


# =======================
# 8) A2C training on latent belief state
# =======================

def a2c_train(env_params, world_model, actor_critic,
              iters=150, gamma=0.99, lr=3e-4, hist_len=32, max_steps=500):
    ac = actor_critic
    opt = optim.Adam(ac.parameters(), lr=lr)

    returns = []
    # for it in range(1, iters + 1):
    for it in trange(1, iters + 1, desc="A2C Training"):
        env = POMDPCartPole(env_params)

        # collect on-policy trajectory using latent belief
        obs, acts, rews, states = rollout_episode(env, policy=ac, model=world_model,
                                                  hist_len=hist_len, max_steps=max_steps)
        G = float(np.sum(rews))
        returns.append(G)

        # build tensors for training
        # We need latent h_t for each step t from history window.
        T = acts.shape[0]
        h_list = []
        # create rolling histories
        o_hist = [np.zeros(3, dtype=np.float32) for _ in range(hist_len)]
        u_hist = [np.zeros(1, dtype=np.float32) for _ in range(hist_len)]

        for t in range(T):
            o_t = obs[t]
            o_hist.pop(0); o_hist.append(o_t)
            o_hist_t = torch.tensor(np.stack(o_hist)[None, ...], device=device)
            u_hist_t = torch.tensor(np.stack(u_hist)[None, ...], device=device)

            h_t = encode_latent_from_history(world_model, o_hist_t, u_hist_t)  # (1,d)
            h_list.append(h_t)

            # then update action history with the action taken at time t
            u_hist.pop(0); u_hist.append(acts[t].astype(np.float32))

        H = torch.cat(h_list, dim=0)  # (T,d)
        A = torch.tensor(acts, device=device)  # (T,1)
        R = torch.tensor(rews, device=device).unsqueeze(-1)  # (T,1)

        # compute returns-to-go
        RtG = torch.zeros_like(R)
        running = 0.0
        for t in reversed(range(T)):
            running = float(R[t].item()) + gamma * running
            RtG[t, 0] = running
        RtG = RtG.detach()

        # policy + value
        mu, std, V = ac.forward(H)
        # sample-free log prob of taken action (rough tanh-normal approx)
        # (good enough for this educational demo)
        a_scaled = A / env_params.force_limit
        a_scaled = torch.clamp(a_scaled, -0.999, 0.999)
        pre_tanh = torch.atanh(a_scaled)
        logp = -0.5 * (((pre_tanh - mu) ** 2) / (std ** 2) + 2*ac.log_std + math.log(2*math.pi))
        logp = logp.sum(dim=-1, keepdim=True)

        adv = (RtG - V).detach()

        loss_pi = -(logp * adv).mean()
        loss_v = ((V - RtG) ** 2).mean()
        loss = loss_pi + 0.5 * loss_v

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(ac.parameters(), 5.0)
        opt.step()

    return returns


# =======================
# 9) Main demo
# =======================

def main():
    env_params = EnvParams()
    env = POMDPCartPole(env_params)

    # ---------
    # Collect random data for world model
    # ---------
    replay = SeqReplay(max_episodes=2000)
    n_random_eps = 300
    max_steps = 300

    # for _ in range(n_random_eps):
    for _ in trange(n_random_eps, desc="Collecting Random Episodes"):
        obs, acts, rews, states = rollout_episode(env, policy=None, model=None, max_steps=max_steps)
        # need obs aligned with acts for sequences: we store obs[0:T], acts[0:T]
        # here obs has length T+1, acts length T
        replay.add_episode(obs[:-1], acts)

    # ---------
    # Train world model
    # ---------
    world = MambaKoopmanWorldModel(obs_dim=3, act_dim=1, d_model=128, d_state=32, latent_dim=64).to(device)

    loss_hist, pred_hist, koop_hist = train_world_model(
        world, replay, epochs=800, batch_size=64, seq_len=32, lam_koop=0.7, lr=3e-4
    )

    # ---------
    # Evaluate world model on a fresh random episode (rollout prediction)
    # ---------
    env_eval = POMDPCartPole(env_params)
    obs, acts, rews, states = rollout_episode(env_eval, policy=None, model=None, max_steps=250)
    T = acts.shape[0]
    hist_len = 32

    # Build predicted next observation step-by-step using:
    # h_t from history -> h_lin = A h_t + B u_t -> o_hat_{t+1} = dec(h_lin)
    o_hist = [np.zeros(3, dtype=np.float32) for _ in range(hist_len)]
    u_hist = [np.zeros(1, dtype=np.float32) for _ in range(hist_len)]
    o_hat_list = []
    o_true_next_list = []

    world.eval()
    with torch.no_grad():
        # for t in range(min(T, 200)):
        for t in tqdm(range(min(T, 200)), desc="Evaluating World Model"):
            o_t = obs[t]
            a_t = acts[t]

            o_hist.pop(0); o_hist.append(o_t.astype(np.float32))
            u_hist.pop(0); u_hist.append(a_t.astype(np.float32))

            o_hist_t = torch.tensor(np.stack(o_hist)[None, ...], device=device)
            u_hist_t = torch.tensor(np.stack(u_hist)[None, ...], device=device)

            # get latent at time t
            h_seq, _, _ = world(o_hist_t, u_hist_t)
            h_t = h_seq[:, -1, :]  # (1,d)

            # one-step prediction using Koopman linear latent
            u_t = torch.tensor(a_t[None, ...], device=device)
            h_next = world.A(h_t) + world.B(u_t)
            o_hat_next = world.dec(h_next).cpu().numpy().reshape(-1)

            o_hat_list.append(o_hat_next)
            o_true_next_list.append(obs[t+1].copy())

    o_hat_arr = np.array(o_hat_list)
    o_true_arr = np.array(o_true_next_list)

    # RMSE for observations
    rmse_obs = np.sqrt(np.mean((o_hat_arr - o_true_arr) ** 2, axis=0))

    # ---------
    # RL on latent belief state
    # ---------
    ac = ActorCritic(latent_dim=64, act_limit=env_params.force_limit).to(device)
    returns = a2c_train(env_params, world, ac, iters=200, lr=3e-4, hist_len=32, max_steps=400)

    # Rollout with learned policy
    env_pol = POMDPCartPole(env_params)
    obs_p, acts_p, rews_p, states_p = rollout_episode(env_pol, policy=ac, model=world, hist_len=32, max_steps=400)

    # Rollout with random policy for comparison
    env_rand = POMDPCartPole(env_params)
    obs_r, acts_r, rews_r, states_r = rollout_episode(env_rand, policy=None, model=None, max_steps=400)

    # ---------
    # PLOTS
    # ---------
    plt.figure()
    plt.plot(loss_hist, label="total")
    plt.plot(pred_hist, label="pred")
    plt.plot(koop_hist, label="koop")
    plt.title("World Model Training Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(o_true_arr[:, 0], label="true pos(t+1)")
    plt.plot(o_hat_arr[:, 0], label="pred pos(t+1)")
    plt.title(f"One-step Observation Prediction (pos). RMSE={rmse_obs[0]:.4f}")
    plt.xlabel("t")
    plt.ylabel("pos")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(o_true_arr[:, 1], label="true sin(theta)(t+1)")
    plt.plot(o_hat_arr[:, 1], label="pred sin(theta)(t+1)")
    plt.title(f"One-step Observation Prediction (sin theta). RMSE={rmse_obs[1]:.4f}")
    plt.xlabel("t")
    plt.ylabel("sin(theta)")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(returns)
    plt.title("A2C Returns (latent belief state from Selective-SSM + Koopman)")
    plt.xlabel("iteration")
    plt.ylabel("episode return")
    plt.grid(True)

    plt.figure()
    # Compare true theta over time under learned vs random
    th_pol = states_p[:, 2]
    th_rnd = states_r[:, 2]
    plt.plot(th_pol, label="theta (policy)")
    plt.plot(th_rnd, label="theta (random)")
    plt.title("Pole Angle Trajectory: Learned Policy vs Random")
    plt.xlabel("t")
    plt.ylabel("theta (rad)")
    plt.legend()
    plt.grid(True)

    print("Observation RMSE [pos, sin(theta), cos(theta)]:", rmse_obs)
    print("Final policy episode return:", float(np.sum(rews_p)))
    print("Random episode return:", float(np.sum(rews_r)))

    plt.show()


if __name__ == "__main__":
    main()
