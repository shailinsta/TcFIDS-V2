#!/usr/bin/env python3
"""
TC-FedIDS: Trajectory-Aware Federated Intrusion Detection System
================================================================
Architectural validation & aggregation simulation.

Requires Python 3.9+, numpy and torch; manuscript exports additionally require
pandas, matplotlib and seaborn. No parquet files are loaded by this program.
Manuscript tables/figures are fixed supplied values, not computed results.
The batch-gradient noise demonstration provides no formal DP guarantee.

This implementation  aligns the pipeline with stable, deployment-oriented
federated learning practices.

  1. GroupNorm instead of BatchNorm (non-IID + DP stable)
  2. Dynamic Huber threshold via residual MAD
  3. Continuous soft trust weights with floor τ ≥ 0.35
  4. Guarded FISTA with Lipschitz step-size bound
  5. Safe DP-SGD (float parameters only)

Run:
    python tc_fedids.py
"""



# ## 0. Environment

# In[1]:


import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Resolve Dataset relative to this file, including in ~/TCFIDS-2.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = str(PROJECT_ROOT / "Dataset")


def report_environment() -> None:
    print("Python executable:", sys.executable)
    print("NumPy:", np.__version__)
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("DATA_ROOT =", DATA_ROOT)
    if Path(DATA_ROOT).is_dir():
        print("Dataset entries:", ", ".join(sorted(os.listdir(DATA_ROOT))))
    else:
        print("Dataset folder absent; synthetic validation can still run.")
    print("MODE: synthetic updates and dummy-batch validation; no parquet training.")


# Feature dimension after preprocessing (as used in the paper)
FEATURE_DIM = {
    "NF-CSE-CIC-IDS2018": 10,
    "NF-ToN-IoT":         10,
    "UNSW-NB15":           5,   # Harm5
    "CIDDS-001":           5,   # Harm5
}

# Native attack rates (for reference)
NATIVE_ATTACK_RATE = {
    "NF-CSE-CIC-IDS2018": 0.0647,
    "NF-ToN-IoT":         0.8280,
    "UNSW-NB15":          0.5490,
    "CIDDS-001":          0.0149,
}


# =====================================================================
# 1. REPRODUCIBILITY
# =====================================================================

def set_deterministic_seed(seed: int = 11) -> None:
    """Seed random generators; backend determinism remains platform-dependent.

    PYTHONHASHSEED must be set before interpreter startup to affect hashing.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# 2. RES-MLP WITH GROUPNORM
# =====================================================================

class ResBlock(nn.Module):
    """Residual block using GroupNorm (no BatchNorm drift / DP buffer issues)."""

    def __init__(self, dim: int, num_groups: int = 8, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=dim)
        self.fc2 = nn.Linear(dim, dim)
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.gn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.gn2(self.fc2(out))
        return F.relu(out + residual)


class ResMLP(nn.Module):
    """
    Residual MLP for NetFlow intrusion classification.
    Default trainable parameters: 68,994 (10 inputs, 2 classes, width 128).
    """

    def __init__(self, in_features: int = 10, num_classes: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
            nn.ReLU(),
        )
        self.block1 = ResBlock(hidden_dim, num_groups=8)
        self.block2 = ResBlock(hidden_dim, num_groups=8)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x)


# =====================================================================
# 3. TRAJECTORY TRUST ENGINE (soft weights + floor τ ≥ 0.35)
# =====================================================================

class TrajectoryTrustEngine:
    """
    Rolling multi-round history → 4-D trajectory features →
    continuous soft trust scores with hard floor τ_i ≥ 0.35.
    """

    def __init__(
        self,
        history_len: int = 5,
        alpha_ema: float = 0.85,
        gamma: float = 3.0,
        trust_floor: float = 0.35,
    ):
        if history_len < 1 or not 0 <= alpha_ema <= 1 or not 0 <= trust_floor <= 1 or gamma < 0:
            raise ValueError("Invalid history length, EMA, trust floor or gamma.")
        self.H = history_len
        self.alpha = alpha_ema
        self.gamma = gamma
        self.floor = trust_floor
        self.history: Dict[int, List[torch.Tensor]] = {}
        self.prev_trust: Dict[int, float] = {}
        self.env_references: Dict[int, torch.Tensor] = {}

    def extract_trajectory_features(
        self, client_id: int, current_update: torch.Tensor
    ) -> torch.Tensor:
        if client_id not in self.history:
            self.history[client_id] = []

        self.history[client_id].append(current_update.detach().clone())
        if len(self.history[client_id]) > self.H:
            self.history[client_id].pop(0)

        hist = self.history[client_id]
        if len(hist) < 2:
            return torch.zeros(4, device=current_update.device)

        S = torch.stack(hist)  # (K, D)
        K = S.shape[0]

        # 1. Direction Persistence
        norms = torch.norm(S, dim=1, keepdim=True) + 1e-8
        S_norm = S / norms
        cos_sims = torch.sum(S_norm[:-1] * S_norm[1:], dim=1)
        persist = torch.mean(cos_sims)

        # 2. Spectral Entropy
        s_norms = torch.norm(S, dim=1)
        p = s_norms / (torch.sum(s_norms) + 1e-8)
        p = torch.clamp(p, min=1e-8)
        spec_ent = -torch.sum(p * torch.log(p))

        # 3. Second-Order Acceleration
        if K >= 3:
            accel = torch.mean(torch.norm(S[2:] - 2 * S[1:-1] + S[:-2], dim=1))
        else:
            accel = torch.tensor(0.0, device=current_update.device)

        # 4. Norm Variance
        norm_var = (
            torch.var(s_norms)
            if K > 1
            else torch.tensor(0.0, device=current_update.device)
        )

        return torch.stack([persist, spec_ent, accel, norm_var])

    def compute_trust(
        self,
        client_updates: List[torch.Tensor],
        env_labels: torch.Tensor,
        val_accuracies: List[float],
    ) -> torch.Tensor:
        N = len(client_updates)
        if not N or len(val_accuracies) != N or env_labels.numel() != N:
            raise ValueError("Updates, environment labels and validation scores must align.")
        env_labels = env_labels.to(client_updates[0].device)
        updates_stacked = torch.stack(client_updates)
        median_update = torch.median(updates_stacked, dim=0).values

        risks = []
        for i in range(N):
            u = client_updates[i]
            env_id = int(env_labels[i].item())
            phi = self.extract_trajectory_features(i, u)

            # Geometric & Norm risks
            r_geo = 1.0 - F.cosine_similarity(
                u.unsqueeze(0), median_update.unsqueeze(0)
            ).item()
            norm_m = torch.norm(median_update).item()
            r_norm = abs(torch.norm(u).item() - norm_m) / (norm_m + 1e-8)

            # Trajectory composite risk
            p_val = max(phi[0].item(), 0.0)
            e_val = min(phi[1].item() * 1.5, 1.0)
            a_val = min(phi[2].item() * 2.67, 1.0)
            v_val = min(phi[3].item() * 3.75, 1.0)
            r_traj = (
                0.55 * (1.0 - p_val)
                + 0.25 * e_val
                + 0.15 * a_val
                + 0.05 * v_val
            )

            # Environment alignment risk
            if env_id in self.env_references:
                ref_u = self.env_references[env_id]
                r_env = 1.0 - F.cosine_similarity(
                    u.unsqueeze(0), ref_u.unsqueeze(0)
                ).item()
            else:
                r_env = 0.25

            # Validation residual risk
            r_val = 1.0 - val_accuracies[i]

            # Composite risk
            r_composite = (
                0.32 * r_geo
                + 0.22 * r_norm
                + 0.28 * r_traj
                + 0.10 * r_env
                + 0.08 * r_val
            )
            risks.append(r_composite)

        risks_t = torch.tensor(risks, device=updates_stacked.device)
        mad = torch.median(torch.abs(risks_t - torch.median(risks_t)))
        theta = torch.median(risks_t) + 0.75 * mad

        trust_scores = []
        for i in range(N):
            # Stable sigmoid avoids math.exp overflow for large finite risks.
            raw_trust = torch.sigmoid(
                -self.gamma * (risks_t[i] - theta)
            ).item()
            prev = self.prev_trust.get(i, raw_trust)
            smoothed = self.alpha * raw_trust + (1.0 - self.alpha) * prev

            # Soft weight with hard floor
            final_trust = max(self.floor, smoothed)
            self.prev_trust[i] = final_trust
            trust_scores.append(final_trust)

        # Update environment references from high-trust clients
        trust_tensor = torch.tensor(trust_scores, device=updates_stacked.device)
        for env_id in torch.unique(env_labels):
            e_int = int(env_id.item())
            idx = (
                (env_labels == env_id) & (trust_tensor > 0.55)
            ).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                self.env_references[e_int] = (
                    torch.mean(updates_stacked[idx], dim=0).detach()
                )

        return trust_tensor


# =====================================================================
# 4. ADAPTIVE GROUPDRO-FISTA AGGREGATOR
# =====================================================================

class AdaptiveGroupDROFISTAAggregator:
    """
    Huber-regularized GroupDRO solved by guarded FISTA.
    Dynamic residual MAD threshold + Lipschitz-bounded step size.
    """

    def __init__(
        self,
        num_environments: int = 4,
        lambda_huber: float = 0.06,
        eta_gd: float = 0.05,
        t_max: int = 12,
        tol: float = 1e-5,
    ):
        if num_environments < 1 or lambda_huber < 0 or eta_gd < 0 or t_max < 1 or tol <= 0:
            raise ValueError("Invalid aggregation configuration.")
        self.num_environments = num_environments
        self.lambda_huber = lambda_huber
        self.eta_gd = eta_gd
        self.t_max = t_max
        self.tol = tol

    def compute_adaptive_huber_delta(
        self, client_updates: List[torch.Tensor], median_update: torch.Tensor
    ) -> float:
        """Dynamic δ from parameter residual MAD (adapts to Gaussian noise)."""
        residuals = torch.stack(
            [torch.norm(u - median_update) for u in client_updates]
        )
        mad = torch.median(torch.abs(residuals - torch.median(residuals)))
        delta = torch.clamp(mad * 1.4826, min=0.5, max=3.0).item()
        return delta

    def aggregate(
        self,
        client_updates: List[torch.Tensor],
        trust_weights: torch.Tensor,
        env_labels: torch.Tensor,
    ) -> torch.Tensor:
        if not client_updates:
            raise ValueError("At least one client update is required.")
        updates_stacked = torch.stack(client_updates)  # (N, D)
        env_labels = env_labels.to(device=updates_stacked.device, dtype=torch.long)
        trust_weights = trust_weights.to(updates_stacked)
        if env_labels.numel() != len(client_updates) or trust_weights.numel() != len(client_updates):
            raise ValueError("One environment label and trust weight is required per update.")
        if not torch.isfinite(updates_stacked).all() or not torch.isfinite(trust_weights).all():
            raise ValueError("Updates and trust weights must be finite.")
        if (trust_weights <= 0).any():
            raise ValueError("Trust weights must be positive.")
        if ((env_labels < 0) | (env_labels >= self.num_environments)).any():
            raise ValueError("Environment label outside configured range.")
        median_update = torch.median(updates_stacked, dim=0).values

        delta = self.compute_adaptive_huber_delta(client_updates, median_update)

        w = median_update.clone()
        y = median_update.clone()
        t = 1.0

        # The softmax risk Hessian includes a covariance term. Bound it by
        # 4 * eta * diameter^2; 2 + lambda alone is not sufficient.
        radius = torch.linalg.vector_norm(updates_stacked - median_update, dim=1).max().item()
        L = 2.0 + self.lambda_huber + 16.0 * self.eta_gd * radius ** 2 + 1e-6
        step_size = 0.8 / L
        active_envs = torch.bincount(env_labels, minlength=self.num_environments) > 0

        for _ in range(self.t_max):
            R_e = torch.zeros(self.num_environments, device=w.device)
            grad_R_e = torch.zeros(
                (self.num_environments, updates_stacked.shape[1]),
                device=w.device,
            )

            for e in range(self.num_environments):
                idx = (env_labels == e).nonzero(as_tuple=True)[0]
                if len(idx) == 0:
                    continue
                e_updates = updates_stacked[idx]
                e_trusts = trust_weights[idx]
                norm_trusts = e_trusts / (torch.sum(e_trusts) + 1e-8)

                diffs = y.unsqueeze(0) - e_updates
                sq_norms = torch.sum(diffs ** 2, dim=1)

                R_e[e] = torch.sum(norm_trusts * sq_norms)
                grad_R_e[e] = 2.0 * torch.sum(
                    norm_trusts.unsqueeze(1) * diffs, dim=0
                )

            q = F.softmax((self.eta_gd * R_e).masked_fill(~active_envs, -torch.inf), dim=0)
            grad_risk = torch.sum(q.unsqueeze(1) * grad_R_e, dim=0)

            # Smooth Huber regularizer centered at median
            diff_m = y - median_update
            abs_diff_m = torch.abs(diff_m)
            grad_huber = torch.where(
                abs_diff_m <= delta,
                diff_m,
                delta * torch.sign(diff_m),
            )

            total_grad = grad_risk + self.lambda_huber * grad_huber

            # FISTA update
            w_next = y - step_size * total_grad
            if not torch.isfinite(w_next).all():
                raise FloatingPointError("Aggregation produced a non-finite update.")
            t_next = (1.0 + math.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
            y = w_next + ((t - 1.0) / t_next) * (w_next - w)

            step_norm = torch.norm(w_next - w) / (torch.norm(w) + 1e-8)
            w = w_next
            t = t_next

            if step_norm < self.tol:
                break

        return w


# =====================================================================
# 5. SAFE DIFFERENTIAL PRIVACY
# =====================================================================

def apply_safe_dp_sgd(
    model: nn.Module,
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 0.5,
) -> None:
    """
    Demonstrate batch-gradient clipping and Gaussian noise on float parameters.
    This is not per-example DP-SGD and supplies no epsilon/delta accounting.
    Integer tracking buffers are completely ignored.
    """
    if max_grad_norm <= 0 or noise_multiplier < 0:
        raise ValueError("Gradient norm must be positive and noise multiplier nonnegative.")
    float_parameters = [
        p
        for p in model.parameters()
        if p.requires_grad and p.is_floating_point()
    ]
    if not float_parameters:
        return

    torch.nn.utils.clip_grad_norm_(float_parameters, max_norm=max_grad_norm)

    for p in float_parameters:
        if p.grad is not None:
            noise = torch.normal(
                mean=0.0,
                std=max_grad_norm * noise_multiplier,
                size=p.grad.shape,
                device=p.grad.device,
                dtype=p.dtype,
            )
            p.grad.add_(noise)


# =====================================================================
# 6. ALIE ATTACK
# =====================================================================

def apply_alie_attack(
    honest_updates: List[torch.Tensor],
    num_byzantine: int,
    z_max: float = 1.5,
) -> List[torch.Tensor]:
    """Construct ALIE malicious updates."""
    if not honest_updates or num_byzantine < 0:
        raise ValueError("ALIE requires honest updates and a nonnegative Byzantine count.")
    stacked = torch.stack(honest_updates)
    mu = torch.mean(stacked, dim=0)
    std = torch.std(stacked, dim=0, unbiased=len(honest_updates) > 1) + 1e-8
    malicious_update = mu - z_max * std
    return [malicious_update.clone() for _ in range(num_byzantine)]


# =====================================================================
# 7. MAIN SIMULATION
# =====================================================================

def main(rounds: int = 5, seed: int = 11, device_name: str = "auto") -> None:
    print("=" * 70)
    print("TC-FedIDS: Trajectory-Aware Federated Intrusion Detection System")
    print("Architecture Validation & Aggregation Round Simulation")
    print("=" * 70)

    report_environment()
    set_deterministic_seed(seed)

    # Federation parameters (same as original)
    N_CLIENTS = 28
    BYZANTINE_RATIO = 0.30
    N_BYZANTINE = int(N_CLIENTS * BYZANTINE_RATIO)
    N_HONEST = N_CLIENTS - N_BYZANTINE
    N_ENVIRONMENTS = 4

    print(f"[CONFIGURATION] Total Clients     : {N_CLIENTS}")
    print(f"[CONFIGURATION] Honest Clients    : {N_HONEST}")
    print(f"[CONFIGURATION] Byzantine (ALIE)  : {N_BYZANTINE}")
    print(f"[CONFIGURATION] Environments      : {N_ENVIRONMENTS}")

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu.")
    global_model = ResMLP(in_features=10, num_classes=2, hidden_dim=128).to(device)
    param_count = sum(
        p.numel() for p in global_model.parameters() if p.requires_grad
    )
    print(f"[MODEL] ResMLP on {device} | Trainable parameters: {param_count:,}")

    trust_engine = TrajectoryTrustEngine(
        history_len=5, alpha_ema=0.85, gamma=3.0, trust_floor=0.35
    )
    aggregator = AdaptiveGroupDROFISTAAggregator(
        num_environments=N_ENVIRONMENTS,
        lambda_huber=0.06,
        eta_gd=0.05,
    )

    env_labels = torch.tensor(
        [i % N_ENVIRONMENTS for i in range(N_CLIENTS)], device=device
    )

    print(f"\n[EXECUTION] Simulating {rounds} synthetic communication rounds...")
    for round_num in range(1, rounds + 1):
        # Synthetic honest updates (no real dataset required)
        honest_updates = [
            torch.randn(param_count, device=device) * 0.1 + (0.02 * i)
            for i in range(N_HONEST)
        ]

        byzantine_updates = apply_alie_attack(
            honest_updates, num_byzantine=N_BYZANTINE
        )
        all_updates = honest_updates + byzantine_updates

        val_accuracies = [
            0.92 + (0.01 * (i % 5)) if i < N_HONEST else 0.45
            for i in range(N_CLIENTS)
        ]

        trust_weights = trust_engine.compute_trust(
            all_updates, env_labels, val_accuracies
        )
        agg_update = aggregator.aggregate(
            all_updates, trust_weights, env_labels
        )

        min_trust = torch.min(trust_weights).item()
        mean_trust_honest = torch.mean(trust_weights[:N_HONEST]).item()
        mean_trust_byz = torch.mean(trust_weights[N_HONEST:]).item()

        print(
            f"  Round {round_num}/{rounds} | "
            f"AggNorm={torch.norm(agg_update):.4f} | "
            f"HonestTrust={mean_trust_honest:.3f} | "
            f"ByzTrust={mean_trust_byz:.3f} | "
            f"MinFloor={min_trust:.3f}"
        )

    # Safe DP-SGD demonstration
    print("\n[PRIVACY] Testing Safe Differential Privacy Noise Injection...")
    dummy_input = torch.randn(32, 10, device=device)
    dummy_target = torch.randint(0, 2, (32,), device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(global_model.parameters(), lr=1e-3)

    optimizer.zero_grad()
    outputs = global_model(dummy_input)
    loss = criterion(outputs, dummy_target)
    loss.backward()

    apply_safe_dp_sgd(global_model, max_grad_norm=1.0, noise_multiplier=0.5)
    optimizer.step()

    print("  [SUCCESS] Batch-gradient noise demonstration completed (no formal DP guarantee).")
    print("=" * 70)
    print("TC-FedIDS Codebase Verification Completed Successfully.")
    print("=" * 70)


# =====================================================================
# 8. EXPERIMENTAL TABLES + FIGURES
# =====================================================================

def print_all_tables_and_figures(save_dir: str = ".") -> None:
    """
    Export nine fixed manuscript tables and seven fixed manuscript figures.
    These values are supplied reference data, not outputs of this simulation.
    """
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")
    os.makedirs(save_dir, exist_ok=True)
    Path(save_dir, "PROVENANCE.txt").write_text(
        "All CSV and PNG values are hard-coded manuscript reference values.\n"
        "They were not computed from Dataset or from the synthetic simulation.\n"
        "Historical parameter counts/privacy claims are retained as supplied;\n"
        "they do not describe validation of this implementation.\n", encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print("FIXED MANUSCRIPT REFERENCE TABLES — NOT NEW EXPERIMENT RESULTS")
    print("=" * 80)

    # ----- Table 1 -----
    print("\n--- Table 1. Characteristics of the Four Real Network Environments ---")
    t1 = pd.DataFrame({
        "Source": ["NF-CSE-CIC-IDS2018", "NF-ToN-IoT", "UNSW-NB15", "CIDDS-001"],
        "Processing Mode": ["Full", "Full", "Harm5", "Harm5"],
        "Sub-sampled Flows": [50000, 50000, 50000, 50000],
        "Native Attack Rate": [0.0647, 0.8280, 0.5490, 0.0149],
        "Feature Dimension": [10, 10, 5, 5],
    })
    print(t1.to_string(index=False))
    t1.to_csv(os.path.join(save_dir, "table1_environments.csv"), index=False)

    # ----- Table 2 -----
    print("\n--- Table 2. Clean Baseline Performance (0% Byzantine, ε=∞) ---")
    t2 = pd.DataFrame({
        "Aggregator": ["FedAvg", "Coordinate Median", "Trimmed Mean", "Krum", "TC-FedIDS"],
        "Balanced Accuracy (Mean ± Std)": ["0.9270 ± 0.0014", "0.9254 ± 0.0018", "0.9263 ± 0.0016", "0.8029 ± 0.0116", "0.9137 ± 0.0053"],
        "Macro-F1 (Mean ± Std)": ["0.9118 ± 0.0019", "0.9098 ± 0.0022", "0.9107 ± 0.0019", "0.8036 ± 0.0099", "0.9001 ± 0.0044"],
        "Minority Recall (Mean ± Std)": ["0.9688 ± 0.0029", "0.9687 ± 0.0025", "0.9693 ± 0.0025", "0.7451 ± 0.0253", "0.9466 ± 0.0100"],
        "Total Wall Time (s)": [1441.76, 1445.66, 1447.13, 1453.93, 1496.76],
        "s/round": [48.06, 48.19, 48.24, 48.46, 49.89],
        "Parameters": [77122, 77122, 77122, 77122, 77122],
    })
    print(t2.to_string(index=False))
    t2.to_csv(os.path.join(save_dir, "table2_clean_baseline.csv"), index=False)

    # ----- Table 3 -----
    print("\n--- Table 3. Robustness under 30% ALIE Byzantine Attack (ε=∞) ---")
    t3 = pd.DataFrame({
        "Aggregator": ["FedAvg", "Coordinate Median", "Trimmed Mean", "Krum", "TC-FedIDS"],
        "Balanced Accuracy (Mean ± Std)": ["0.9211 ± 0.0055", "0.9169 ± 0.0064", "0.9015 ± 0.0201", "0.7152 ± 0.0697", "0.9049 ± 0.0173"],
        "Macro-F1 (Mean ± Std)": ["0.9091 ± 0.0051", "0.9037 ± 0.0097", "0.8876 ± 0.0140", "0.7207 ± 0.0791", "0.8859 ± 0.0234"],
        "Minority Recall (Mean ± Std)": ["0.9489 ± 0.0138", "0.9487 ± 0.0119", "0.9330 ± 0.0551", "0.5208 ± 0.1635", "0.9598 ± 0.0108"],
        "Total Wall Time (s)": [1454.00, 1454.59, 1453.89, 1463.86, 1497.92],
        "s/round": [48.47, 48.49, 48.46, 48.80, 49.93],
        "Parameters": [77122, 77122, 77122, 77122, 77122],
    })
    print(t3.to_string(index=False))
    t3.to_csv(os.path.join(save_dir, "table3_alie_robustness.csv"), index=False)

    # ----- Table 4 -----
    print("\n--- Table 4. Component Ablation under 30% ALIE ---")
    t4 = pd.DataFrame({
        "Architectural Variant": ["Coordinate Median", "TC-Median (Hard Filter)", "Full TC-FedIDS (Soft Weights + GroupDRO)"],
        "Balanced Accuracy (Mean ± Std)": ["0.9169 ± 0.0064", "0.8975 ± 0.0241", "0.9049 ± 0.0173"],
        "Macro-F1 (Mean ± Std)": ["0.9037 ± 0.0097", "0.8858 ± 0.0203", "0.8859 ± 0.0234"],
        "Minority Recall (Mean ± Std)": ["0.9487 ± 0.0119", "0.9194 ± 0.0501", "0.9598 ± 0.0108"],
        "Total Wall Time (s)": [1445.87, 1480.94, 1491.64],
        "s/round": [48.20, 49.37, 49.72],
        "Parameters": [77122, 77122, 77122],
    })
    print(t4.to_string(index=False))
    t4.to_csv(os.path.join(save_dir, "table4_ablation.csv"), index=False)

    # ----- Table 5 -----
    print("\n--- Table 5. Computational Cost and Latency ---")
    t5 = pd.DataFrame({
        "Aggregator Method": ["FedAvg", "Coordinate Median", "Trimmed Mean", "Krum", "TC-FedIDS"],
        "Total Wall Time (s)": [1454.00, 1454.59, 1453.89, 1463.86, 1497.92],
        "Average Latency (s/round)": [48.47, 48.49, 48.46, 48.80, 49.93],
        "Overhead vs FedAvg (%)": ["0.00%", "+0.04%", "-0.02%", "+0.68%", "+2.97%"],
        "Total Parameters": [77122, 77122, 77122, 77122, 77122],
    })
    print(t5.to_string(index=False))
    t5.to_csv(os.path.join(save_dir, "table5_latency.csv"), index=False)

    # ----- Table 6 -----
    print("\n--- Table 6. Differential Privacy Sweep under 30% ALIE (δ=1e-5) ---")
    t6 = pd.DataFrame({
        "Target Privacy Budget (ε)": ["Unconstrained (ε=∞)", "Moderate (ε=8.0)", "Strict (ε=4.0)", "High (ε=2.0)"],
        "Balanced Accuracy (Mean ± Std)": ["0.8987 ± 0.0130", "0.5304 ± 0.0526", "0.5000 ± 0.0000", "0.5000 ± 0.0000"],
        "Macro-F1 (Mean ± Std)": ["0.8773 ± 0.0185", "0.3296 ± 0.1085", "0.2669 ± 0.0000", "0.2669 ± 0.0000"],
        "Minority Recall (Mean ± Std)": ["0.9629 ± 0.0104", "0.9947 ± 0.0092", "1.0000 ± 0.0000", "1.0000 ± 0.0000"],
        "Processing Time (s/round)": [50.30, 51.28, 51.41, 51.40],
        "Operational State": ["Functional Detection", "Utility Degradation", "Model Collapse", "Model Collapse"],
    })
    print(t6.to_string(index=False))
    t6.to_csv(os.path.join(save_dir, "table6_dp_sweep.csv"), index=False)

    # ----- Table 7 -----
    print("\n--- Table 7. Empirical Byzantine Robustness Radius under ALIE ---")
    t7 = pd.DataFrame({
        "Evaluated Byzantine Fraction (β)": ["0.00 (0%)", "0.15 (15%)", "0.25 (25%)", "0.35 (35%)", "0.40 (40%)"],
        "Mean Minority Recall": [0.9133, 0.9541, 0.9524, 0.9680, 0.9743],
        "Threshold Satisfied (MinRec ≥ 0.50)": [True, True, True, True, True],
        "Empirical Diagnostic": [
            "Clean Baseline Operating Point",
            "Robust Detection Maintained",
            "Robust Detection Maintained",
            "Robust Detection Maintained",
            "Robust Detection Maintained",
        ],
    })
    print(t7.to_string(index=False))
    t7.to_csv(os.path.join(save_dir, "table7_robustness_radius.csv"), index=False)

    # ----- Table 8 -----
    print("\n--- Table 8. Performance Across Byzantine Attack Types (30% Participation) ---")
    t8 = pd.DataFrame({
        "Attack Threat Model": ["ALIE Attack", "Sign-Flip Attack", "Gaussian Noise Attack"],
        "Balanced Accuracy (Mean ± Std)": ["0.8987 ± 0.0130", "0.8500 ± 0.0040", "0.6941 ± 0.1265"],
        "Macro-F1 (Mean ± Std)": ["0.8773 ± 0.0185", "0.8428 ± 0.0070", "0.6842 ± 0.1371"],
        "Minority Recall (Mean ± Std)": ["0.9629 ± 0.0104", "0.8437 ± 0.0404", "0.5448 ± 0.3138"],
        "Recall Variance (σ)": ["Stable (σ=0.0104)", "Moderate (σ=0.0404)", "High Variance (σ=0.3138)"],
    })
    print(t8.to_string(index=False))
    t8.to_csv(os.path.join(save_dir, "table8_attack_types.csv"), index=False)

    # ----- Table 9 -----
    print("\n--- Table 9. System Positioning Relative to Recent Literature ---")
    t9 = pd.DataFrame({
        "Study / System": [
            "Thein et al. (2024)", "Jin et al. (2024)", "Doriguzzi-Corin (2024)",
            "SAFE-IDS (2025)", "TrustFed-IDS (2026)", "FedCmp (2025)",
            "DBFL (2025)", "TPFL (2025)", "TC-FedIDS (Proposed)",
        ],
        "FL-IDS Domain": ["Yes", "Yes", "Yes", "Yes", "Yes", "No", "No", "No", "Yes"],
        "Non-IID Handling": [
            "Personalization", "Incremental Drift", "Loss-based",
            "Gradient Sign", "Trust-based", "Clustering",
            "Dynamic Clipping", "Personalization", "GroupDRO Risk",
        ],
        "Trajectory Tracking": [
            "None", "None", "None", "None", "Short-term", "None", "None", "None", "4D Multi-Round",
        ],
        "Aggregation Engine": [
            "Weighted Averaging", "Adaptive Aggregation", "Adaptive Participation",
            "Weighted Loss", "Multi-Criteria Soft Weights", "Cluster Voting",
            "Robust Distance Aggregation", "Similarity Matrix", "Huber-Median GroupDRO-FISTA",
        ],
        "Differential Privacy": ["No", "No", "No", "Yes", "No", "No", "Yes", "Yes", "Rényi DP"],
        "Focus Metric": [
            "Class Weights", "None", "None", "Weighted Loss", "Class Balanced",
            "None", "None", "None", "Minority Recall",
        ],
    })
    print(t9.to_string(index=False))
    t9.to_csv(os.path.join(save_dir, "table9_literature.csv"), index=False)

    print("\nAll 9 tables printed and saved as CSV.")

    # =================================================================
    # FIGURES
    # =================================================================
    print("\n" + "=" * 80)
    print("GENERATING MANUSCRIPT FIGURES")
    print("=" * 80)

    aggregators = ["FedAvg", "Coord.\nMedian", "Trimmed\nMean", "Krum", "TC-FedIDS"]
    x = np.arange(len(aggregators))
    width = 0.35

    # Figure 1
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bacc = [0.9270, 0.9254, 0.9263, 0.8029, 0.9137]
    macro_f1 = [0.9118, 0.9098, 0.9107, 0.8036, 0.9001]
    min_recall = [0.9688, 0.9687, 0.9693, 0.7451, 0.9466]
    axes[0].bar(x - width/2, bacc, width, label="Balanced Accuracy", color="#4C72B0")
    axes[0].bar(x + width/2, macro_f1, width, label="Macro-F1", color="#55A868")
    axes[0].set_ylabel("Score"); axes[0].set_title("(a) Balanced Accuracy and Macro-F1 (Clean)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(aggregators); axes[0].legend(); axes[0].set_ylim(0.7, 1.0)
    axes[1].bar(aggregators, min_recall, color=["#4C72B0"]*3 + ["#C44E52", "#8172B2"])
    axes[1].set_ylabel("Minority-Class Recall"); axes[1].set_title("(b) Minority Recall (Clean)")
    axes[1].set_ylim(0.7, 1.0)
    for i, v in enumerate(min_recall):
        axes[1].text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)
    plt.suptitle("Figure 1. Clean-Setting Performance Profiles", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.text(0.5, -0.02, "Fixed manuscript reference values — not computed by this run", ha="center", fontsize=9)
    fig.savefig(os.path.join(save_dir, "fig1_clean_setting.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Figure 1 saved")

    # Figure 2
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bacc_alie = [0.9211, 0.9169, 0.9015, 0.7152, 0.9049]
    macro_alie = [0.9091, 0.9037, 0.8876, 0.7207, 0.8859]
    recall_alie = [0.9489, 0.9487, 0.9330, 0.5208, 0.9598]
    axes[0].bar(x - width/2, bacc_alie, width, label="Balanced Accuracy", color="#4C72B0")
    axes[0].bar(x + width/2, macro_alie, width, label="Macro-F1", color="#55A868")
    axes[0].set_ylabel("Score"); axes[0].set_title("(a) Global Metrics under 30% ALIE")
    axes[0].set_xticks(x); axes[0].set_xticklabels(aggregators); axes[0].legend(); axes[0].set_ylim(0.6, 1.0)
    axes[1].bar(aggregators, recall_alie, color=["#4C72B0"]*3 + ["#C44E52", "#8172B2"])
    axes[1].set_ylabel("Minority-Class Recall"); axes[1].set_title("(b) Minority Recall under 30% ALIE")
    axes[1].set_ylim(0.4, 1.05)
    for i, v in enumerate(recall_alie):
        axes[1].text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold" if i == 4 else "normal")
    plt.suptitle("Figure 2. Performance under 30% ALIE Byzantine Attack", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.text(0.5, -0.02, "Fixed manuscript reference values — not computed by this run", ha="center", fontsize=9)
    fig.savefig(os.path.join(save_dir, "fig2_alie_attack.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Figure 2 saved")

    # Figure 3
    fig, ax = plt.subplots(figsize=(10, 5))
    variants = ["Coordinate\nMedian", "TC-Median\n(Hard Filter)", "Full TC-FedIDS\n(Soft + GroupDRO)"]
    recalls = [0.9487, 0.9194, 0.9598]
    ax.bar(variants, recalls, color=["#4C72B0", "#C44E52", "#8172B2"], width=0.6)
    ax.set_ylabel("Minority-Class Recall"); ax.set_title("Figure 3. Component Ablation under 30% ALIE")
    ax.set_ylim(0.85, 1.0)
    for i, v in enumerate(recalls):
        ax.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.text(0.5, -0.02, "Fixed manuscript reference values — not computed by this run", ha="center", fontsize=9)
    fig.savefig(os.path.join(save_dir, "fig3_ablation.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Figure 3 saved")

    # Figure 4
    fig, ax = plt.subplots(figsize=(10, 5))
    methods = ["FedAvg", "Coord. Median", "Trimmed Mean", "Krum", "TC-FedIDS"]
    latency = [48.47, 48.49, 48.46, 48.80, 49.93]
    overhead = [0.00, 0.04, -0.02, 0.68, 2.97]
    ax.bar(methods, latency, color=["#4C72B0"]*4 + ["#8172B2"])
    ax.set_ylabel("Average Latency (s/round)"); ax.set_title("Figure 4. Computational Cost and Latency")
    ax.set_ylim(47.5, 51)
    for i, (v, o) in enumerate(zip(latency, overhead)):
        ax.text(i, v + 0.15, f"{v:.2f}s\n({o:+.2f}%)", ha="center", fontsize=9)
    ax.axhline(y=48.47, color="gray", linestyle="--", alpha=0.6, label="FedAvg baseline")
    ax.legend()
    plt.tight_layout()
    fig.text(0.5, -0.02, "Fixed manuscript reference values — not computed by this run", ha="center", fontsize=9)
    fig.savefig(os.path.join(save_dir, "fig4_latency.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Figure 4 saved")

    # Figure 5
    fig, ax = plt.subplots(figsize=(10, 5))
    eps_labels = ["ε=∞", "ε=8.0", "ε=4.0", "ε=2.0"]
    bacc_dp = [0.8987, 0.5304, 0.5000, 0.5000]
    f1_dp = [0.8773, 0.3296, 0.2669, 0.2669]
    recall_dp = [0.9629, 0.9947, 1.0000, 1.0000]
    x_dp = np.arange(len(eps_labels)); width_dp = 0.25
    ax.bar(x_dp - width_dp, bacc_dp, width_dp, label="Balanced Accuracy", color="#4C72B0")
    ax.bar(x_dp, f1_dp, width_dp, label="Macro-F1", color="#55A868")
    ax.bar(x_dp + width_dp, recall_dp, width_dp, label="Minority Recall", color="#C44E52")
    ax.set_ylabel("Score"); ax.set_title("Figure 5. Privacy-Utility Trade-off under DP")
    ax.set_xticks(x_dp); ax.set_xticklabels(eps_labels); ax.legend(); ax.set_ylim(0, 1.15)
    ax.axvspan(1.5, 3.5, alpha=0.15, color="red")
    ax.text(2.5, 1.08, "Model Collapse\n(ε ≤ 4.0)", ha="center", color="darkred", fontsize=10)
    plt.tight_layout()
    fig.text(0.5, -0.02, "Fixed manuscript reference values — not computed by this run", ha="center", fontsize=9)
    fig.savefig(os.path.join(save_dir, "fig5_dp_tradeoff.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Figure 5 saved")

    # Figure 6
    fig, ax = plt.subplots(figsize=(10, 5))
    beta = [0.00, 0.15, 0.25, 0.35, 0.40]
    recall_radius = [0.9133, 0.9541, 0.9524, 0.9680, 0.9743]
    ax.plot(beta, recall_radius, marker="o", linewidth=2.5, markersize=10, color="#8172B2", label="TC-FedIDS Minority Recall")
    ax.fill_between(beta, recall_radius, alpha=0.2, color="#8172B2")
    ax.axhline(y=0.50, color="red", linestyle="--", linewidth=1.5, label="MinRec ≥ 0.50 Threshold")
    for b, r in zip(beta, recall_radius):
        ax.text(b, r + 0.008, f"{r:.4f}", ha="center", fontsize=9)
    ax.set_xlabel("Byzantine Fraction β"); ax.set_ylabel("Mean Minority-Class Recall")
    ax.set_title("Figure 6. Empirical Byzantine Robustness Radius")
    ax.set_ylim(0.4, 1.05); ax.set_xlim(-0.02, 0.45); ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.text(0.5, -0.02, "Fixed manuscript reference values — not computed by this run", ha="center", fontsize=9)
    fig.savefig(os.path.join(save_dir, "fig6_robustness_radius.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Figure 6 saved")

    # Figure 7
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    attacks = ["ALIE", "Sign-Flip", "Gaussian"]
    bacc_atk = [0.8987, 0.8500, 0.6941]
    f1_atk = [0.8773, 0.8428, 0.6842]
    recall_atk = [0.9629, 0.8437, 0.5448]
    sigma = [0.0104, 0.0404, 0.3138]
    x_atk = np.arange(len(attacks)); width_atk = 0.25
    axes[0].bar(x_atk - width_atk, bacc_atk, width_atk, label="Balanced Acc", color="#4C72B0")
    axes[0].bar(x_atk, f1_atk, width_atk, label="Macro-F1", color="#55A868")
    axes[0].bar(x_atk + width_atk, recall_atk, width_atk, label="Minority Recall", color="#C44E52")
    axes[0].set_ylabel("Score"); axes[0].set_title("(a) Performance Metrics by Attack Type")
    axes[0].set_xticks(x_atk); axes[0].set_xticklabels(attacks); axes[0].legend(); axes[0].set_ylim(0, 1.1)
    axes[1].bar(attacks, sigma, color=["#55A868", "#DD8452", "#C44E52"])
    axes[1].set_ylabel("Recall Standard Deviation (σ)"); axes[1].set_title("(b) Recall Variance by Attack Type")
    for i, v in enumerate(sigma):
        axes[1].text(i, v + 0.01, f"σ={v:.4f}", ha="center", fontsize=10)
    plt.suptitle("Figure 7. Attack-Type Robustness Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.text(0.5, -0.02, "Fixed manuscript reference values — not computed by this run", ha="center", fontsize=9)
    fig.savefig(os.path.join(save_dir, "fig7_attack_types.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Figure 7 saved")

    print("\nAll 7 figures saved as PNG.")
    print("=" * 80)


# =====================================================================
# Command-line entry point
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TC-FedIDS synthetic architecture validation; no parquet training.")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-manuscript", action="store_true", help="Skip fixed reference CSV/PNG exports.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "manuscript_reference")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    main(rounds=args.rounds, seed=args.seed, device_name=args.device)
    if not args.skip_manuscript:
        print_all_tables_and_figures(save_dir=str(args.output_dir))
