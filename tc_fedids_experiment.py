#!/usr/bin/env python3
"""TC-FedIDS real-data federation benchmark (new, explicitly defined protocol).

Uses all four environments per federation; independent trials are sharded.
The ResMLP/GroupNorm, four-feature trajectory trust and Huber GroupDRO-FISTA
components are retained. All sources share five explicitly harmonized inputs.
No hard-coded manuscript results, synthetic training fallback or DP claims.
See TC_FedIDS_RUN_GUIDE.md for the feature/split/label protocol and commands.
Requires Python 3.9+, numpy, pandas, pyarrow, torch; plots need matplotlib.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

ROOT = Path(__file__).resolve().parent
ENVIRONMENTS = ("NF-CSE-CIC-IDS2018", "NF-ToN-IoT", "UNSW-NB15", "CIDDS-001")
FEATURES = ("duration_seconds", "total_bytes", "total_packets", "mean_packet_bytes", "ip_protocol")
VERSION = "tc-real-common5-v1"

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
    Default benchmark uses five harmonized inputs, width 128 and two classes.
    """

    def __init__(self, in_features: int = 5, num_classes: int = 2, hidden_dim: int = 128):
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
        use_trajectory: bool = True,
    ):
        if history_len < 1 or not 0 <= alpha_ema <= 1 or not 0 <= trust_floor <= 1 or gamma < 0:
            raise ValueError("Invalid history length, EMA, trust floor or gamma.")
        self.use_trajectory = use_trajectory
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
            if not self.use_trajectory:
                r_composite = (r_composite - 0.28 * r_traj) / 0.72
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



# Real data preparation: bounded, uniform random sampling over all eligible rows.
SOURCE_FILES = {
    ENVIRONMENTS[0]: ["NF-CSE-CIC-IDS2018.parquet"],
    ENVIRONMENTS[1]: ["NF-ToN-IoT.parquet"],
    ENVIRONMENTS[2]: ["UNSW_NB15_training-set.parquet", "UNSW_NB15_testing-set.parquet"],
    ENVIRONMENTS[3]: ["cidds-001-openstack.parquet", "cidds-001-externalserver.parquet"],
}
PROTOCOL_CODES = {"icmp": 1, "igmp": 2, "tcp": 6, "udp": 17, "ipv6": 41,
                  "gre": 47, "esp": 50, "ah": 51, "ipv6-icmp": 58,
                  "icmp6": 58, "ospf": 89, "sctp": 132}


def atomic_json(path, value):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


@contextmanager
def directory_lock(path):
    """Atomic exclusive lock. Stale locks require explicit manual review/removal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Already locked: {path}. Check whether the previous process is still running.") from exc
    try:
        os.write(fd, f"pid={os.getpid()}\n".encode())
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


def source_signature(data_dir):
    files = []
    for names in SOURCE_FILES.values():
        for name in names:
            path = data_dir / name
            stat = path.stat()  # Missing sources fail before any experiment starts.
            files.append({"name": name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return files


def harmonize(frame, environment):
    """No label/attack metadata enters X. CIDDS ambiguous labels are excluded."""
    import pandas as pd
    if environment.startswith("NF-"):
        duration = frame["FLOW_DURATION_MILLISECONDS"].astype(float).to_numpy() / 1000.0
        byte_count = frame["IN_BYTES"].astype(float).to_numpy() + frame["OUT_BYTES"].astype(float).to_numpy()
        packets = frame["IN_PKTS"].astype(float).to_numpy() + frame["OUT_PKTS"].astype(float).to_numpy()
        proto = frame["PROTOCOL"].astype(float).to_numpy()
        labels = pd.to_numeric(frame["Label"], errors="coerce").to_numpy(dtype=float)
    elif environment == "UNSW-NB15":
        duration = frame["dur"].astype(float).to_numpy()
        byte_count = frame["sbytes"].astype(float).to_numpy() + frame["dbytes"].astype(float).to_numpy()
        packets = frame["spkts"].astype(float).to_numpy() + frame["dpkts"].astype(float).to_numpy()
        proto = frame["proto"].astype("string").str.strip().str.lower().map(PROTOCOL_CODES).fillna(0).to_numpy(dtype=float)
        labels = pd.to_numeric(frame["label"], errors="coerce").to_numpy(dtype=float)
    else:
        duration = frame["duration"].astype(float).to_numpy()
        byte_count = frame["bytes"].astype(float).to_numpy()
        packets = frame["packets"].astype(float).to_numpy()
        proto = frame["proto"].astype("string").str.strip().str.lower().map(PROTOCOL_CODES).fillna(0).to_numpy(dtype=float)
        labels = frame["label"].astype("string").str.strip().str.lower().map({"normal": 0, "attacker": 1, "victim": 1}).to_numpy(dtype=float, na_value=np.nan)
    mean_bytes = np.divide(byte_count, packets, out=np.zeros_like(byte_count), where=packets > 0)
    x = np.column_stack((duration, byte_count, packets, mean_bytes, proto))
    valid_label = np.isin(labels, [0, 1])
    valid_features = np.isfinite(x).all(axis=1) & (x >= 0).all(axis=1)
    keep = valid_label & valid_features
    counts = {"read": len(frame), "excluded_label": int((~valid_label).sum()),
              "excluded_feature": int((valid_label & ~valid_features).sum())}
    return x[keep].astype(np.float32), labels[keep].astype(np.int64), counts


def read_sample(paths, environment, cap, rng):
    import pyarrow.parquet as pq
    if environment.startswith("NF-"):
        columns = ["FLOW_DURATION_MILLISECONDS", "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "PROTOCOL", "Label"]
    elif environment == "UNSW-NB15":
        columns = ["dur", "sbytes", "dbytes", "spkts", "dpkts", "proto", "label"]
    else:
        columns = ["duration", "bytes", "packets", "proto", "label"]
    xs = np.empty((0, 5), dtype=np.float32)
    ys = np.empty(0, dtype=np.int64)
    priorities = np.empty(0, dtype=float)
    counters = Counter()
    for path in paths:
        print(f"[PREPARE] Scanning {path.name}", flush=True)
        parquet = pq.ParquetFile(path)
        absent = set(columns) - set(parquet.schema_arrow.names)
        if absent:
            raise ValueError(f"{path.name}: missing columns {sorted(absent)}")
        for batch in parquet.iter_batches(batch_size=65536, columns=columns):
            x, y, counts = harmonize(batch.to_pandas(), environment)
            counters.update(counts)
            counters["eligible_benign"] += int((y == 0).sum())
            counters["eligible_attack"] += int((y == 1).sum())
            keys = rng.random(len(y))
            if len(priorities) == cap:
                accept = keys < priorities.max()
                x, y, keys = x[accept], y[accept], keys[accept]
            xs = np.concatenate((xs, x))
            ys = np.concatenate((ys, y))
            priorities = np.concatenate((priorities, keys))
            if len(ys) > cap:
                selected = np.argpartition(priorities, cap - 1)[:cap]
                xs, ys, priorities = xs[selected], ys[selected], priorities[selected]
    order = rng.permutation(len(ys))
    counters["sampled"] = len(ys)
    counters["sampled_attack"] = int(ys.sum())
    return xs[order], ys[order], dict(counters)


def stratified_split(y, seed, proportions):
    rng = np.random.default_rng(seed)
    groups = [[] for _ in proportions]
    for label in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == label))
        if len(idx) < 10:
            raise ValueError(f"Only {len(idx)} sampled rows for label={label}; need at least 10. Increase --sample-cap.")
        cuts = [int(len(idx) * p) for p in proportions[:-1]]
        chunks = np.split(idx, np.cumsum(cuts))
        for group, chunk in zip(groups, chunks):
            group.extend(chunk.tolist())
    return [rng.permutation(np.asarray(group, dtype=np.int64)) for group in groups]


def transformed(x):
    z = x.astype(np.float64)
    z[:, :4] = np.log1p(z[:, :4])
    z[:, 4] /= 255.0
    return z


def prepare(args):
    data_dir = args.data_dir.resolve()
    signature = source_signature(data_dir)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    with directory_lock(args.cache_dir / ".prepare.lock"):
        manifest_path = args.cache_dir / "manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous["source_files"] == signature and previous["sample_cap"] == args.sample_cap and previous["split_seed"] == args.split_seed and previous["version"] == VERSION:
                print("Prepared cache matches sources and protocol; reusing it.", flush=True)
                validate_cache(args.cache_dir)
                return
            raise RuntimeError("Cache belongs to different data/settings. Use a new --cache-dir.")
        splits, stats = {}, {}
        rng = np.random.default_rng(args.split_seed)
        for env_i, env in enumerate(ENVIRONMENTS):
            paths = [data_dir / name for name in SOURCE_FILES[env]]
            if env == "UNSW-NB15":
                # Official test file is never included in training or scaling.
                x, y, train_stats = read_sample(paths[:1], env, args.sample_cap, rng)
                tx, ty, test_stats = read_sample(paths[1:], env, args.sample_cap, rng)
                train, val = stratified_split(y, args.split_seed + env_i, (0.8, 0.2))
                if min(np.bincount(ty, minlength=2)) < 2:
                    raise ValueError("UNSW held-out sample must contain both classes.")
                parts = {"train": (x[train], y[train]), "val": (x[val], y[val]), "test": (tx, ty)}
                stats[env] = {"training_source": train_stats, "testing_source": test_stats, "split": "official test; 80/20 training/validation"}
            else:
                x, y, counts = read_sample(paths, env, args.sample_cap, rng)
                train, val, test = stratified_split(y, args.split_seed + env_i, (0.6, 0.2, 0.2))
                parts = {name: (x[ix], y[ix]) for name, ix in zip(("train", "val", "test"), (train, val, test))}
                stats[env] = {"source": counts, "split": "stratified random row split 60/20/20"}
            splits[env] = parts
        training_x = np.concatenate([transformed(splits[e]["train"][0]) for e in ENVIRONMENTS])
        means = training_x.mean(axis=0)
        scales = training_x.std(axis=0)
        scales[scales < 1e-8] = 1.0
        for env_i, env in enumerate(ENVIRONMENTS):
            stats[env]["partitions"] = {}
            for part, (x, y) in splits[env].items():
                z = ((transformed(x) - means) / scales).astype(np.float32)
                if not np.isfinite(z).all():
                    raise ValueError(f"Non-finite standardized values in {env}/{part}")
                np.save(args.cache_dir / f"env{env_i}_{part}_x.npy", z, allow_pickle=False)
                np.save(args.cache_dir / f"env{env_i}_{part}_y.npy", y, allow_pickle=False)
                stats[env]["partitions"][part] = {"rows": len(y), "benign": int((y == 0).sum()), "attack": int(y.sum())}
        manifest = {"version": VERSION, "source_files": signature, "source_directory": str(data_dir),
                    "features": FEATURES, "sample_cap": args.sample_cap, "split_seed": args.split_seed,
                    "cidds_policy": "normal=0; attacker/victim=1; suspicious/unknown excluded",
                    "normalization": {"mean": means.tolist(), "std": scales.tolist(), "log1p_columns": list(FEATURES[:4])},
                    "environments": stats, "protocol_unknown": 0,
                    "limitations": ["Random row splits for NF and combined CIDDS; not a temporal or host-disjoint evaluation.",
                                     "The schema duration fields are interpreted as seconds for UNSW/CIDDS and milliseconds for NF.",
                                     "This common-five-input protocol is new; it is not a reconstruction of the manuscript Harm5 pipeline."]}
        files = sorted(args.cache_dir.glob("env*_*.npy"))
        manifest["cache_sha256"] = {p.name: file_hash(p) for p in files}
        atomic_json(manifest_path, manifest)
        validate_cache(args.cache_dir)
        print("All four environments prepared; train-only scaling and held-out evaluation splits saved.", flush=True)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_cache(cache_dir):
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest["version"] != VERSION:
        raise ValueError("Incompatible cache protocol; prepare a new cache.")
    for name, digest in manifest["cache_sha256"].items():
        if file_hash(cache_dir / name) != digest:
            raise ValueError(f"Cache content changed: {name}. Prepare a new cache.")
    return manifest


def load_data(cache_dir):
    manifest = validate_cache(cache_dir)
    data = []
    for i, env in enumerate(ENVIRONMENTS):
        parts = {}
        for part in ("train", "val", "test"):
            x = np.load(cache_dir / f"env{i}_{part}_x.npy", mmap_mode="r", allow_pickle=False)
            y = np.load(cache_dir / f"env{i}_{part}_y.npy", mmap_mode="r", allow_pickle=False)
            if x.shape != (len(y), 5) or not np.isfinite(x).all() or not np.isin(y, (0, 1)).all():
                raise ValueError(f"Invalid prepared data: {env}/{part}")
            parts[part] = (x, y)
        parts["minority"] = int(np.argmin(np.bincount(parts["train"][1], minlength=2)))
        data.append(parts)
    return data, manifest


# Trial planning. Full suite includes measured ablations and a TC ALIE ratio sweep.
CORE_METHODS = ("fedavg", "median", "trimmed", "krum", "tc")
FULL_METHODS = CORE_METHODS + ("tc_no_trajectory", "tc_no_groupdro", "tc_hard_median")


def make_plan(args):
    methods = args.methods or list(CORE_METHODS if args.suite == "core" else FULL_METHODS)
    attacks = args.attacks or (["none", "alie"] if args.suite == "core" else ["none", "alie", "signflip", "gaussian"])
    jobs = []
    for seed in args.seeds:
        for method in methods:
            for attack in attacks:
                jobs.append({"seed": seed, "method": method, "attack": attack,
                             "beta": 0.0 if attack == "none" else args.byzantine_ratio})
        if args.suite == "full" and "tc" in methods:
            for beta in (0.15, 0.25, 0.35, 0.40):
                candidate = {"seed": seed, "method": "tc", "attack": "alie", "beta": beta}
                if candidate not in jobs:
                    jobs.append(candidate)
    for job in jobs:
        job["id"] = f"{job['method']}__{job['attack']}__b{job['beta']:.6f}__s{job['seed']}"
    return jobs


def config_for(args, manifest):
    keys = ("rounds", "local_epochs", "batch_size", "learning_rate", "clients_per_env", "dirichlet_alpha",
            "byzantine_ratio", "alie_z", "signflip_scale", "gaussian_scale", "server_lr", "threads", "shards")
    config = {key: getattr(args, key) for key in keys}
    config["version"] = VERSION
    config["cache_manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    config["program_sha256"] = file_hash(Path(__file__))
    config["jobs"] = make_plan(args)
    config["privacy"] = "No differential privacy; no noise/accountant/epsilon sweep."
    config["partition"] = "Seven clients per environment by default; within-environment class Dirichlet allocation, minimum nonempty size."
    config["local_objective"] = "Unweighted cross-entropy, Adam freshly initialized each local round; norm-5 batch-gradient clipping."
    config["validation"] = "Server-held-out validation balanced accuracy per client environment; test used only for reporting."
    config["timing"] = "Observed durations under this execution environment; parallel GPU contention affects them."
    return config


def partition_clients(data, seed, per_env, alpha):
    clients = []
    rng = np.random.default_rng(seed)
    for env_i, parts in enumerate(data):
        x, y = parts["train"]
        minimum = max(2, min(32, len(y) // (2 * per_env)))
        if len(y) < 2 * per_env:
            raise ValueError("Not enough training rows for the requested clients.")
        for attempt in range(1000):
            groups = [[] for _ in range(per_env)]
            for label in (0, 1):
                idx = rng.permutation(np.flatnonzero(y == label))
                weights = rng.dirichlet(np.full(per_env, alpha))
                sizes = rng.multinomial(len(idx), weights)
                for group, chunk in zip(groups, np.split(idx, np.cumsum(sizes)[:-1])):
                    group.extend(chunk.tolist())
            if min(map(len, groups)) >= minimum:
                break
        else:
            raise ValueError("Unable to create nonempty Dirichlet clients; increase --dirichlet-alpha.")
        for ids in groups:
            ids = np.asarray(ids, dtype=np.int64)
            clients.append((env_i, np.ascontiguousarray(x[ids]), np.ascontiguousarray(y[ids])))
    return clients


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def vector(model):
    return parameters_to_vector(model.parameters()).detach().clone()


def assign_vector(model, weights):
    # vector_to_parameters changes parameter storage; always pass a private clone.
    with torch.no_grad():
        vector_to_parameters(weights.detach().clone(), model.parameters())


@torch.no_grad()
def confusion(model, x, y, device, batch_size=4096):
    model.eval()
    result = np.zeros((2, 2), dtype=np.int64)
    for start in range(0, len(y), batch_size):
        batch = torch.from_numpy(np.array(x[start:start + batch_size], copy=True)).to(device)
        logits = model(batch)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite evaluation logits")
        prediction = logits.argmax(dim=1).cpu().numpy()
        target = np.asarray(y[start:start + batch_size])
        result += np.bincount(2 * target + prediction, minlength=4).reshape(2, 2)
    return result


def metrics(cm, minority):
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    recalls = [tn / max(1, tn + fp), tp / max(1, tp + fn)]
    f1s = [2 * tn / max(1, 2 * tn + fp + fn), 2 * tp / max(1, 2 * tp + fp + fn)]
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"balanced_accuracy": sum(recalls) / 2, "macro_f1": sum(f1s) / 2,
            "attack_recall": recalls[1], "minority_recall": recalls[minority],
            "minority_label": minority, "mcc": (tp * tn - fp * fn) / den if den else 0.0,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp, "test_rows": int(cm.sum())}


def train_client(local_model, base, x, y, device, args, seed):
    # The same client/round RNG seed is used across aggregators and conditions.
    set_deterministic_seed(seed)
    assign_vector(local_model, base)
    local_model.train()
    optimizer = torch.optim.Adam(local_model.parameters(), lr=args.learning_rate)
    tx = torch.from_numpy(x).to(device)
    ty = torch.from_numpy(y).to(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loss_sum, batches = 0.0, 0
    for _ in range(args.local_epochs):
        order = torch.randperm(len(y), generator=generator)
        for start in range(0, len(y), args.batch_size):
            ids = order[start:start + args.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(local_model(tx[ids]), ty[ids])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite local training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(local_model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.detach().item()
            batches += 1
    delta = vector(local_model) - base
    if not torch.isfinite(delta).all():
        raise FloatingPointError("Non-finite local update")
    return delta, loss_sum / batches


def attacked_updates(updates, bad_ids, job, args, seed):
    if not bad_ids:
        return updates
    honest = torch.stack([u for i, u in enumerate(updates) if i not in bad_ids])
    if not len(honest):
        raise ValueError("At least one honest client is required")
    mu = honest.mean(dim=0)
    std = honest.std(dim=0, unbiased=len(honest) > 1)
    poisoned = [u.clone() for u in updates]
    if job["attack"] == "alie":
        attack = mu - args.alie_z * (std + 1e-8)
        for i in bad_ids:
            poisoned[i] = attack.clone()
    elif job["attack"] == "signflip":
        for i in bad_ids:
            poisoned[i] = -args.signflip_scale * updates[i]
    elif job["attack"] == "gaussian":
        generator = torch.Generator(device=mu.device).manual_seed(seed)
        scale = honest.square().mean().sqrt().clamp_min(1e-8) * args.gaussian_scale
        for i in bad_ids:
            poisoned[i] = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype) * scale
    else:
        raise ValueError(f"Unsupported attack: {job['attack']}")
    return poisoned


def aggregate_updates(updates, method, sample_sizes, bad_count, trust, env_labels, aggregator):
    u = torch.stack(updates)
    if method == "fedavg":
        weights = sample_sizes / sample_sizes.sum()
        return (u * weights[:, None]).sum(dim=0)
    if method == "median":
        return u.median(dim=0).values
    if method == "trimmed":
        ordered = u.sort(dim=0).values
        # Oracle attack-count trimming, disclosed in the protocol.
        if 2 * bad_count >= len(updates):
            raise ValueError("Trimmed mean needs n > 2*f")
        return ordered[bad_count:len(updates) - bad_count].mean(dim=0)
    if method == "krum":
        n = len(updates)
        if n <= 2 * bad_count + 2:
            raise ValueError("Krum requires n > 2*f+2")
        distances = torch.cdist(u, u).square()
        distances.fill_diagonal_(torch.inf)
        scores = distances.topk(n - bad_count - 2, largest=False, dim=1).values.sum(dim=1)
        return u[scores.argmin()]
    if method == "tc_no_groupdro":
        return (u * (trust / trust.sum())[:, None]).sum(dim=0)
    if method == "tc_hard_median":
        # Explicit new ablation threshold; does not claim to recreate manuscript TC-Median.
        keep = trust >= 0.55
        if not keep.any():
            return u.median(dim=0).values
        return u[keep].median(dim=0).values
    return aggregator.aggregate(updates, trust, env_labels)


def write_rows(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty result table")
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def run_trial(job, data, args, device, trial_dir, config_hash):
    set_deterministic_seed(job["seed"])
    clients = partition_clients(data, job["seed"], args.clients_per_env, args.dirichlet_alpha)
    n = len(clients)
    bad_count = int(n * job["beta"])
    bad_ids = set(np.random.default_rng(job["seed"] + 100003).permutation(n)[:bad_count].tolist())
    env_labels = torch.tensor([c[0] for c in clients], device=device)
    sizes = torch.tensor([len(c[2]) for c in clients], device=device, dtype=torch.float32)
    global_model = ResMLP().to(device)
    local_model = ResMLP().to(device)
    validation_model = ResMLP().to(device)
    engine = TrajectoryTrustEngine(use_trajectory=job["method"] != "tc_no_trajectory")
    aggregator = AdaptiveGroupDROFISTAAggregator()
    parameters = sum(p.numel() for p in global_model.parameters())
    client_manifest = [{"id": i, "environment": ENVIRONMENTS[e], "rows": len(y),
                        "attack_rows": int(y.sum()), "byzantine": i in bad_ids}
                       for i, (e, _, y) in enumerate(clients)]
    atomic_json(trial_dir / "clients.json", client_manifest)
    records = []
    synchronize(device)
    trial_start = time.perf_counter()
    for round_no in range(1, args.rounds + 1):
        synchronize(device)
        start = time.perf_counter()
        base = vector(global_model)
        updates, losses = [], []
        for i, (_, x, y) in enumerate(clients):
            local_seed = (job["seed"] * 1000003 + round_no * 1009 + i * 97) % (2 ** 32 - 1)
            delta, loss = train_client(local_model, base, x, y, device, args, local_seed)
            updates.append(delta)
            losses.append(loss)
        updates = attacked_updates(updates, bad_ids, job, args, job["seed"] + round_no * 37)
        trust = torch.ones(n, device=device)
        if job["method"].startswith("tc"):
            val_scores = []
            for i, update in enumerate(updates):
                env_i = clients[i][0]
                assign_vector(validation_model, base + update)
                x, y = data[env_i]["val"]
                cm = confusion(validation_model, x, y, device)
                val_scores.append(metrics(cm, data[env_i]["minority"])["balanced_accuracy"])
            trust = engine.compute_trust(updates, env_labels, val_scores)
        synchronize(device)
        aggregation_start = time.perf_counter()
        with torch.no_grad():
            delta = aggregate_updates(updates, job["method"], sizes, bad_count, trust, env_labels, aggregator)
            if not torch.isfinite(delta).all():
                raise FloatingPointError("Non-finite aggregate update")
            assign_vector(global_model, base + args.server_lr * delta)
        synchronize(device)
        aggregation_seconds = time.perf_counter() - aggregation_start
        train_seconds = time.perf_counter() - start
        confusion_by_env = [confusion(global_model, *parts["test"], device) for parts in data]
        synchronize(device)
        round_seconds = time.perf_counter() - start
        honest_ids = [i for i in range(n) if i not in bad_ids]
        honest_trust = trust[honest_ids].mean().item()
        bad_trust = trust[sorted(bad_ids)].mean().item() if bad_ids else None
        # Report individual environments and an equal-environment mean.
        metrics_by_env = [metrics(cm, parts["minority"]) for cm, parts in zip(confusion_by_env, data)]
        for env_i, env in enumerate(ENVIRONMENTS):
            row = {"trial": job["id"], "seed": job["seed"], "method": job["method"], "attack": job["attack"],
                   "requested_beta": job["beta"], "actual_beta": bad_count / n, "round": round_no,
                   "environment": env, **metrics_by_env[env_i], "honest_trust": honest_trust,
                   "byzantine_trust": bad_trust, "mean_local_loss": float(np.mean(losses)),
                   "aggregation_seconds": aggregation_seconds, "training_and_validation_seconds": train_seconds,
                   "round_seconds": round_seconds, "parameters": parameters}
            records.append(row)
        global_balanced = np.mean([m["balanced_accuracy"] for m in metrics_by_env])
        print(f"[{job['id']}] round {round_no}/{args.rounds} | equal-env BalAcc={global_balanced:.4f} | {round_seconds:.2f}s", flush=True)
        # Save an atomic partial result snapshot; incomplete trials never enter analysis.
        write_rows(trial_dir / "rounds.csv", records)
    synchronize(device)
    seconds = time.perf_counter() - trial_start
    checkpoint = trial_dir / f"model.{os.getpid()}.tmp"
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in global_model.state_dict().items()},
                "job": job, "config_hash": config_hash, "input_features": FEATURES}, checkpoint)
    os.replace(checkpoint, trial_dir / "model.pt")
    atomic_json(trial_dir / "complete.json", {"job": job, "config_hash": config_hash,
                "rounds": args.rounds, "wall_seconds": seconds, "bad_count": bad_count,
                "client_count": n, "device": str(device), "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "rounds_sha256": file_hash(trial_dir / "rounds.csv"),
                "model_sha256": file_hash(trial_dir / "model.pt")})


def run_worker(args):
    data, manifest = load_data(args.cache_dir)
    config = config_for(args, manifest)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Bootstrap from independent processes uses an exclusive lock with bounded retry.
    for attempt in range(100):
        try:
            with directory_lock(args.output_dir / ".config.lock"):
                path = args.output_dir / "config.json"
                if path.exists():
                    if json.loads(path.read_text()) != config:
                        raise ValueError("Output directory has different experiment settings. Use a new --output-dir.")
                else:
                    atomic_json(path, config)
            break
        except RuntimeError as exc:
            if "Already locked:" not in str(exc) or attempt == 99:
                raise
            time.sleep(0.1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    own_jobs = [job for i, job in enumerate(config["jobs"]) if i % args.shards == args.shard]
    worker_dir = args.output_dir / f"worker{args.shard}"
    worker_dir.mkdir(exist_ok=True)
    with directory_lock(worker_dir / ".running.lock"):
        (worker_dir / "complete.json").unlink(missing_ok=True)
        print(f"Worker {args.shard}/{args.shards}: {len(own_jobs)} trials on {device}; all four environments per trial.", flush=True)
        for job in own_jobs:
            trial_dir = worker_dir / job["id"]
            trial_dir.mkdir(exist_ok=True)
            marker = trial_dir / "complete.json"
            if marker.exists():
                previous = json.loads(marker.read_text())
                if previous["config_hash"] != config_hash:
                    raise ValueError("Completed trial has an incompatible config")
                if file_hash(trial_dir / "rounds.csv") != previous["rounds_sha256"] or file_hash(trial_dir / "model.pt") != previous["model_sha256"]:
                    raise ValueError(f"Completed trial files changed: {trial_dir}")
                print(f"[RESUME] Already complete: {job['id']}", flush=True)
                continue
            # A stopped partial trial is restarted from round 1 with the same seed.
            run_trial(job, data, args, device, trial_dir, config_hash)
        atomic_json(worker_dir / "complete.json", {"config_hash": config_hash, "trials": len(own_jobs)})
        print(f"Worker {args.shard} completed all {len(own_jobs)} trials.", flush=True)


def analyze(args):
    import pandas as pd
    config = json.loads((args.output_dir / "config.json").read_text())
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    expected = {job["id"] for job in config["jobs"]}
    completed, frames, times = set(), [], []
    for marker in sorted(args.output_dir.glob("worker*/*/complete.json")):
        value = json.loads(marker.read_text())
        trial = value["job"]["id"]
        if trial not in expected or value["config_hash"] != config_hash or trial in completed:
            raise ValueError(f"Unexpected/incompatible/duplicate trial: {trial}")
        path = marker.parent / "rounds.csv"
        if file_hash(path) != value["rounds_sha256"]:
            raise ValueError(f"Result contents changed: {path}")
        frame = pd.read_csv(path)
        if len(frame) != config["rounds"] * 4 or frame.duplicated(["round", "environment"]).any():
            raise ValueError(f"Missing or duplicate evaluation rows: {path}")
        if set(frame["round"]) != set(range(1, config["rounds"] + 1)) or set(frame["environment"]) != set(ENVIRONMENTS):
            raise ValueError(f"Invalid round/environment coverage: {path}")
        if set(frame["trial"]) != {trial}:
            raise ValueError(f"Trial identifier mismatch: {path}")
        completed.add(trial)
        frames.append(frame)
        times.append({"trial": trial, "wall_seconds": value["wall_seconds"], "device": value["device"]})
    missing = sorted(expected - completed)
    if missing:
        atomic_json(args.output_dir / "analysis_pending.json", {"complete_trials": len(completed), "expected_trials": len(expected), "missing_trials": missing})
        raise RuntimeError(f"Analysis refused: {len(missing)} of {len(expected)} trials incomplete. See analysis_pending.json.")
    all_rows = pd.concat(frames, ignore_index=True)
    final = all_rows[all_rows["round"] == config["rounds"]].copy()
    metrics_names = ["balanced_accuracy", "macro_f1", "minority_recall", "attack_recall", "mcc"]
    keys = ["method", "attack", "requested_beta", "environment"]
    summary = final.groupby(keys)[metrics_names].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join(str(x) for x in col if x != "") if isinstance(col, tuple) else col for col in summary.columns]
    # Across-seed SD, not across rounds. One-seed SD remains undefined (blank).
    # Equal-environment summary is first calculated per seed to retain paired runs.
    equal = final.groupby(["trial", "seed", "method", "attack", "requested_beta"], as_index=False)[metrics_names].mean()
    eq_summary = equal.groupby(["method", "attack", "requested_beta"])[metrics_names].agg(["mean", "std", "count"]).reset_index()
    eq_summary.columns = ["_".join(str(x) for x in col if x != "") if isinstance(col, tuple) else col for col in eq_summary.columns]
    analysis_dir = args.output_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    all_rows.to_csv(analysis_dir / "all_rounds.csv", index=False)
    final.to_csv(analysis_dir / "final_per_seed.csv", index=False)
    summary.to_csv(analysis_dir / "summary_by_environment.csv", index=False)
    eq_summary.to_csv(analysis_dir / "summary_equal_environment.csv", index=False)
    time_frame = pd.DataFrame(times)
    time_frame = time_frame.merge(final[["trial", "method", "attack", "requested_beta"]].drop_duplicates(), on="trial", validate="one_to_one")
    time_frame.to_csv(analysis_dir / "trial_timing.csv", index=False)
    # Reporting overhead aggregated once per trial/round, not once per environment.
    timing = all_rows.drop_duplicates(["trial", "round"]).groupby(["method", "attack", "requested_beta"])[["round_seconds", "training_and_validation_seconds", "aggregation_seconds"]].agg(["mean", "std"])
    timing.to_csv(analysis_dir / "timing_summary.csv")
    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for attack, subset in all_rows.groupby("attack"):
            # Plot the main configured ratio only; ratio sweeps have their own table.
            beta = 0.0 if attack == "none" else config["byzantine_ratio"]
            subset = subset[np.isclose(subset["requested_beta"], beta)]
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
            for ax, env in zip(axes.ravel(), ENVIRONMENTS):
                for method, df in subset[subset["environment"] == env].groupby("method"):
                    curve = df.groupby("round")["balanced_accuracy"].agg(["mean", "std"])
                    ax.plot(curve.index, curve["mean"], label=method)
                    sd = curve["std"].fillna(0)
                    ax.fill_between(curve.index, curve["mean"] - sd, curve["mean"] + sd, alpha=0.12)
                ax.set_title(env)
                ax.set_xlabel("Communication round")
                ax.set_ylabel("Test balanced accuracy")
                ax.set_ylim(0, 1)
            axes[0, 0].legend(fontsize=8)
            fig.suptitle(f"Measured real-data results: {attack}, beta={beta:g}; bands = seed SD")
            fig.tight_layout()
            fig.savefig(analysis_dir / f"balanced_accuracy_{attack}.png", dpi=160)
            plt.close(fig)
    atomic_json(analysis_dir / "complete.json", {"config_hash": config_hash, "completed_trials": len(completed), "expected_trials": len(expected), "source": "Measured real-data held-out evaluations only"})
    (args.output_dir / "analysis_pending.json").unlink(missing_ok=True)
    print(f"Complete: analyzed {len(completed)} trials and {len(all_rows)} measured environment/round rows. Output: {analysis_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="TC-FedIDS real parquet training across four environments. No fixed manuscript results.")
    parser.add_argument("mode", choices=("prepare", "run", "analyze", "plan"))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "Dataset")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "prepared_common5")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results_real")
    parser.add_argument("--sample-cap", type=int, default=50000, help="Uniform eligible-row cap per environment, or per official UNSW split.")
    parser.add_argument("--split-seed", type=int, default=11)
    parser.add_argument("--suite", choices=("core", "full"), default="core")
    parser.add_argument("--methods", nargs="+", choices=FULL_METHODS)
    parser.add_argument("--attacks", nargs="+", choices=("none", "alie", "signflip", "gaussian"))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(11, 21)))
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--server-lr", type=float, default=1.0)
    parser.add_argument("--clients-per-env", type=int, default=7)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--byzantine-ratio", type=float, default=0.30)
    parser.add_argument("--alie-z", type=float, default=1.5)
    parser.add_argument("--signflip-scale", type=float, default=1.0)
    parser.add_argument("--gaussian-scale", type=float, default=3.0, help="Gaussian standard deviation as multiplier of honest update RMS.")
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    for key in ("sample_cap", "rounds", "local_epochs", "batch_size", "clients_per_env", "shards", "threads"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    for key in ("learning_rate", "server_lr", "dirichlet_alpha"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")
    for key in ("alie_z", "signflip_scale", "gaussian_scale"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            parser.error(f"--{key.replace('_', '-')} must be finite and nonnegative")
    if not 0 <= args.byzantine_ratio < 0.5:
        parser.error("--byzantine-ratio must be in [0, 0.5)")
    if not 0 <= args.shard < args.shards:
        parser.error("--shard must be within [0, --shards)")
    if any(not 0 <= s < 2 ** 32 - 100004 for s in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct nonnegative integers below 2**32-100004")
    if args.methods and len(set(args.methods)) != len(args.methods) or args.attacks and len(set(args.attacks)) != len(args.attacks):
        parser.error("Methods/attacks must not repeat")
    if args.mode in ("run", "plan"):
        jobs = make_plan(args)
        if len({job["id"] for job in jobs}) != len(jobs):
            parser.error("Duplicate trial IDs")
        n = 4 * args.clients_per_env
        if any(job["method"] == "krum" and n <= 2 * int(n * job["beta"]) + 2 for job in jobs):
            parser.error("Configured Krum trials require n > 2*f+2")
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "run":
        run_worker(args)
    elif args.mode == "analyze":
        analyze(args)
    else:
        jobs = make_plan(args)
        print(json.dumps({"suite": args.suite, "trials": len(jobs), "rounds_per_trial": args.rounds,
                          "clients": 4 * args.clients_per_env, "environments_per_trial": ENVIRONMENTS,
                          "shard_counts": [sum(i % args.shards == j for i in range(len(jobs))) for j in range(args.shards)],
                          "jobs": jobs}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FAILED] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
