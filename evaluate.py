"""Evaluate baselines and trained PPO agents; write CSVs and figures to --results-dir."""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ev_charging import EVChargingEnv, EnvConfig
from ev_charging.baselines import BASELINES

KEYS = ["ret", "cost", "service_rate", "completion_rate", "mean_idle_steps", "unmet_kwh",
        "rejected", "arrived"]


def run_episode(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    done, info = False, {}
    while not done:
        obs, _, done, _, info = env.step(policy(env, obs))
    return {k: info[k] for k in KEYS}


def evaluate(policies, noise, episodes, seed0=10_000, **cfg_kwargs):
    rows = []
    for name, pol in policies.items():
        env = EVChargingEnv(EnvConfig(depart_noise_std=noise, **cfg_kwargs))
        for ep in range(episodes):
            rows.append(dict(method=name, noise=noise, seed=seed0 + ep,
                             **run_episode(env, pol, seed0 + ep)))
    return pd.DataFrame(rows)


def pool_ppo(df):
    """Relabel PPO-s0, PPO-s1, ... as one 'PPO' method (all seeds' episodes pooled)."""
    df = df.copy()
    df["method"] = df["method"].where(~df["method"].str.startswith("PPO-s"), "PPO")
    return df


def bar_panels(df, path):
    order = list(df.groupby("method")["ret"].mean().sort_values(ascending=False).index)
    metrics = [("cost", "Energy cost"), ("service_rate", "Cars fully charged / arrived"),
               ("rejected", "Rejected cars"), ("ret", "Episode return")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    for ax, (m, title) in zip(axes, metrics):
        g = df.groupby("method")[m].agg(["mean", "std"]).loc[order]
        ax.bar(range(len(g)), g["mean"], yerr=g["std"], capsize=2)
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(g.index, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def training_curve(prefix, path):
    files = glob.glob(os.path.join("logs", prefix + "*", "*.monitor.csv"))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(4, 2.6))
    runs = {}
    for f in files:
        d = pd.read_csv(f, skiprows=1)
        d["steps"] = d["l"].cumsum()
        runs.setdefault(os.path.dirname(f), []).append(d)
    for name, frames in runs.items():
        d = pd.concat(frames).sort_values("steps")
        d["bin"] = d["steps"] // 20000
        g = d.groupby("bin").agg(steps=("steps", "mean"), r=("r", "mean"))
        ax.plot(g["steps"], g["r"], lw=1, label=os.path.basename(name))
    ax.set_xlabel("env steps (per env)")
    ax.set_ylabel("training episode return")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["models/ppo_ev"],
                    help="PPO model paths (one per training seed)")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--noise-sweep", action="store_true")
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--alpha", type=float, default=EnvConfig.alpha)
    ap.add_argument("--arrival-scale", type=float, default=EnvConfig.arrival_scale)
    ap.add_argument("--price-spread", type=float, default=EnvConfig.price_spread)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--log-prefix", default="ppo_ev")
    args = ap.parse_args()
    out = args.results_dir
    os.makedirs(out, exist_ok=True)
    kw = dict(alpha=args.alpha, arrival_scale=args.arrival_scale,
              price_spread=args.price_spread)

    policies = dict(BASELINES)
    if not args.baselines_only:
        from stable_baselines3 import PPO
        for i, path in enumerate(args.models):
            model = PPO.load(path)
            policies[f"PPO-s{i}"] = (lambda m: lambda env, obs: m.predict(obs, deterministic=True)[0])(model)

    raw = evaluate(policies, noise=2.0, episodes=args.episodes, **kw)
    raw.to_csv(os.path.join(out, "main_per_seed.csv"), index=False)
    df = pool_ppo(raw)
    df.to_csv(os.path.join(out, "main.csv"), index=False)
    print(df.groupby("method")[KEYS].mean().round(3).sort_values("ret", ascending=False).to_string())
    ppo = raw[raw["method"].str.startswith("PPO-s")]
    if len(args.models) > 1 and not ppo.empty:
        per = ppo.groupby("method")[["ret", "cost", "service_rate"]].mean().round(3)
        print("\nPer-seed PPO means:\n" + per.to_string())
        print(f"PPO return across seeds: {per['ret'].mean():.1f} +/- {per['ret'].std():.1f}")
    bar_panels(df, os.path.join(out, "main_comparison.png"))
    training_curve(args.log_prefix, os.path.join(out, "training_curve.png"))

    if args.noise_sweep:
        sweep = pd.concat([evaluate(policies, n, max(args.episodes // 2, 20), **kw)
                           for n in (0, 1, 2, 4, 6)])
        sweep = pool_ppo(sweep)
        sweep.to_csv(os.path.join(out, "noise_sweep.csv"), index=False)
        fig, ax = plt.subplots(figsize=(4, 2.6))
        for name, g in sweep.groupby("method"):
            m = g.groupby("noise")["service_rate"].mean()
            ax.plot(m.index, m.values, marker="o", label=name)
        ax.set_xlabel("departure noise std (steps)")
        ax.set_ylabel("cars fully charged / arrived")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "noise_robustness.png"), dpi=200)


if __name__ == "__main__":
    main()
