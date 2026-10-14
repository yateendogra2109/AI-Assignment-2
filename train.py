"""Train a PPO agent on the EV charging environment."""
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from ev_charging import EVChargingEnv, EnvConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--noise", type=float, default=2.0, help="departure-time noise std (steps)")
    ap.add_argument("--alpha", type=float, default=EnvConfig.alpha, help="penalty per unmet kWh")
    ap.add_argument("--arrival-scale", type=float, default=EnvConfig.arrival_scale)
    ap.add_argument("--price-spread", type=float, default=EnvConfig.price_spread)
    ap.add_argument("--out", default="models/ppo_ev")
    args = ap.parse_args()

    log_dir = os.path.join("logs", os.path.basename(args.out))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cfg = EnvConfig(depart_noise_std=args.noise, alpha=args.alpha,
                    arrival_scale=args.arrival_scale, price_spread=args.price_spread)
    venv = make_vec_env(lambda: EVChargingEnv(cfg), n_envs=args.n_envs, seed=args.seed,
                        monitor_dir=log_dir)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.99)

    model = PPO("MlpPolicy", venv, seed=args.seed, verbose=1, device="cpu",
                n_steps=96 * 8, batch_size=256, n_epochs=10, learning_rate=3e-4,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
                policy_kwargs=dict(net_arch=[128, 128]))
    ckpt = CheckpointCallback(save_freq=max(250_000 // args.n_envs, 1),
                              save_path="models/checkpoints",
                              name_prefix=os.path.basename(args.out))
    model.learn(total_timesteps=args.timesteps, callback=ckpt)
    model.save(args.out)
    print(f"saved {args.out}.zip")


if __name__ == "__main__":
    main()
