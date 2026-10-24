# RL for EV Charging under a Limited Grid Budget

AI Assignment 2 (mini project): a Reinforcement Learning agent (PPO) that schedules power to
electric vehicles at a single charging station with limited grid capacity, time-of-use prices
and uncertain departure times.

**Team:** Yateen (B23CS1082), Neer Modi (B23CS1043)

## Problem

A station has **10 ports (11 kW each)** but a **40 kW grid limit**, so only about 4 ports can run
at full power at once. Every 15 minutes the agent decides how much power each port gets, over a
24 h episode (96 steps).

- **Cars:** Poisson arrivals with morning/evening rushes. Three types (short-stop, commuter,
  overnight) with different energy needs (5-50 kWh) and stay lengths (1-16 h); the mix depends on
  the time of day. Each car has its own max charge rate (3.7 / 7.4 / 11 kW).
- **Uncertain departures:** a car declares a departure time but leaves at declared + N(0, 2^2)
  steps. A car also leaves when fully charged.
- **Prices:** time-of-use tariff (0.10 night / 0.20 day / 0.40 peak 17:00-21:00 per kWh) with 5%
  noise.

### RL formulation

| | |
|---|---|
| **State** (55-d) | per port (10 x 5): occupied, remaining energy, declared time left, urgency, max rate; global (5): sin/cos of time of day, current price, tariff +1 h, +4 h |
| **Action** | `MultiDiscrete([3]*10)`: power level {0, 0.5, 1} x 11 kW per port, capped by the car's max rate; requests above 40 kW are scaled down proportionally |
| **Reward** | `r = -price * energy - alpha * (unmet + rejected kWh) - beta * idle_cars + gamma * completed_cars`, with alpha = 1.0, beta = 0.05, gamma = 1.0 |
| **Algorithm** | PPO (Stable-Baselines3), MLP 128x128 |

### Baselines (non-learning)

FIFO, EDF (earliest declared departure), LLF (least laxity first), Peak-hold LLF (LLF, but
cars with lots of slack are held back when the price is above 0.30), and Random.

## Repository layout

```
ev_charging/
  env.py          Gymnasium environment (EVChargingEnv, EnvConfig)
  baselines.py    FIFO / EDF / LLF / Peak-hold LLF / Random
train.py          PPO training script
evaluate.py       Evaluation, plots and noise-robustness sweep
requirements.txt  Python dependencies
models/           Trained agent (models/ppo_ev.zip) and training checkpoints
results/          Evaluation CSVs and figures (created by evaluate.py)
report/report.tex Two-page report (LaTeX, two-column)
old/              Models and logs from earlier experimental versions (not used)
```

## Installation

Requires Python 3.10+ (developed with 3.13).

```bash
python -m venv .venv
# Windows PowerShell:  .\.venv\Scripts\Activate.ps1
# Linux / macOS:       source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies: numpy, gymnasium, stable-baselines3, torch, matplotlib, pandas.

## Usage

**1. Evaluate the baselines only (no training needed):**

```bash
python evaluate.py --baselines-only --episodes 100
```

**2. Train the PPO agent** (about 1 hour on a laptop CPU with 2 environments):

```bash
python train.py --timesteps 1500000 --n-envs 2 --seed 0 --out models/ppo_ev
```

This saves `models/ppo_ev.zip`, periodic checkpoints in `models/checkpoints/`, and training
logs in `logs/ppo_ev/`. Use `--n-envs 4` or `8` if you have the memory.

**3. Evaluate the trained agent against all baselines** (100 episodes, same seeds for every
method), including the departure-noise sweep:

```bash
python evaluate.py --models models/ppo_ev --episodes 100 --noise-sweep
```

Outputs are written to `results/`: `main.csv`, `main_comparison.png`, `training_curve.png`,
`noise_sweep.csv`, `noise_robustness.png`.

Useful options (both scripts): `--alpha`, `--arrival-scale`, `--price-spread`;
`train.py` also takes `--noise` (departure-noise std in steps) and `--seed`.

## Reproducing the results

Run steps 2 and 3 above with the same seed (`--seed 0`). Evaluation uses fixed episode seeds
(10000 to 10099), so baseline numbers are exactly reproducible. PPO numbers can differ slightly
between machines because of floating-point and library-version differences.

## Results

Mean over 100 evaluation episodes (return: higher is better; cost in currency units; "fully
charged" = fraction of arriving cars that finished charging):

Method Return (higher is better) Cost Cars fully charged / arrived Unmet kWh
FIFO −229 103.8 0.638 104.1
PPO −231 103.6 0.656 112.0
EDF −232 103.4 0.631 110.0
Peak-hold LLF −243 92.6 0.609 124.3
Random −295 88.9 0.423 144.