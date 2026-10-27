"""Single-station EV charging environment (Gymnasium).

One station, N ports, a grid limit well below the sum of port limits. Every 15 minutes
the agent picks a power level per port. Cars arrive by a Poisson process, differ in
energy demand, stay length and maximum charge rate, declare a departure time, and
actually leave at the declared time plus noise.
"""
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# name -> (energy demand range kWh, declared stay range in steps)
CAR_TYPES = {
    "short_stop": ((5.0, 15.0), (4, 12)),      # 1-3 h
    "commuter": ((10.0, 30.0), (28, 40)),      # 7-10 h
    "overnight": ((20.0, 50.0), (40, 64)),     # 10-16 h
}
# max charge rate of the car (kW) and its probability
CAR_RATES = (3.7, 7.4, 11.0)
CAR_RATE_PROBS = (0.35, 0.45, 0.20)


def type_probs(hour: float) -> dict:
    """Mix of car types depends on the time of day."""
    if 5 <= hour < 11:
        return {"short_stop": 0.3, "commuter": 0.6, "overnight": 0.1}
    if 11 <= hour < 17:
        return {"short_stop": 0.6, "commuter": 0.3, "overnight": 0.1}
    return {"short_stop": 0.2, "commuter": 0.2, "overnight": 0.6}


@dataclass
class EnvConfig:
    n_ports: int = 10
    p_port: float = 11.0         # kW hardware limit per port
    p_grid: float = 40.0         # kW station limit (~4 ports at full power)
    dt: float = 0.25             # hours per step
    horizon: int = 96            # steps per episode (24 h)
    levels: tuple = (0.0, 0.5, 1.0)
    depart_noise_std: float = 2.0   # steps; actual = declared + N(0, std)
    arrival_scale: float = 1.0
    # reward weights
    alpha: float = 1.0           # per kWh unmet at departure (or rejected)
    beta: float = 0.05           # per idle plugged car per step
    gamma: float = 1.0           # per car fully charged
    price_noise: float = 0.05
    price_spread: float = 1.0    # scales the tariff's deviation from the 0.20 mid price


@dataclass
class Car:
    arrival: int
    need: float                  # remaining energy, kWh
    max_rate: float              # kW the car can accept
    declared_dep: int
    actual_dep: int
    idle_steps: int = 0


def tou_price(step: int, dt: float = 0.25) -> float:
    """Deterministic time-of-use tariff in currency per kWh."""
    hour = (step * dt) % 24
    if 17 <= hour < 21:
        return 0.40              # evening peak
    if 7 <= hour < 17:
        return 0.20              # daytime
    return 0.10                  # night off-peak


def arrival_rate(step: int, dt: float = 0.25) -> float:
    """Expected arrivals per step: low base + morning and evening rushes."""
    hour = (step * dt) % 24
    g = lambda mu, sd: np.exp(-0.5 * ((hour - mu) / sd) ** 2)
    return 0.08 + 0.5 * g(8.0, 1.5) + 0.6 * g(18.0, 2.0)


class EVChargingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        c = self.cfg
        self.action_space = spaces.MultiDiscrete([len(c.levels)] * c.n_ports)
        # per port: occupied, remaining energy, declared time left, urgency, max rate
        # global: sin, cos, price, price +1h, price +4h
        self.obs_dim = 5 * c.n_ports + 5
        self.observation_space = spaces.Box(-1.0, 3.0, shape=(self.obs_dim,), dtype=np.float32)
        self.base_prices = np.array([0.20 + c.price_spread * (tou_price(t, c.dt) - 0.20)
                                     for t in range(c.horizon + 17)])
        self.ports: list[Car | None] = []
        self.t = 0

    # ------------------------------------------------------------------ helpers
    def car_power(self, car: Car, level: float) -> float:
        """kW a car actually draws at a given port power level."""
        return min(level * self.cfg.p_port, car.max_rate)

    def _sample_car(self) -> Car:
        c = self.cfg
        probs = type_probs((self.t * c.dt) % 24)
        names = list(probs)
        kind = names[self.rng.choice(len(names), p=[probs[n] for n in names])]
        (e_lo, e_hi), (s_lo, s_hi) = CAR_TYPES[kind]
        need = float(self.rng.uniform(e_lo, e_hi))
        rate = float(self.rng.choice(CAR_RATES, p=CAR_RATE_PROBS))
        stay = int(self.rng.integers(s_lo, s_hi + 1))
        declared = self.t + stay
        noise = int(round(self.rng.normal(0.0, c.depart_noise_std)))
        actual = max(self.t + 1, declared + noise)
        return Car(self.t, need, rate, declared, actual)

    def _arrivals(self):
        c = self.cfg
        n = self.rng.poisson(arrival_rate(self.t, c.dt) * c.arrival_scale)
        for _ in range(n):
            car = self._sample_car()
            self.stats["arrived"] += 1
            free = [i for i, p in enumerate(self.ports) if p is None]
            if free:
                self.ports[free[0]] = car
            else:
                self.stats["rejected"] += 1
                self._step_rejected_kwh += car.need   # rejected demand counts as unmet

    def _obs(self) -> np.ndarray:
        c = self.cfg
        o = []
        for car in self.ports:
            if car is None:
                o += [0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                left = max(car.declared_dep - self.t, 0)
                need_rate = car.need / (max(left, 1) * car.max_rate * c.dt)  # >1: infeasible
                o += [1.0, car.need / 50.0, left / c.horizon, min(need_rate, 2.0),
                      car.max_rate / c.p_port]
        ang = 2 * np.pi * self.t / c.horizon
        o += [np.sin(ang), np.cos(ang), self.prices[min(self.t, c.horizon - 1)] / 0.4,
              self.base_prices[self.t + 4] / 0.4, self.base_prices[self.t + 16] / 0.4]
        return np.asarray(o, dtype=np.float32)

    def laxity(self, car: Car) -> float:
        """Steps of slack: declared time left minus steps needed at the car's max rate."""
        return (car.declared_dep - self.t) - car.need / (car.max_rate * self.cfg.dt)

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = self.np_random
        c = self.cfg
        self.t = 0
        self.ports = [None] * c.n_ports
        noise = np.exp(self.rng.normal(0.0, c.price_noise, size=c.horizon))
        self.prices = self.base_prices[: c.horizon] * noise
        self.stats = dict(arrived=0, rejected=0, completed=0, unmet_cars=0, unmet_kwh=0.0,
                          cost=0.0, energy=0.0, idle_steps=0, ret=0.0)
        self._step_rejected_kwh = 0.0
        self._arrivals()
        return self._obs(), {}

    def step(self, action):
        c = self.cfg
        self._step_rejected_kwh = 0.0
        req = np.zeros(c.n_ports)
        for i, car in enumerate(self.ports):
            if car is not None:
                req[i] = self.car_power(car, c.levels[int(action[i])])
        total = req.sum()
        if total > c.p_grid:                       # enforce the grid limit
            req *= c.p_grid / total

        price = self.prices[self.t]
        delivered, idle = 0.0, 0
        for i, car in enumerate(self.ports):
            if car is None:
                continue
            e = min(req[i] * c.dt, car.need)
            car.need -= e
            delivered += e
            if e < 1e-9:
                car.idle_steps += 1
                idle += 1
        cost = price * delivered

        self.t += 1
        unmet_kwh, done_cnt = 0.0, 0
        last = self.t >= c.horizon
        for i, car in enumerate(self.ports):
            if car is None:
                continue
            if car.need <= 1e-6:
                done_cnt += 1
                self.ports[i] = None
            elif self.t >= car.actual_dep or last:
                unmet_kwh += car.need
                self.stats["unmet_cars"] += 1
                self.ports[i] = None
        if not last:
            self._arrivals()

        reward = (-cost - c.alpha * (unmet_kwh + self._step_rejected_kwh)
                  - c.beta * idle + c.gamma * done_cnt)
        s = self.stats
        s["ret"] += reward
        s["cost"] += cost
        s["energy"] += delivered
        s["unmet_kwh"] += unmet_kwh
        s["completed"] += done_cnt
        s["idle_steps"] += idle
        info = {}
        if last:
            served = max(s["arrived"] - s["rejected"], 1)
            info = dict(s, completion_rate=s["completed"] / served,
                        service_rate=s["completed"] / max(s["arrived"], 1),
                        mean_idle_steps=s["idle_steps"] / served)
        return self._obs(), float(reward), last, False, info
