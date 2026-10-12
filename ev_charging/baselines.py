"""Non-learning scheduling baselines. Each takes the env and returns an action."""
import numpy as np


def _allocate(env, order):
    """Give full power in priority order until the grid budget runs out."""
    c = env.cfg
    action = np.zeros(c.n_ports, dtype=np.int64)
    budget = c.p_grid
    for i in order:
        car = env.ports[i]
        for lvl in range(len(c.levels) - 1, 0, -1):   # highest level that still fits
            p = env.car_power(car, c.levels[lvl])
            if budget >= p - 1e-9:
                action[i] = lvl
                budget -= p
                break
    return action


def _occupied(env):
    return [i for i, car in enumerate(env.ports) if car is not None]


def fifo(env, rng=None):
    idx = sorted(_occupied(env), key=lambda i: env.ports[i].arrival)
    return _allocate(env, idx)


def edf(env, rng=None):
    idx = sorted(_occupied(env), key=lambda i: env.ports[i].declared_dep)
    return _allocate(env, idx)


def llf(env, rng=None):
    idx = sorted(_occupied(env), key=lambda i: env.laxity(env.ports[i]))
    return _allocate(env, idx)


def peak_hold(env, rng=None, slack=24.0, threshold=0.3):
    """LLF, but at peak price cars with plenty of slack are held back."""
    idx = sorted(_occupied(env), key=lambda i: env.laxity(env.ports[i]))
    if env.prices[env.t] > threshold:
        idx = [i for i in idx if env.laxity(env.ports[i]) < slack]
    return _allocate(env, idx)


def random_policy(env, rng=None):
    return env.action_space.sample()


BASELINES = {
    "FIFO": fifo,
    "EDF": edf,
    "LLF": llf,
    "Peak-hold LLF": peak_hold,
    "Random": random_policy,
}
