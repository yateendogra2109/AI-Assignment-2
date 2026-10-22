from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report"
FIGURES = OUT / "figures"
MAIN_CSV = ROOT / "results" / "main.csv"
NOISE_CSV = ROOT / "results" / "noise_sweep.csv"
PAGE_SIZE = (8.27, 11.69)
METHOD_ORDER = ["FIFO", "EDF", "LLF", "Peak-hold LLF", "PPO", "Random"]
COLORS = {
    "FIFO": "#3976a8",
    "EDF": "#59a14f",
    "LLF": "#9c755f",
    "Peak-hold LLF": "#9467bd",
    "PPO": "#d97706",
    "Random": "#7f7f7f",
}


def load_results():
    if not MAIN_CSV.exists() or not NOISE_CSV.exists():
        raise FileNotFoundError("Expected results/main.csv and results/noise_sweep.csv")
    main = pd.read_csv(MAIN_CSV)
    noise = pd.read_csv(NOISE_CSV)
    required = {"method", "ret", "cost", "service_rate", "unmet_kwh"}
    if not required.issubset(main.columns):
        raise ValueError(f"{MAIN_CSV} is missing columns: {sorted(required - set(main.columns))}")
    if not {"method", "noise", "service_rate"}.issubset(noise.columns):
        raise ValueError(f"{NOISE_CSV} must contain method, noise, and service_rate")
    return main, noise


def make_figures(main, noise):
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary = main.groupby("method").agg(
        ret=("ret", "mean"),
        ret_sem=("ret", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        cost=("cost", "mean"),
        cost_sem=("cost", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        service=("service_rate", "mean"),
        service_sem=("service_rate", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
    ).reindex(METHOD_ORDER)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.25))
    panels = [
        ("cost", "cost_sem", "Mean energy cost (tariff units)", False),
        ("service", "service_sem", "Cars fully charged / arrived", True),
        ("ret", "ret_sem", "Mean episode return (higher is better)", False),
    ]
    for ax, (metric, error, title, percent) in zip(axes, panels):
        vals = summary[metric].to_numpy()
        errs = summary[error].to_numpy()
        ax.bar(
            np.arange(len(summary)), vals, yerr=errs, capsize=2,
            color=[COLORS[name] for name in summary.index],
        )
        ax.set_xticks(np.arange(len(summary)))
        ax.set_xticklabels(summary.index, rotation=42, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        if percent:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Evaluation at departure-noise standard deviation 2 steps (100 episodes / method)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    main_fig = FIGURES / "policy_comparison.png"
    fig.savefig(main_fig, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.3, 3.0))
    for method in METHOD_ORDER:
        subset = noise[noise["method"] == method]
        grouped = subset.groupby("noise")["service_rate"]
        means = grouped.mean()
        sems = grouped.sem()
        ax.errorbar(
            means.index, means.values, yerr=sems.values, marker="o",
            capsize=2, linewidth=1.5, markersize=3,
            color=COLORS[method], label=method,
        )
    ax.set_xlabel("Departure-time noise (steps; 15 min / step)", fontsize=8)
    ax.set_ylabel("Cars fully charged / arrived", fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=6, ncol=2, frameon=False)
    ax.set_title("Service under departure-time uncertainty (50 episodes / point)", fontsize=9)
    fig.tight_layout()
    noise_fig = FIGURES / "noise_robustness.png"
    fig.savefig(noise_fig, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary, main_fig, noise_fig


def wrap_lines(text, width=65):
    return textwrap.wrap(
        text, width=width, break_long_words=False, break_on_hyphens=False,
        replace_whitespace=True,
    ) or [""]


def add_text(fig, x, y, content, *, size=7.1, color="#20252b",
             weight="normal", width=65, family="DejaVu Sans"):
    line_height = size * 1.28 / 72 / PAGE_SIZE[1]
    for line in wrap_lines(content, width):
        fig.text(x, y, line, ha="left", va="top", fontsize=size,
                 color=color, weight=weight, family=family)
        y -= line_height
    return y


def add_heading(fig, x, y, content, *, size=9.0, width=65):
    y = add_text(fig, x, y, content, size=size, color="#174a6e",
                 weight="bold", width=width)
    return y - 0.012


def add_paragraph(fig, x, y, content, *, size=7.1, width=65):
    y = add_text(fig, x, y, content, size=size, width=width)
    return y - 0.010


def add_image(fig, path, rect):
    ax = fig.add_axes(rect)
    ax.imshow(plt.imread(path))
    ax.axis("off")


def draw_page_one():
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.text(0.5, 0.972, "REAL-LIFE REINFORCEMENT LEARNING",
             ha="center", va="top", fontsize=14, weight="bold", color="#174a6e")
    fig.text(0.5, 0.946, "Grid-Constrained Electric-Vehicle Charging",
             ha="center", va="top", fontsize=11, weight="bold")
    fig.text(0.5, 0.924,
             "Shubham Bagwari (p22cs201@iitj.ac.in)  |  "
             "Divyaansh Mertia (p25cs0004@iitj.ac.in)",
             ha="center", va="top", fontsize=6.7)
    fig.text(0.5, 0.909,
             "Aniruddha Bohra (p25cs201@iitj.ac.in)  |  "
             "Gajera Jahanvi Arvindkumar (m25csa012@iitj.ac.in)",
             ha="center", va="top", fontsize=6.7)
    fig.text(0.5, 0.894,
             "Bhatt Vasishth Bimalbhai (m25csa007@iitj.ac.in)",
             ha="center", va="top", fontsize=6.7)
    fig.text(0.5, 0.879, "GitHub repository: [insert team repository URL]",
             ha="center", va="top", fontsize=7.0, color="#a04a00")

    left_x, right_x, top = 0.075, 0.535, 0.850
    y = add_heading(fig, left_x, top, "Abstract")
    y = add_paragraph(
        fig, left_x, y,
        "EV stations allocate a constrained 40-kW grid across ten ports amid "
        "variable tariffs and uncertain departures. We model charging as a "
        "finite-horizon MDP: PPO observes car needs, urgency, rate limits and "
        "tariff context, chooses per-port power, and earns completion rewards "
        "while penalized for energy cost, unmet demand and idle charging.",
    )
    y = add_heading(fig, left_x, y, "Problem and simulated environment")
    y = add_paragraph(
        fig, left_x, y,
        "The operational problem is online charging control: serve arriving cars "
        "without exceeding station capacity, avoid expensive periods when slack "
        "allows, and finish as many cars as possible before they leave. The "
        "Gymnasium [3] simulator represents one 24-hour day in 96 steps of 15 "
        "minutes. Ten ports share a 40-kW grid cap; each hardware port is rated "
        "up to 11 kW.",
    )
    y = add_paragraph(
        fig, left_x, y,
        "Cars have 5–50 kWh requests, with type-dependent stays: short-stop "
        "1–3 h, commuter 7–10 h, and overnight 10–16 h. Their maximum accepted "
        "rates are 3.7, 7.4 or 11 kW. Evening/night arrivals (17:00–05:00) are "
        "mostly overnight parkers (60%); departure is perturbed by Gaussian "
        "noise (default standard deviation 2 steps = 30 minutes). Arrival counts "
        "are stochastic, and a full station rejects new demand.",
    )
    y = add_heading(fig, left_x, y, "State space  S")
    y = add_paragraph(
        fig, left_x, y,
        "The observation is a 55-dimensional vector (5 values × 10 ports + 5 "
        "global values). Per port: occupancy, remaining energy / 50 kWh, "
        "declared time remaining / 96 steps, normalized urgency (need divided "
        "by feasible energy at the car's maximum rate, clipped at 2), and "
        "maximum rate / 11 kW. Global features: sine and cosine of time, current "
        "price / 0.40, and base tariff 1 h and 4 h ahead / 0.40. Actual "
        "departure time is not observed.",
    )
    y = add_heading(fig, left_x, y, "Dynamics and uncertainty")
    y = add_paragraph(
        fig, left_x, y,
        "The tariff is 0.10 currency/kWh overnight, 0.20 daytime, and 0.40 "
        "during the 17:00–21:00 peak, with small multiplicative price noise. "
        "Arrivals follow a time-varying Poisson rate. A car that remains "
        "incomplete at its actual departure (or episode end) contributes "
        "unmet energy. These randomized arrivals, needs, rates, stays and "
        "departures make the decision process stochastic.",
    )
    left_bottom = y

    y = add_heading(fig, right_x, top, "Action space  A")
    y = add_paragraph(
        fig, right_x, y,
        "At each step PPO selects one of three discrete levels for each of the "
        "10 ports: 0, 0.5 or 1.0. This is a MultiDiscrete action with 3^10 "
        "possible joint choices. Requested power is 0, 5.5 or 11 kW, capped "
        "by the connected car's individual rate. If the total exceeds 40 kW, "
        "all requested powers are scaled proportionally to enforce the grid "
        "limit. Empty ports receive no energy.",
    )
    y = add_heading(fig, right_x, y, "Reward function  R")
    y = add_paragraph(
        fig, right_x, y,
        "For step t, let E_t be delivered kWh, p_t the tariff, U_t unmet "
        "departure/terminal kWh, J_t rejected-arrival kWh, I_t the number of "
        "occupied ports receiving zero energy, and C_t cars completed. The "
        "implemented reward is:",
    )
    fig.text(right_x + 0.015, y - 0.002,
             "r_t = -p_t E_t - 1.0(U_t + J_t) - 0.05 I_t + 1.0 C_t",
             ha="left", va="top", fontsize=7.2, family="DejaVu Sans Mono",
             color="#20252b")
    y -= 0.034
    y = add_paragraph(
        fig, right_x, y,
        "Thus the objective trades tariff cost and unmet demand against a "
        "one-unit bonus per completed car, with a small idle penalty. The "
        "episode return is the sum of step rewards. Coefficients are simulator "
        "reward weights, not currency conversion factors.",
    )
    y = add_heading(fig, right_x, y, "Algorithm and training")
    y = add_paragraph(
        fig, right_x, y,
        "The learned controller is Proximal Policy Optimization (PPO) with a "
        "two-layer 128-unit MLP policy (Stable-Baselines3 [1, 2]). Training uses "
        "parallel environments, 96 × 8 rollout steps, batch size 256, 10 "
        "epochs, learning rate 3×10⁻⁴, discount γ=0.99, GAE λ=0.95 and "
        "entropy coefficient 0.01. Observation normalization is disabled and "
        "reward normalization is enabled during training. Evaluation uses "
        "deterministic policy actions.",
    )
    y = add_heading(fig, right_x, y, "Baselines")
    y = add_paragraph(
        fig, right_x, y,
        "We compare against FIFO (arrival order), EDF (earliest declared "
        "departure), LLF (least laxity), Peak-hold LLF (holds back cars with "
        "ample slack at peak price), and a random action policy. The scheduling "
        "baselines greedily allocate the highest power that fits the remaining "
        "grid budget in their priority order; they require no training.",
    )
    y = add_heading(fig, right_x, y, "Reproducibility")
    y = add_paragraph(
        fig, right_x, y,
        "Reported evaluation values are read from the project's results/main.csv "
        "and results/noise_sweep.csv. The comparison uses 100 episodes per "
        "method at departure-noise standard deviation 2; the PPO row pools "
        "episodes from two saved training seeds. Noise-sweep points use 50 "
        "episodes per method and noise setting. Figures show means with "
        "standard-error bars; no evaluation records were altered.",
    )

    if left_bottom < 0.045 or y < 0.045:
        raise RuntimeError(
            f"Page 1 column overflow: left={left_bottom:.3f}, right={y:.3f}"
        )
    fig.text(0.5, 0.022, "Assignment 2 | EV charging with reinforcement learning | 1",
             ha="center", va="bottom", fontsize=6.5, color="#626a70")
    return fig


def draw_page_two(summary, main_fig, noise_fig):
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.text(0.5, 0.976, "EXPERIMENTS, RESULTS AND DISCUSSION",
             ha="center", va="top", fontsize=12, weight="bold", color="#174a6e")
    left_x, right_x, top = 0.075, 0.535, 0.944
    y_left = add_heading(fig, left_x, top, "Evaluation protocol")
    y_left = add_paragraph(
        fig, left_x, y_left,
        "Each method is evaluated on 100 episodes at departure-noise "
        "standard deviation 2 steps; PPO pools two saved seeds. The noise "
        "sweep uses 50 episodes at each of 0, 1, 2, 4 and 6 steps. Service "
        "rate divides completed cars by all arrivals (including rejected); "
        "completion rate uses admitted cars only.",
    )
    y_left = add_heading(fig, left_x, y_left, "Main comparison")
    y_left = add_paragraph(
        fig, left_x, y_left,
        "Means from the existing evaluation CSV are summarized below. Return "
        "is better when less negative; lower cost / unmet kWh and higher "
        "service are better.",
    )

    rows = [
        ("Policy", "Return", "Cost", "Service", "Unmet kWh"),
        ("LLF", -222.6, 106.0, 64.9, 97.2),
        ("EDF", -224.8, 105.7, 66.3, 105.6),
        ("FIFO", -226.0, 105.4, 64.4, 103.3),
        ("Peak-hold LLF", -237.2, 93.6, 61.2, 119.1),
        ("PPO", -246.1, 98.9, 57.0, 114.5),
        ("Random", -334.0, 79.8, 34.7, 180.6),
    ]
    table_x = left_x + 0.002
    table_y = y_left
    line_h = 0.016
    for i, row in enumerate(rows):
        if i == 0:
            line = f"{row[0]:<15}{row[1]:>8}{row[2]:>8}{row[3]:>9}{row[4]:>10}"
        else:
            line = f"{row[0]:<15}{row[1]:>8.1f}{row[2]:>8.1f}{row[3]:>8.1f}%{row[4]:>9.1f}"
        fig.text(table_x, table_y, line, ha="left", va="top", fontsize=5.8,
                 family="DejaVu Sans Mono",
                 weight="bold" if i == 0 else "normal")
        table_y -= line_h
    y_left = table_y - 0.008
    y_left = add_paragraph(
        fig, left_x, y_left,
        "LLF has the best mean return (-222.6); EDF has the highest service "
        "(66.3%). PPO costs 6.4% less than EDF (98.9 vs 105.7), but service "
        "is 9.3 points lower and return is worse by 21.3. PPO clearly beats "
        "Random on service (57.0% vs 34.7%) and return (-246.1 vs -334.0).",
    )
    y_left = add_heading(fig, left_x, y_left, "Interpretation")
    y_left = add_paragraph(
        fig, left_x, y_left,
        "PPO is not the best overall policy in these data. It has lower cost "
        "than FIFO, EDF and LLF, but completes fewer cars and returns less. "
        "Peak-hold LLF has both lower cost (93.6) and higher service (61.2%) "
        "than PPO. Random's lower cost reflects much less demand served, so "
        "cost alone is not a sufficient objective.",
    )
    y_left = add_heading(fig, left_x, y_left, "Limitations and next steps")
    y_left = add_paragraph(
        fig, left_x, y_left,
        "This is a one-day simulator study, not deployment evidence; tariffs "
        "and arrivals are specified rather than fitted to station data. "
        "Only two PPO seeds are available. Next, tune reward weights on "
        "training seeds, evaluate held-out seeds, and compare matched-service "
        "cost or a Pareto frontier. Keep test episodes separate from tuning.",
    )

    y_right = add_heading(fig, right_x, top, "Robustness to departure noise")
    y_right = add_paragraph(
        fig, right_x, y_right,
        "Across noise standard deviations 0–6 steps (0–90 minutes), PPO "
        "service varies from 57.6% to 56.2%. EDF remains near 67.4% to 65.7%, "
        "and LLF near 66.0% to 64.4%. PPO is therefore relatively stable "
        "but does not close its service gap to the strongest scheduling "
        "baselines in this tested range.",
    )
    add_image(fig, noise_fig, [right_x, 0.555, 0.385, 0.205])
    y_right = 0.535
    y_right = add_heading(fig, right_x, y_right, "Conclusion")
    y_right = add_paragraph(
        fig, right_x, y_right,
        "The project formulates a realistic constrained EV charging problem "
        "as an RL task and demonstrates PPO learning beyond a random policy. "
        "On the current benchmark, PPO favors lower cost over maximal "
        "completion, while simple EDF/LLF heuristics remain stronger on "
        "service and episode return. This is the honest empirical conclusion; "
        "the comparative figure reports the recorded results without "
        "rescaling or changing them.",
    )

    add_image(fig, main_fig, [0.07, 0.075, 0.86, 0.30])
    fig.text(0.5, 0.045,
             "Figure 1. Main benchmark (mean ± standard error, n=100 episodes / method). "
             "PPO episodes pool two trained seeds.",
             ha="center", va="center", fontsize=6.3, color="#454b50")
    fig.text(0.5, 0.022, "Assignment 2 | EV charging with reinforcement learning | 2",
             ha="center", va="bottom", fontsize=6.5, color="#626a70")
    if y_left < 0.39:
        raise RuntimeError(f"Page 2 left column overlaps the main figure: y={y_left:.3f}")
    if y_right < 0.37:
        raise RuntimeError(f"Page 2 right column overlaps the main figure: y={y_right:.3f}")
    return fig


def draw_references_page():
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.text(0.075, 0.95, "References", ha="left", va="top",
             fontsize=13, weight="bold", color="#174a6e")
    refs = [
        "[1] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. "
        "(2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.",
        "[2] Raffin, A. et al. (2021). Stable-Baselines3: Reliable Reinforcement "
        "Learning Implementations. Journal of Machine Learning Research, 22(268), 1–8.",
        "[3] Towers, M. et al. (2024). Gymnasium: A Standard Interface for "
        "Reinforcement Learning Environments. arXiv:2407.17032.",
        "[4] Project source and recorded evaluation data: ev_charging/env.py, "
        "ev_charging/baselines.py, train.py, evaluate.py, results/main.csv, "
        "and results/noise_sweep.csv.",
    ]
    y = 0.90
    for ref in refs:
        y = add_paragraph(fig, 0.075, y, ref, size=8.5, width=100)
        y -= 0.02
    fig.text(0.5, 0.022, "References | excluded from the two-page report limit",
             ha="center", va="bottom", fontsize=6.5, color="#626a70")
    return fig


def main():
    main_data, noise_data = load_results()
    summary, main_fig, noise_fig = make_figures(main_data, noise_data)
    OUT.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT / "assignment2_report.pdf"
    figures = [
        draw_page_one(),
        draw_page_two(summary, main_fig, noise_fig),
        draw_references_page(),
    ]
    with PdfPages(pdf_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"Generated {pdf_path}")
    print(f"Generated {main_fig}")
    print(f"Generated {noise_fig}")
    print(f"Main evaluation records: {len(main_data)}")
    print(f"Noise-sweep records: {len(noise_data)}")
    print("Mean metrics by method:")
    print(main_data.groupby("method")[["ret", "cost", "service_rate", "unmet_kwh"]]
          .mean().reindex(METHOD_ORDER).round(3).to_string())


if __name__ == "__main__":
    main()
