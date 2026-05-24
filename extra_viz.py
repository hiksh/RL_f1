"""
Additional visualizations for the report.
Generates 5 extra figures into results/.
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(__file__))
from env import F1RaceEnv, N_STATES, N_ACTIONS, N_LAP

RESULTS = os.path.join(os.path.dirname(__file__), "results")
ACTION_NAMES   = ["Maintain", "Push", "Recharge", "Pit-Soft", "Pit-Med", "Pit-Hard"]
METHOD_LABELS  = ["Value\nIteration", "Policy\nIteration", "MC\nControl",
                  "SARSA", "Q-learning", "Double\nQ-learning"]
METHOD_KEYS    = ["VI", "PI", "MC", "SARSA", "QL", "DQL"]
PALETTE        = ["#1E88E5", "#42A5F5", "#F5A623", "#7ED321", "#D0021B", "#9B59B6"]

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})


def _load():
    p_vi    = np.load(f"{RESULTS}/policy_vi.npy")
    p_pi    = np.load(f"{RESULTS}/policy_pi.npy")
    Q_mc    = np.load(f"{RESULTS}/Q_mc.npy")
    Q_sarsa = np.load(f"{RESULTS}/Q_sarsa.npy")
    Q_ql    = np.load(f"{RESULTS}/Q_ql.npy")
    Q_dql   = np.load(f"{RESULTS}/Q_dql.npy")
    r_mc    = np.load(f"{RESULTS}/rewards_mc.npy")
    r_sarsa = np.load(f"{RESULTS}/rewards_sarsa.npy")
    r_ql    = np.load(f"{RESULTS}/rewards_ql.npy")
    r_dql   = np.load(f"{RESULTS}/rewards_dql.npy")
    policies = [p_vi, p_pi,
                np.argmax(Q_mc, axis=1), np.argmax(Q_sarsa, axis=1),
                np.argmax(Q_ql, axis=1), np.argmax(Q_dql,  axis=1)]
    mf_rewards = [r_mc, r_sarsa, r_ql, r_dql]
    return policies, mf_rewards


# ── 1. Epsilon Decay Curve ────────────────────────────────────────────────────
def plot_epsilon_decay(eps_start=1.0, eps_min=0.05, decay=0.99993, n=50_000,
                       save_path=None):
    eps = eps_start
    curve = []
    converge_ep = None
    for i in range(n):
        curve.append(eps)
        eps = max(eps_min, eps * decay)
        if converge_ep is None and eps <= eps_min + 1e-9:
            converge_ep = i + 1

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(curve, color="#D0021B", lw=1.8, label=r"$\varepsilon$ (exploration rate)")
    ax.axhline(eps_min, color="#888888", lw=1.0, ls="--", label=f"Min ε = {eps_min}")
    ax.axvline(converge_ep, color="#F5A623", lw=1.4, ls="--",
               label=f"Reaches min at ep ≈ {converge_ep:,}")
    ax.fill_between(range(n), curve, eps_min, alpha=0.08, color="#D0021B")

    ax.annotate(f"ε → {eps_min}\n(ep {converge_ep:,})",
                xy=(converge_ep, eps_min + 0.01),
                xytext=(converge_ep + 3000, 0.25),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
                fontsize=9, color="#555555")

    # shade phases
    ax.axvspan(0, converge_ep, alpha=0.04, color="#D0021B", label="Exploration phase")
    ax.axvspan(converge_ep, n, alpha=0.04, color="#1E88E5", label="Exploitation phase")

    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("ε (Epsilon)", fontsize=10)
    ax.set_title("Epsilon-Greedy Decay Schedule  (decay = 0.99993, 50 k episodes)",
                 fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = save_path or f"{RESULTS}/epsilon_decay.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── 2. Action Distribution Stacked Bar ───────────────────────────────────────
def plot_action_distribution(policies, save_path=None):
    ACTION_COLORS = ["#455A64","#2E7D32","#7B1FA2","#C62828","#E65100","#FF8F00"]
    n_methods = len(policies)
    data = np.zeros((n_methods, N_ACTIONS))
    for i, pol in enumerate(policies):
        counts = np.bincount(pol, minlength=N_ACTIONS)
        data[i] = counts / len(pol) * 100

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(n_methods)
    bottoms = np.zeros(n_methods)
    bars_all = []
    for a in range(N_ACTIONS):
        bars = ax.bar(x, data[:, a], bottom=bottoms,
                      color=ACTION_COLORS[a], label=ACTION_NAMES[a],
                      edgecolor="white", linewidth=0.5)
        bars_all.append(bars)
        for b_idx, (bar, val) in enumerate(zip(bars, data[:, a])):
            if val > 4.0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bottoms[b_idx] + val / 2,
                        f"{val:.1f}%", ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
        bottoms += data[:, a]

    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_LABELS, fontsize=10)
    ax.set_ylabel("Proportion of States (%)", fontsize=10)
    ax.set_title("Action Distribution Across All States — Greedy Policy per Method",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9, ncol=2,
              framealpha=0.9, edgecolor="#cccccc")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    path = save_path or f"{RESULTS}/action_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── 3. Pairwise Policy Agreement Matrix ──────────────────────────────────────
def plot_policy_agreement(policies, save_path=None):
    n = len(policies)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = (policies[i] == policies[j]).mean() * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="Agreement (%)")
    ax.set_xticks(range(n)); ax.set_xticklabels(METHOD_KEYS, fontsize=10)
    ax.set_yticks(range(n)); ax.set_yticklabels(METHOD_KEYS, fontsize=10)
    for i in range(n):
        for j in range(n):
            color = "white" if mat[i, j] > 60 else "#222222"
            ax.text(j, i, f"{mat[i,j]:.1f}%", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")
    ax.set_title("Pairwise Policy Agreement Matrix\n(% of states with same greedy action)",
                 fontsize=11)
    plt.tight_layout()
    path = save_path or f"{RESULTS}/policy_agreement.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── 4. Convergence Analysis (smoothed + phases) ───────────────────────────────
def plot_convergence_analysis(mf_rewards, window=500, converge_ep=42800,
                              save_path=None):
    mf_labels  = ["MC Control", "SARSA", "Q-learning", "Double Q-learning"]
    mf_colors  = ["#F5A623", "#7ED321", "#D0021B", "#9B59B6"]

    def smooth(x):
        return np.convolve(x, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    # ── Left: smoothed curves with phase shading ──
    ax = axes[0]
    n_ep = len(mf_rewards[0])
    ax.axvspan(0, converge_ep, alpha=0.05, color="#D0021B", zorder=0)
    ax.axvspan(converge_ep, n_ep - window, alpha=0.05, color="#1E88E5", zorder=0)
    ax.axvline(converge_ep, color="#888888", lw=1.0, ls="--", zorder=1)
    ax.text(converge_ep + 500, -119, "ε min\n~42,800", fontsize=8,
            color="#666666", va="top")

    for rewards, label, color in zip(mf_rewards, mf_labels, mf_colors):
        s = smooth(rewards)
        ax.plot(s, label=label, color=color, lw=1.6, zorder=2)

    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Smoothed Total Reward  (window=500)", fontsize=10)
    ax.set_title("Learning Curves with Training Phases", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.text(5000, -121, "Exploration\nPhase", fontsize=9, color="#C62828",
            alpha=0.7, ha="left")
    ax.text(44000, -121, "Exploitation\nPhase", fontsize=9, color="#1565C0",
            alpha=0.7, ha="left")

    # ── Right: improvement bar (first 1k → last 1k) ──
    ax2 = axes[1]
    improvements = []
    for rewards in mf_rewards:
        imp = rewards[-1000:].mean() - rewards[:1000].mean()
        improvements.append(imp)

    bars = ax2.bar(mf_labels, improvements,
                   color=mf_colors, alpha=0.85, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, improvements):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 val + 0.3 if val >= 0 else val - 0.3,
                 f"{val:+.1f}", ha="center",
                 va="bottom" if val >= 0 else "top",
                 fontsize=10, fontweight="bold", color="#333333")

    ax2.axhline(0, color="#888888", lw=0.8, ls="--")
    ax2.set_ylabel("Reward Improvement\n(Last 1k − First 1k episodes)", fontsize=10)
    ax2.set_title("Total Learning Gain over 50k Episodes", fontsize=11)
    ax2.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    path = save_path or f"{RESULTS}/convergence_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── 5. Eval Performance with DP Optimal Reference ────────────────────────────
def plot_performance_with_dp_ref(n_eval=300, save_path=None):
    env = F1RaceEnv(n_laps=N_LAP)

    def _eval(policy):
        totals = []
        for _ in range(n_eval):
            s, done, total = env.reset(), False, 0.0
            while not done:
                s, r, done = env.step(int(policy[s]))
                total += r
            totals.append(total)
        return np.array(totals)

    p_vi    = np.load(f"{RESULTS}/policy_vi.npy")
    p_pi    = np.load(f"{RESULTS}/policy_pi.npy")
    Q_mc    = np.load(f"{RESULTS}/Q_mc.npy")
    Q_sarsa = np.load(f"{RESULTS}/Q_sarsa.npy")
    Q_ql    = np.load(f"{RESULTS}/Q_ql.npy")
    Q_dql   = np.load(f"{RESULTS}/Q_dql.npy")

    entries = [
        ("Value\nIteration",   p_vi),
        ("Policy\nIteration",  p_pi),
        ("MC\nControl",        np.argmax(Q_mc,    axis=1)),
        ("SARSA",              np.argmax(Q_sarsa, axis=1)),
        ("Q-learning",         np.argmax(Q_ql,    axis=1)),
        ("Double\nQ-learning", np.argmax(Q_dql,   axis=1)),
    ]

    print("  Evaluating policies (this may take a moment)...")
    all_data = [_eval(e[1]) for e in entries]
    labels   = [e[0] for e in entries]
    means    = [d.mean() for d in all_data]

    dp_mean = np.mean(means[:2])  # DP reference line

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # ── Left: violin + box ──
    ax = axes[0]
    parts = ax.violinplot(all_data, positions=range(len(labels)),
                          showmedians=True, showextrema=True)
    for i, (pc, color) in enumerate(zip(parts["bodies"], PALETTE)):
        pc.set_facecolor(color)
        pc.set_alpha(0.45)
    parts["cmedians"].set_color("#222222")
    parts["cmedians"].set_linewidth(1.8)

    for key in ("cmaxes", "cmins", "cbars"):
        parts[key].set_color("#888888")
        parts[key].set_linewidth(1.0)

    for i, (mean, color) in enumerate(zip(means, PALETTE)):
        ax.scatter(i, mean, color=color, s=60, zorder=5,
                   edgecolors="white", linewidths=0.8)

    ax.axhline(dp_mean, color="#1E88E5", lw=1.5, ls="--", alpha=0.8,
               label=f"DP reference ({dp_mean:.1f})")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Episode Total Reward", fontsize=10)
    ax.set_title(f"Evaluation Performance Distribution  (n={n_eval} episodes)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    # ── Right: gap to DP optimal ──
    ax2 = axes[1]
    gaps = [m - dp_mean for m in means]
    bar_colors = ["#1E88E5" if i < 2 else "#E57373" for i in range(len(gaps))]
    bars = ax2.bar(labels, gaps, color=bar_colors, alpha=0.85,
                   edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, gaps):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 val * 0.5 if val < 0 else val + 0.3,
                 f"{val:+.1f}", ha="center", va="center",
                 fontsize=9, fontweight="bold", color="white" if val < -2 else "#333333")

    ax2.axhline(0, color="#555555", lw=1.0, ls="--")
    ax2.set_ylabel("Gap to DP Reference (reward units)", fontsize=10)
    ax2.set_title("Performance Gap: Each Method vs. DP Optimal", fontsize=11)
    dp_patch = mpatches.Patch(color="#1E88E5", alpha=0.85, label="DP methods")
    mf_patch = mpatches.Patch(color="#E57373", alpha=0.85, label="Model-free methods")
    ax2.legend(handles=[dp_patch, mf_patch], fontsize=9)
    ax2.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    path = save_path or f"{RESULTS}/performance_gap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    print("Loading data...")
    policies, mf_rewards = _load()

    print("\n[1/5] Epsilon decay curve")
    plot_epsilon_decay()

    print("[2/5] Action distribution")
    plot_action_distribution(policies)

    print("[3/5] Policy agreement matrix")
    plot_policy_agreement(policies)

    print("[4/5] Convergence analysis")
    plot_convergence_analysis(mf_rewards)

    print("[5/5] Performance gap (running eval episodes...)")
    plot_performance_with_dp_ref()

    print("\nDone. All figures saved to results/")


if __name__ == "__main__":
    main()
