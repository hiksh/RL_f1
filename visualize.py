import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from env import (decode, encode,
                 N_BATTERY, N_TIRE, N_TRAFFIC, N_SECTION,
                 SECTION_TYPE, STRAIGHT, CORNER, CHICANE, PIT_SECTION,
                 DRS_ZONES, DIRTY_AIR, CLEAN_AIR,
                 CRITICAL, LOW, MEDIUM, HIGH,
                 DEGRADED, WORN, FRESH,
                 MAINTAIN, PUSH, RECHARGE, PIT_STOP)

ACTION_NAMES  = ["Maintain", "Push", "Recharge", "Pit-Stop"]
BATTERY_NAMES = ["Critical", "Low", "Medium", "High"]
TIRE_NAMES    = ["Degraded", "Worn", "Fresh"]
TRAFFIC_NAMES = ["Dirty Air", "Clean Air"]
STYPE_NAMES   = ["Straight", "Corner", "Chicane"]

# Sector definitions (inclusive section ranges)
SECTORS = [
    (list(range(0,  5)),  "#E53935", "S1"),   # Sector 1 — red
    (list(range(5,  15)), "#1E88E5", "S2"),   # Sector 2 — blue
    (list(range(15, 19)), "#FDD835", "S3"),   # Sector 3 — yellow
]

# 19-section circuit coordinates (clockwise, C01-C19)
# Layout mirrors the circuit map image: hairpin bottom-left → left DRS straight
# → top complex → right-side chicanes → bottom DRS → pit/S-F
_SECTION_POS = np.array([
    [ 0.5,  0.3],   # S0  C01 Hairpin          (Corner)
    [ 1.5,  3.5],   # S1  C02 Left DRS str     (Straight)  ← DRS 1
    [ 2.0,  5.5],   # S2  C03 Upper left       (Corner)
    [ 3.5,  6.0],   # S3  C04 Top left         (Corner)
    [ 5.5,  7.5],   # S4  C05 Top centre       (Corner)
    [ 7.0,  7.0],   # S5  C06 Top              (Corner)
    [ 8.5,  7.5],   # S6  C07 Top-right        (Chicane)
    [ 9.5,  6.0],   # S7  C08 Far right        (Corner)
    [ 8.5,  5.0],   # S8  C09 Right side       (Corner)
    [ 6.5,  5.0],   # S9  C10 Inner str        (Straight)
    [ 5.5,  4.0],   # S10 C11 Centre corner    (Corner)
    [ 7.5,  3.0],   # S11 C12 Right chicane    (Chicane)
    [ 9.0,  2.5],   # S12 C13 Lower right      (Corner)
    [ 9.5,  1.5],   # S13 C14 Far lower right  (Corner)
    [ 8.5,  0.0],   # S14 C15 Bottom right     (Corner)
    [ 6.5,  1.5],   # S15 C16 Bottom centre    (Straight)
    [ 4.5,  2.5],   # S16 C17 Bottom DRS str   (Straight)  ← DRS 2
    [ 2.0,  1.5],   # S17 C18 Start/Finish PIT (Straight)
    [ 1.8,  2.8],   # S18 C19 Hairpin exit     (Corner)
])

TRACK_X = _SECTION_POS[:, 0]
TRACK_Y = _SECTION_POS[:, 1]

# Corner names shown in animation info panel
SECTION_NAMES = [
    "C01","C02","C03","C04","C05",
    "C06","C07","C08","C09","C10",
    "C11","C12","C13","C14","C15",
    "C16","C17","C18","C19",
]

# Speed-trap position (midpoint of C04→C05 long straight)
_SPEED_TRAP = (_SECTION_POS[3] + _SECTION_POS[4]) / 2

_CIRCUIT_CENTER = _SECTION_POS.mean(axis=0)
_N = 30   # spline points per section


def _smooth_circuit(pts, n=_N):
    """Catmull-Rom spline through a closed set of 2-D control points."""
    p = np.vstack([pts[-1], pts, pts[:3]])
    xs, ys = [], []
    for i in range(1, len(pts) + 1):
        p0, p1, p2, p3 = p[i-1], p[i], p[i+1], p[i+2]
        t  = np.linspace(0, 1, n, endpoint=False)
        t2, t3 = t**2, t**3
        xs.append(0.5*(2*p1[0]+(-p0[0]+p2[0])*t+(2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2+(-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3))
        ys.append(0.5*(2*p1[1]+(-p0[1]+p2[1])*t+(2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2+(-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3))
    return np.concatenate(xs), np.concatenate(ys)


# ── Pre-compute spline once ───────────────────────────────────────────────────
_SX, _SY = _smooth_circuit(_SECTION_POS, n=_N)


def _draw_circuit(ax, highlight_section=None, title=""):
    """
    Render the F1 circuit in the reference-image style:
    grid, axes, thick track, sector colours, corner circles,
    DRS markers, speed trap, start/finish, sector labels.
    """
    n = _N
    sx, sy = _SX, _SY

    # ── Canvas ────────────────────────────────────────────────────────────────
    ax.set_xlim(-1.5, 11.5)
    ax.set_ylim(-2.0,  9.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X", fontsize=9)
    ax.set_ylabel("Y", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, color="#d0d0d0", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=10, pad=7)

    # ── 1. Track outline (dark border) ───────────────────────────────────────
    sx_cl = np.append(sx, sx[0])
    sy_cl = np.append(sy, sy[0])
    ax.plot(sx_cl, sy_cl, color="#1a1a1a", lw=20,
            solid_capstyle="round", zorder=1)

    # ── 2. Sector-coloured fills ──────────────────────────────────────────────
    for sec_list, color, _ in SECTORS:
        i0 = sec_list[0]  * n
        i1 = (sec_list[-1] + 1) * n
        xs = sx[i0:i1]
        ys = sy[i0:i1]
        if sec_list[-1] == N_SECTION - 1:        # last sector wraps to origin
            xs = np.append(xs, sx[0])
            ys = np.append(ys, sy[0])
        ax.plot(xs, ys, color=color, lw=12,
                solid_capstyle="butt", zorder=2)

    # ── 3. DRS zone highlights (cyan overlay) ─────────────────────────────────
    for di in sorted(DRS_ZONES):
        xs = sx[di*n : (di+1)*n + 1]
        ys = sy[di*n : (di+1)*n + 1]
        ax.plot(xs, ys, color="#00E5FF", lw=12, alpha=0.55, zorder=3)

    # ── 4. Corner number circles ──────────────────────────────────────────────
    for i in range(N_SECTION):
        ax.plot(TRACK_X[i], TRACK_Y[i], "o", ms=15,
                color="white", markeredgecolor="#222222",
                markeredgewidth=1.3, zorder=5)
        ax.text(TRACK_X[i], TRACK_Y[i], f"{i+1:02d}",
                ha="center", va="center",
                fontsize=5.8, fontweight="bold", color="#111111", zorder=6)

    # ── 5. Start/Finish checkered flag ────────────────────────────────────────
    sf_x, sf_y = TRACK_X[PIT_SECTION], TRACK_Y[PIT_SECTION]
    cw = 0.18
    for col in range(4):
        for row in range(2):
            fc = "#111111" if (col + row) % 2 == 0 else "white"
            ax.add_patch(plt.Rectangle(
                (sf_x - 0.36 + col * cw, sf_y - 0.65 + row * cw),
                cw, cw, color=fc, zorder=7
            ))

    # ── 6. DRS detection markers + labels ─────────────────────────────────────
    dt_labels = {sorted(DRS_ZONES)[0]: "DT1", sorted(DRS_ZONES)[1]: "DT2"}
    for di, lbl in dt_labels.items():
        ax.plot(TRACK_X[di], TRACK_Y[di] + 0.4, "o",
                ms=7, color="#00C853", zorder=8)
        ax.text(TRACK_X[di] + 0.25, TRACK_Y[di] + 0.65,
                lbl, fontsize=7.5, color="#00C853",
                fontweight="bold", zorder=8)

    # ── 7. Speed trap ─────────────────────────────────────────────────────────
    ax.plot(_SPEED_TRAP[0], _SPEED_TRAP[1] + 0.3, "o",
            ms=7, color="#E040FB", zorder=8)
    ax.text(_SPEED_TRAP[0] + 0.25, _SPEED_TRAP[1] + 0.55,
            "ST", fontsize=7.5, color="#E040FB",
            fontweight="bold", zorder=8)

    # ── 8. Sector labels ──────────────────────────────────────────────────────
    for sec_list, color, lbl in SECTORS:
        mid_sec = sec_list[len(sec_list) // 2]
        d = _SECTION_POS[mid_sec] - _CIRCUIT_CENTER
        norm = np.linalg.norm(d) + 1e-9
        pos  = _SECTION_POS[mid_sec] + (d / norm) * 1.6
        ax.text(pos[0], pos[1], lbl,
                ha="center", va="center",
                fontsize=11, fontweight="bold", color=color, zorder=7)

    # ── 9. Car marker (animation use) ────────────────────────────────────────
    car_handle = None
    if highlight_section is not None:
        car_handle, = ax.plot(
            TRACK_X[highlight_section], TRACK_Y[highlight_section],
            "D", ms=13, color="red",
            markeredgecolor="white", markeredgewidth=1.2, zorder=9
        )

    # ── 10. Legend ────────────────────────────────────────────────────────────
    legend_items = [
        Line2D([0],[0], color="#E53935", lw=5, label="Sector 1"),
        Line2D([0],[0], color="#1E88E5", lw=5, label="Sector 2"),
        Line2D([0],[0], color="#FDD835", lw=5, label="Sector 3"),
        Line2D([0],[0], color="#00E5FF", lw=5, alpha=0.6, label="DRS Zone"),
        Line2D([0],[0], marker="o", ls="", ms=6,
               markerfacecolor="#00C853", label="DRS Detection"),
        Line2D([0],[0], marker="o", ls="", ms=6,
               markerfacecolor="#E040FB", label="Speed Trap"),
    ]
    ax.legend(handles=legend_items, loc="upper left",
              fontsize=6.5, framealpha=0.9, edgecolor="#aaaaaa",
              handlelength=1.5)

    return car_handle


# ── Public plotting functions ─────────────────────────────────────────────────

def plot_circuit_map(save_path="results/circuit_map.png"):
    """Standalone high-quality circuit map (reference-image style)."""
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("white")
    _draw_circuit(ax, title="F1 Race Strategy — Circuit Layout")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_learning_curves(rewards_dict, window=500,
                         save_path="results/learning_curves.png"):
    """rewards_dict: {label: array_of_episode_rewards}"""
    def smooth(x):
        return np.convolve(x, np.ones(window) / window, mode="valid")

    palette = ["#F5A623", "#4A90D9", "#7ED321", "#D0021B"]
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor("white")
    for (label, rewards), color in zip(rewards_dict.items(), palette):
        ax.plot(smooth(rewards), label=label, color=color, lw=1.6)
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Total Reward per Episode", fontsize=10)
    ax.set_title(f"Learning Curves (smoothed, window={window})", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_policy_heatmap(policy, title, traffic=CLEAN_AIR, section=0,
                        save_path=None):
    """Battery (y) × Tire (x) → chosen action, for fixed traffic & section."""
    grid = np.array([[policy[encode(b, t, traffic, section)]
                      for t in range(N_TIRE)]
                     for b in range(N_BATTERY)])

    cmap = plt.cm.get_cmap("tab10", 4)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(N_TIRE));    ax.set_xticklabels(TIRE_NAMES)
    ax.set_yticks(range(N_BATTERY)); ax.set_yticklabels(BATTERY_NAMES)
    ax.set_xlabel("Tire State")
    ax.set_ylabel("Battery State")
    stype = STYPE_NAMES[SECTION_TYPE[section]]
    drs   = " [DRS]" if section in DRS_ZONES else ""
    ax.set_title(
        f"{title}  |  {TRAFFIC_NAMES[traffic]},  "
        f"S{section} {SECTION_NAMES[section]} ({stype}){drs}",
        fontsize=9
    )
    for b in range(N_BATTERY):
        for t in range(N_TIRE):
            ax.text(t, b, ACTION_NAMES[grid[b, t]][0],
                    ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, ticks=range(4))
    cbar.ax.set_yticklabels(ACTION_NAMES)
    plt.tight_layout()
    path = save_path or (f"results/policy_{title.lower().replace(' ','_')}"
                         f"_s{section}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def animate_episode(trajectory, title="", save_path="results/race_animation.gif"):
    cum_rewards = np.cumsum([r for _, _, r in trajectory])

    fig = plt.figure(figsize=(15, 7), facecolor="white")
    ax_t = fig.add_axes([0.01, 0.04, 0.58, 0.92])   # circuit
    ax_i = fig.add_axes([0.62, 0.04, 0.36, 0.92])   # info panel

    # Draw static track elements once
    _draw_circuit(ax_t, title=f"Circuit — {title}")

    # Animated elements
    car,     = ax_t.plot([], [], "D", ms=13, color="red",
                         markeredgecolor="white", markeredgewidth=1.2, zorder=9)
    step_txt = ax_t.text(11.2, 9.0, "", fontsize=9, ha="right", zorder=10)

    ax_i.axis("off")
    ax_i.set_facecolor("#f8f8f8")
    ax_i.set_title("Agent State", fontsize=11, pad=6)
    info = ax_i.text(0.06, 0.95, "", transform=ax_i.transAxes,
                     va="top", fontsize=11, fontfamily="monospace",
                     linespacing=2.1)

    # Sector colour for current section
    def _sector_color(sec):
        for sec_list, color, _ in SECTORS:
            if sec in sec_list:
                return color
        return "white"

    def _bar(v, vmax, w=8):
        n = round(v / vmax * w) if vmax > 0 else 0
        return "█" * n + "░" * (w - n)

    def update(frame):
        s, a, r = trajectory[frame]
        b, t, tr, sec = decode(s)
        stype = SECTION_TYPE[sec]
        drs   = " [DRS]" if sec in DRS_ZONES else ""
        pit   = " [PIT]" if sec == PIT_SECTION else ""

        car.set_data([TRACK_X[sec]], [TRACK_Y[sec]])
        step_txt.set_text(f"Step {frame + 1:3d} / {len(trajectory)}")

        lines = [
            f"┌─────────────────────────┐",
            f"│ Battery  {_bar(b, N_BATTERY-1)}  {BATTERY_NAMES[b]:<8}│",
            f"│ Tire     {_bar(t, N_TIRE-1)}     {TIRE_NAMES[t]:<8}│",
            f"├─────────────────────────┤",
            f"│ Traffic  {TRAFFIC_NAMES[tr]:<16}│",
            f"│ Section  {SECTION_NAMES[sec]} ({STYPE_NAMES[stype]}){drs}{pit:<4}│",
            f"├─────────────────────────┤",
            f"│ Action   {ACTION_NAMES[a]:<16}│",
            f"│ Reward   {r:+8.2f}             │",
            f"│ Cumul.   {cum_rewards[frame]:8.2f}             │",
            f"└─────────────────────────┘",
        ]
        info.set_text("\n".join(lines))
        return car, step_txt, info

    anim = FuncAnimation(fig, update,
                         frames=len(trajectory), interval=400, blit=False)
    anim.save(save_path, writer="pillow", fps=2.5)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    rewards_mc    = np.load("results/rewards_mc.npy")
    rewards_sarsa = np.load("results/rewards_sarsa.npy")
    rewards_ql    = np.load("results/rewards_ql.npy")
    rewards_dql   = np.load("results/rewards_dql.npy")

    policy_vi  = np.load("results/policy_vi.npy")
    policy_pi  = np.load("results/policy_pi.npy")
    policy_mc  = np.argmax(np.load("results/Q_mc.npy"),  axis=1)
    policy_dql = np.argmax(np.load("results/Q_dql.npy"), axis=1)

    with open("results/traj_vi.pkl",  "rb") as f: traj_vi  = pickle.load(f)
    with open("results/traj_dql.pkl", "rb") as f: traj_dql = pickle.load(f)

    plot_circuit_map()

    plot_learning_curves({
        "MC Control":        rewards_mc,
        "SARSA":             rewards_sarsa,
        "Q-learning":        rewards_ql,
        "Double Q-learning": rewards_dql,
    })

    for traffic in (CLEAN_AIR, DIRTY_AIR):
        for section in (1, 3, 6, 7):   # DRS straight, top corner, chicane, apex
            for policy, name in [(policy_vi,  "Value Iteration"),
                                 (policy_pi,  "Policy Iteration"),
                                 (policy_mc,  "MC Control"),
                                 (policy_dql, "Double Q-learning")]:
                plot_policy_heatmap(policy, name, traffic=traffic, section=section)

    animate_episode(traj_vi,  title="Value Iteration",
                    save_path="results/anim_vi.gif")
    animate_episode(traj_dql, title="Double Q-learning",
                    save_path="results/anim_dql.gif")


if __name__ == "__main__":
    main()
