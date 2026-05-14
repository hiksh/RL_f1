import os
import time
import pickle
import numpy as np
from env import F1RaceEnv, N_LAPS
from dp import DPSolver
from mc import MCSolver
from sarsa import SARSASolver
from qlearning import QLearning, DoubleQLearning

RESULTS = "results"


def evaluate(env, policy, n_episodes=1000):
    totals = []
    for _ in range(n_episodes):
        s, done, total = env.reset(), False, 0.0
        while not done:
            s, r, done = env.step(policy[s])
            total += r
        totals.append(total)
    return float(np.mean(totals)), float(np.std(totals))


def collect_episode(env, policy):
    traj = []
    s, done = env.reset(), False
    while not done:
        a = int(policy[s])
        ns, r, done = env.step(a)
        traj.append((s, a, r))
        s = ns
    return traj


def _report(label, t_elapsed, env, policy):
    mean, std = evaluate(env, policy)
    print(f"  Done in {t_elapsed:.2f}s  |  eval reward: {mean:.2f} +/- {std:.2f}")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    env = F1RaceEnv(n_laps=N_LAPS)

    # ------------------------------------------------------------------ DP
    print("Building transition table …")
    dp = DPSolver(env)
    dp._get_P()

    print("\n[DP-1] Value Iteration")
    t0 = time.time()
    V_vi, policy_vi = dp.value_iteration()
    _report("VI", time.time() - t0, env, policy_vi)

    print("\n[DP-2] Policy Iteration")
    t0 = time.time()
    V_pi, policy_pi = dp.policy_iteration()
    _report("PI", time.time() - t0, env, policy_pi)

    # ------------------------------------------------------------------ Model-free
    print("\n[MF-1] Monte Carlo Control  (50 k episodes)")
    t0 = time.time()
    Q_mc, policy_mc, rewards_mc = MCSolver(env).train()
    _report("MC", time.time() - t0, env, policy_mc)

    print("\n[MF-2] SARSA  (50 k episodes)")
    t0 = time.time()
    Q_sarsa, policy_sarsa, rewards_sarsa = SARSASolver(env).train()
    _report("SARSA", time.time() - t0, env, policy_sarsa)

    print("\n[MF-3] Q-learning  (50 k episodes)")
    t0 = time.time()
    Q_ql, policy_ql, rewards_ql = QLearning(env).train()
    _report("QL", time.time() - t0, env, policy_ql)

    print("\n[MF-4] Double Q-learning  (50 k episodes)  ← your solution")
    t0 = time.time()
    Q_dql, policy_dql, rewards_dql = DoubleQLearning(env).train()
    _report("DQL", time.time() - t0, env, policy_dql)

    # ------------------------------------------------------------------ Save
    np.save(f"{RESULTS}/V_vi.npy",           V_vi)
    np.save(f"{RESULTS}/policy_vi.npy",      policy_vi)
    np.save(f"{RESULTS}/V_pi.npy",           V_pi)
    np.save(f"{RESULTS}/policy_pi.npy",      policy_pi)
    np.save(f"{RESULTS}/Q_mc.npy",           Q_mc)
    np.save(f"{RESULTS}/rewards_mc.npy",     np.array(rewards_mc))
    np.save(f"{RESULTS}/Q_sarsa.npy",        Q_sarsa)
    np.save(f"{RESULTS}/rewards_sarsa.npy",  np.array(rewards_sarsa))
    np.save(f"{RESULTS}/Q_ql.npy",           Q_ql)
    np.save(f"{RESULTS}/rewards_ql.npy",     np.array(rewards_ql))
    np.save(f"{RESULTS}/Q_dql.npy",          Q_dql)
    np.save(f"{RESULTS}/rewards_dql.npy",    np.array(rewards_dql))

    for name, policy in [("vi", policy_vi), ("pi", policy_pi),
                          ("mc", policy_mc), ("sarsa", policy_sarsa),
                          ("ql", policy_ql), ("dql", policy_dql)]:
        with open(f"{RESULTS}/traj_{name}.pkl", "wb") as f:
            pickle.dump(collect_episode(env, policy), f)

    print(f"\nAll results saved to {RESULTS}/")


if __name__ == "__main__":
    main()
