import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

results = r"C:\Users\Hiksh\Desktop\College\3rd\Reinforcement Learning\pj\f1\results"

V_vi = np.load(f"{results}/V_vi.npy")
V_pi = np.load(f"{results}/V_pi.npy")
p_vi = np.load(f"{results}/policy_vi.npy")
p_pi = np.load(f"{results}/policy_pi.npy")
Q_mc    = np.load(f"{results}/Q_mc.npy")
Q_sarsa = np.load(f"{results}/Q_sarsa.npy")
Q_ql    = np.load(f"{results}/Q_ql.npy")
Q_dql   = np.load(f"{results}/Q_dql.npy")
r_mc    = np.load(f"{results}/rewards_mc.npy")
r_sarsa = np.load(f"{results}/rewards_sarsa.npy")
r_ql    = np.load(f"{results}/rewards_ql.npy")
r_dql   = np.load(f"{results}/rewards_dql.npy")

action_names = ['Maintain','Push','Recharge','Pit-Soft','Pit-Med','Pit-Hard']
policies = {
    'VI':    p_vi,
    'PI':    p_pi,
    'MC':    np.argmax(Q_mc,    axis=1),
    'SARSA': np.argmax(Q_sarsa, axis=1),
    'QL':    np.argmax(Q_ql,    axis=1),
    'DQL':   np.argmax(Q_dql,   axis=1),
}

print("=== DP Value Stats ===")
print(f"V_vi: mean={V_vi.mean():.3f}, std={V_vi.std():.3f}")
print(f"V_pi: mean={V_pi.mean():.3f}, std={V_pi.std():.3f}")
print(f"VI==PI policy agreement: {(p_vi==p_pi).mean()*100:.1f}%")

print("\n=== Reward Stats (50k training episodes) ===")
for name, arr in [('MC', r_mc), ('SARSA', r_sarsa), ('QL', r_ql), ('DQL', r_dql)]:
    print(f"{name}: first1k={arr[:1000].mean():.2f}, last1k={arr[-1000:].mean():.2f}, "
          f"overall={arr.mean():.2f}, std={arr.std():.2f}, min={arr.min():.2f}, max={arr.max():.2f}")

print("\n=== Action Distribution (% of all 6840 states) ===")
for pname, pol in policies.items():
    counts = np.bincount(pol, minlength=6)
    pcts = counts / len(pol) * 100
    parts = [f"{a}:{p:.1f}%" for a, p in zip(action_names, pcts)]
    print(f"  {pname}: {' | '.join(parts)}")

print("\n=== Policy Agreement (model-free vs DP) ===")
for mname in ['MC', 'SARSA', 'QL', 'DQL']:
    agree_vi = (policies['VI'] == policies[mname]).mean() * 100
    agree_pi = (policies['PI'] == policies[mname]).mean() * 100
    print(f"  {mname} vs VI: {agree_vi:.1f}% | vs PI: {agree_pi:.1f}%")

print("\n=== Q-table Coverage ===")
for name, Q in [('MC', Q_mc), ('SARSA', Q_sarsa), ('QL', Q_ql), ('DQL', Q_dql)]:
    nz = np.count_nonzero(Q)
    total = Q.size
    print(f"  {name}: {nz}/{total} nonzero ({100*nz/total:.1f}%)")

# Convergence windows (500-ep smoothed)
w = 500
print("\n=== Smoothed reward at convergence (last 500 ep window) ===")
for name, arr in [('MC', r_mc), ('SARSA', r_sarsa), ('QL', r_ql), ('DQL', r_dql)]:
    smoothed = np.convolve(arr, np.ones(w)/w, mode='valid')
    print(f"  {name}: peak={smoothed.max():.2f} (ep {smoothed.argmax()}), final={smoothed[-1]:.2f}")
