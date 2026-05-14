import numpy as np
from env import N_STATES, N_ACTIONS


class QLearning:
    """Off-policy TD control (Q-learning)."""

    def __init__(self, env, gamma=0.99, alpha=0.1, epsilon=1.0,
                 eps_decay=0.9995, eps_min=0.05):
        self.env       = env
        self.gamma     = gamma
        self.alpha     = alpha
        self.epsilon   = epsilon
        self.eps_decay = eps_decay
        self.eps_min   = eps_min

    def train(self, n_episodes=50_000):
        Q = np.zeros((N_STATES, N_ACTIONS))
        rewards_log = []
        epsilon = self.epsilon

        for _ in range(n_episodes):
            state, done, total = self.env.reset(), False, 0.0
            while not done:
                a = (np.random.randint(N_ACTIONS) if np.random.random() < epsilon
                     else int(np.argmax(Q[state])))
                ns, r, done = self.env.step(a)
                future_q = 0.0 if done else np.max(Q[ns])
                Q[state, a] += self.alpha * (
                    r + self.gamma * future_q - Q[state, a]
                )
                state, total = ns, total + r
            rewards_log.append(total)
            epsilon = max(self.eps_min, epsilon * self.eps_decay)

        return Q, np.argmax(Q, axis=1), rewards_log


class DoubleQLearning:
    """Double Q-learning: decouples action selection and value estimation
    across two Q-tables to reduce maximisation bias."""

    def __init__(self, env, gamma=0.99, alpha=0.1, epsilon=1.0,
                 eps_decay=0.9995, eps_min=0.05):
        self.env       = env
        self.gamma     = gamma
        self.alpha     = alpha
        self.epsilon   = epsilon
        self.eps_decay = eps_decay
        self.eps_min   = eps_min

    def train(self, n_episodes=50_000):
        Qa = np.zeros((N_STATES, N_ACTIONS))
        Qb = np.zeros((N_STATES, N_ACTIONS))
        rewards_log = []
        epsilon = self.epsilon

        for _ in range(n_episodes):
            state, done, total = self.env.reset(), False, 0.0
            while not done:
                a = (np.random.randint(N_ACTIONS) if np.random.random() < epsilon
                     else int(np.argmax(Qa[state] + Qb[state])))
                ns, r, done = self.env.step(a)
                if np.random.random() < 0.5:
                    best     = int(np.argmax(Qa[ns])) if not done else 0
                    future_q = 0.0 if done else Qb[ns, best]
                    Qa[state, a] += self.alpha * (
                        r + self.gamma * future_q - Qa[state, a]
                    )
                else:
                    best     = int(np.argmax(Qb[ns])) if not done else 0
                    future_q = 0.0 if done else Qa[ns, best]
                    Qb[state, a] += self.alpha * (
                        r + self.gamma * future_q - Qb[state, a]
                    )
                state, total = ns, total + r
            rewards_log.append(total)
            epsilon = max(self.eps_min, epsilon * self.eps_decay)

        Q = (Qa + Qb) / 2
        return Q, np.argmax(Q, axis=1), rewards_log
