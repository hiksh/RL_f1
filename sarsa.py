import numpy as np
from env import N_STATES, N_ACTIONS


class SARSASolver:
    """On-policy TD control (SARSA)."""

    def __init__(self, env, gamma=0.99, alpha=0.1, epsilon=1.0,
                 eps_decay=0.9995, eps_min=0.05):
        self.env       = env
        self.gamma     = gamma
        self.alpha     = alpha
        self.epsilon   = epsilon
        self.eps_decay = eps_decay
        self.eps_min   = eps_min

    def _eps_greedy(self, Q, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(N_ACTIONS)
        return int(np.argmax(Q[state]))

    def train(self, n_episodes=50_000):
        Q = np.zeros((N_STATES, N_ACTIONS))
        rewards_log = []
        epsilon = self.epsilon

        for _ in range(n_episodes):
            state, done, total = self.env.reset(), False, 0.0
            action = self._eps_greedy(Q, state, epsilon)
            while not done:
                ns, r, done    = self.env.step(action)
                next_action    = 0 if done else self._eps_greedy(Q, ns, epsilon)
                future_q       = 0.0 if done else Q[ns, next_action]
                Q[state, action] += self.alpha * (
                    r + self.gamma * future_q - Q[state, action]
                )
                state, action, total = ns, next_action, total + r
            rewards_log.append(total)
            epsilon = max(self.eps_min, epsilon * self.eps_decay)

        return Q, np.argmax(Q, axis=1), rewards_log
