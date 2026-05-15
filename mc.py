import numpy as np
from env import N_STATES, N_ACTIONS


class MCSolver:
    def __init__(self, env, gamma=0.99, epsilon=1.0,
                 eps_decay=0.99993, eps_min=0.05):
        self.env       = env
        self.gamma     = gamma
        self.epsilon   = epsilon
        self.eps_decay = eps_decay
        self.eps_min   = eps_min

    def train(self, n_episodes=50_000):
        Q = np.zeros((N_STATES, N_ACTIONS))
        N = np.zeros((N_STATES, N_ACTIONS))
        rewards_log = []
        epsilon = self.epsilon

        for _ in range(n_episodes):
            episode = []
            state, done = self.env.reset(), False
            while not done:
                a = (np.random.randint(N_ACTIONS) if np.random.random() < epsilon
                     else int(np.argmax(Q[state])))
                ns, r, done = self.env.step(a)
                episode.append((state, a, r))
                state = ns

            G = 0.0
            for s, a, r in reversed(episode):
                G       = r + self.gamma * G
                N[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / N[s, a]

            rewards_log.append(sum(x[2] for x in episode))
            epsilon = max(self.eps_min, epsilon * self.eps_decay)

        return Q, np.argmax(Q, axis=1), rewards_log
