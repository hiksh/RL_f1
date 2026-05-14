import numpy as np
from env import N_STATES, N_ACTIONS


class DPSolver:
    def __init__(self, env, gamma=0.99, theta=1e-6):
        self.env   = env
        self.gamma = gamma
        self.theta = theta
        self._P    = None

    def _get_P(self):
        if self._P is None:
            self._P = self.env.build_transition_table()
        return self._P

    def one_step_lookahead(self, V, state):
        P = self._get_P()
        return np.array([
            sum(p * (r + self.gamma * V[ns]) for p, ns, r in P[state][a])
            for a in range(N_ACTIONS)
        ])

    def policy_evaluation(self, policy):
        P = self._get_P()
        V = np.zeros(N_STATES + 1)   # index N_STATES = TERMINAL_STATE, always 0
        while True:
            delta = 0.0
            for s in range(N_STATES):
                a     = int(policy[s])
                v_new = sum(p * (r + self.gamma * V[ns]) for p, ns, r in P[s][a])
                delta = max(delta, abs(V[s] - v_new))
                V[s]  = v_new
            if delta < self.theta:
                break
        return V

    def value_iteration(self):
        V = np.zeros(N_STATES + 1)   # +1 for TERMINAL_STATE
        while True:
            delta = 0.0
            for s in range(N_STATES):
                q     = self.one_step_lookahead(V, s)
                v_new = q.max()
                delta = max(delta, abs(V[s] - v_new))
                V[s]  = v_new
            if delta < self.theta:
                break
        policy = np.array([self.one_step_lookahead(V, s).argmax()
                           for s in range(N_STATES)])
        return V[:N_STATES], policy

    def policy_iteration(self):
        policy = np.zeros(N_STATES, dtype=int)
        while True:
            V          = self.policy_evaluation(policy)
            new_policy = np.array([self.one_step_lookahead(V, s).argmax()
                                   for s in range(N_STATES)])
            if np.array_equal(new_policy, policy):
                break
            policy = new_policy
        return self.policy_evaluation(policy)[:N_STATES], policy
