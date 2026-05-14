import numpy as np

# ── Dimensions ────────────────────────────────────────────────────────────────
N_BATTERY  = 4    # CRITICAL=0, LOW=1, MEDIUM=2, HIGH=3
N_COMPOUND = 3    # SOFT=0, MED=1, HARD=2
N_TIRE     = 3    # DEGRADED=0, WORN=1, FRESH=2
N_SECTION  = 19
N_LAP      = 5    # lap index 0..4
N_WEATHER  = 2    # DRY=0, RAIN=1

N_STATES       = N_BATTERY * N_COMPOUND * N_TIRE * N_SECTION * N_LAP * N_WEATHER  # 6840
TERMINAL_STATE = N_STATES   # absorbing state used only by DPSolver
N_ACTIONS      = 6

# ── Action constants ──────────────────────────────────────────────────────────
MAINTAIN   = 0;  PUSH       = 1;  RECHARGE   = 2
PIT_SOFT   = 3;  PIT_MEDIUM = 4;  PIT_HARD   = 5

_PIT_ACTIONS = frozenset({PIT_SOFT, PIT_MEDIUM, PIT_HARD})
_PIT_COMPOUND = {PIT_SOFT: 0, PIT_MEDIUM: 1, PIT_HARD: 2}

# ── State constants ───────────────────────────────────────────────────────────
CRITICAL = 0;  LOW  = 1;  MEDIUM = 2;  HIGH = 3   # battery
SOFT     = 0;  MED  = 1;  HARD   = 2              # compound (MED≠MEDIUM: different dims)
DEGRADED = 0;  WORN = 1;  FRESH  = 2              # tire wear
DRY      = 0;  RAIN = 1                           # weather
STRAIGHT = 0;  CORNER = 1;  CHICANE = 2           # section type

# 19-section layout (clockwise, C01-C19)
SECTION_TYPE = [
    CORNER,    # S0  C01 hairpin bottom-left
    STRAIGHT,  # S1  C02 left DRS straight       ← DRS zone 1
    CORNER,    # S2  C03 upper-left turn
    CORNER,    # S3  C04 top-left
    CORNER,    # S4  C05 top-centre
    CORNER,    # S5  C06 top
    CHICANE,   # S6  C07 top-right chicane
    CORNER,    # S7  C08 far-right apex
    CORNER,    # S8  C09 right-side
    STRAIGHT,  # S9  C10 inner straight
    CORNER,    # S10 C11 centre corner
    CHICANE,   # S11 C12 right chicane
    CORNER,    # S12 C13 lower-right
    CORNER,    # S13 C14 far lower-right
    CORNER,    # S14 C15 bottom-right
    STRAIGHT,  # S15 C16 bottom centre
    STRAIGHT,  # S16 C17 bottom DRS straight     ← DRS zone 2
    STRAIGHT,  # S17 C18 start/finish (PIT)
    CORNER,    # S18 C19 hairpin exit
]

DRS_ZONES   = frozenset({1, 16})
PIT_SECTION = 17
N_LAPS      = 5

# ── Tire wear base probabilities (baseline = MED compound, dry) ───────────────
_WEAR_P = {
    (STRAIGHT, MAINTAIN): 0.01, (STRAIGHT, PUSH): 0.04, (STRAIGHT, RECHARGE): 0.00,
    (CORNER,   MAINTAIN): 0.04, (CORNER,   PUSH): 0.12, (CORNER,   RECHARGE): 0.02,
    (CHICANE,  MAINTAIN): 0.06, (CHICANE,  PUSH): 0.20, (CHICANE,  RECHARGE): 0.03,
}
_COMPOUND_WEAR_MULT = {SOFT: 1.5, MED: 1.0, HARD: 0.6}

# ── Battery (ERS) transitions ─────────────────────────────────────────────────
_BATTERY_P = {
    (STRAIGHT, MAINTAIN):  (-1, 0.30),
    (STRAIGHT, PUSH):      (-1, 0.80),
    (STRAIGHT, RECHARGE):  (+1, 0.10),
    (CORNER,   MAINTAIN):  (+1, 0.15),
    (CORNER,   PUSH):      (-1, 0.20),
    (CORNER,   RECHARGE):  (+1, 0.25),
    (CHICANE,  MAINTAIN):  (+1, 0.15),
    (CHICANE,  PUSH):      (-1, 0.15),
    (CHICANE,  RECHARGE):  (+1, 0.25),
}

# ── Base traversal time (seconds, lower = faster) ─────────────────────────────
_BASE_TIME = {
    (STRAIGHT, MAINTAIN): 1.0,  (STRAIGHT, PUSH): 0.6,  (STRAIGHT, RECHARGE): 1.3,
    (CORNER,   MAINTAIN): 1.1,  (CORNER,   PUSH): 0.85, (CORNER,   RECHARGE): 1.4,
    (CHICANE,  MAINTAIN): 1.2,  (CHICANE,  PUSH): 1.0,  (CHICANE,  RECHARGE): 1.5,
}

DRS_TIME_BONUS = 0.2

# ── Weather transitions (per lap, applied when crossing section 18→0) ─────────
# P(DRY→RAIN)=0.08, P(RAIN→DRY)=0.18  →  stationary π_RAIN ≈ 0.31
_WEATHER_CHANGE_P = {DRY: 0.08, RAIN: 0.18}
_WEATHER_NEXT     = {DRY: RAIN, RAIN: DRY}


def encode(battery, compound, tire, section, lap, weather):
    return (((((battery * N_COMPOUND + compound) * N_TIRE + tire)
              * N_SECTION + section) * N_LAP + lap) * N_WEATHER + weather)


def decode(state):
    weather  = state % N_WEATHER;  state //= N_WEATHER
    lap      = state % N_LAP;      state //= N_LAP
    section  = state % N_SECTION;  state //= N_SECTION
    tire     = state % N_TIRE;     state //= N_TIRE
    compound = state % N_COMPOUND; battery = state // N_COMPOUND
    return battery, compound, tire, section, lap, weather


class F1RaceEnv:
    def __init__(self, n_laps=N_LAPS):
        self.n_laps  = n_laps
        self.states  = list(range(N_STATES))
        self.actions = list(range(N_ACTIONS))
        self.state   = encode(HIGH, MED, FRESH, 0, 0, DRY)

    # ------------------------------------------------------------------
    # Gym-style interface
    # ------------------------------------------------------------------

    def reset(self):
        self.state = encode(HIGH, MED, FRESH, 0, 0, DRY)
        return self.state

    def step(self, action):
        b, comp, t, sec, lap, w = decode(self.state)
        stype = SECTION_TYPE[sec]
        eff_a = action if (action not in _PIT_ACTIONS or sec == PIT_SECTION) else MAINTAIN

        nb         = self._sample_battery(b, eff_a, stype)
        ncomp, nt  = self._sample_tire(t, comp, eff_a, stype, w)
        ns         = (sec + 1) % N_SECTION
        r          = self._calc_reward(b, comp, t, w, eff_a, stype, sec)

        if ns == 0:
            nl = lap + 1
            nw = self._sample_weather(w)
        else:
            nl, nw = lap, w

        done = (nl >= self.n_laps)
        self.state = encode(nb, ncomp, nt, ns, min(nl, N_LAP - 1), nw)
        return self.state, r, done

    # ------------------------------------------------------------------
    # GridWorld-style interface (used by DPSolver)
    # ------------------------------------------------------------------

    def transition_prob(self, s_next, state, action):
        b, comp, t, sec, lap, w = decode(state)
        stype = SECTION_TYPE[sec]
        eff_a = action if (action not in _PIT_ACTIONS or sec == PIT_SECTION) else MAINTAIN
        ns    = (sec + 1) % N_SECTION

        prob = 0.0
        w_out = self._weather_outcomes(w) if ns == 0 else [(1.0, w)]
        for p_w, nw in w_out:
            nl = lap + 1 if ns == 0 else lap
            if nl >= N_LAP:
                continue
            for p_b, nb in self._battery_outcomes(b, eff_a, stype):
                for p_t, nc, nt in self._tire_outcomes(t, comp, eff_a, stype, w):
                    if encode(nb, nc, nt, ns, nl, nw) == s_next:
                        prob += p_w * p_b * p_t
        return prob

    def reward(self, state, action, s_next=None):
        b, comp, t, sec, lap, w = decode(state)
        stype = SECTION_TYPE[sec]
        eff_a = action if (action not in _PIT_ACTIONS or sec == PIT_SECTION) else MAINTAIN
        return self._calc_reward(b, comp, t, w, eff_a, stype, sec)

    # ------------------------------------------------------------------
    # Pre-computed transition table for DP
    # ------------------------------------------------------------------

    def build_transition_table(self):
        """Returns P[s][a] = list of (prob, next_state, reward).
        Terminal transitions point to TERMINAL_STATE (index N_STATES)."""
        P = [[[] for _ in range(N_ACTIONS)] for _ in range(N_STATES)]
        for s in range(N_STATES):
            b, comp, t, sec, lap, w = decode(s)
            stype = SECTION_TYPE[sec]
            for a in range(N_ACTIONS):
                eff_a = a if (a not in _PIT_ACTIONS or sec == PIT_SECTION) else MAINTAIN
                ns    = (sec + 1) % N_SECTION
                r     = self._calc_reward(b, comp, t, w, eff_a, stype, sec)
                w_out = self._weather_outcomes(w) if ns == 0 else [(1.0, w)]
                for p_w, nw in w_out:
                    nl = lap + 1 if ns == 0 else lap
                    for p_b, nb in self._battery_outcomes(b, eff_a, stype):
                        for p_t, nc, nt in self._tire_outcomes(t, comp, eff_a, stype, w):
                            tp = p_w * p_b * p_t
                            if nl >= N_LAP:
                                P[s][a].append((tp, TERMINAL_STATE, r))
                            else:
                                P[s][a].append((tp, encode(nb, nc, nt, ns, nl, nw), r))
        return P

    # ------------------------------------------------------------------
    # Battery helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_battery(battery, action, stype):
        if action in _PIT_ACTIONS:
            return battery
        direction, p = _BATTERY_P.get((stype, action), (0, 0.0))
        if p == 0.0 or np.random.random() >= p:
            return battery
        return max(CRITICAL, min(HIGH, battery + direction))

    @staticmethod
    def _battery_outcomes(battery, action, stype):
        if action in _PIT_ACTIONS:
            return [(1.0, battery)]
        direction, p = _BATTERY_P.get((stype, action), (0, 0.0))
        if p == 0.0:
            return [(1.0, battery)]
        new_bat = max(CRITICAL, min(HIGH, battery + direction))
        if new_bat == battery:
            return [(1.0, battery)]
        return [(1.0 - p, battery), (p, new_bat)]

    # ------------------------------------------------------------------
    # Tire helpers  (returns compound, tire)
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_tire(tire, compound, action, stype, weather):
        if action in _PIT_ACTIONS:
            return _PIT_COMPOUND[action], FRESH
        if tire == DEGRADED:
            return compound, DEGRADED
        base_p = _WEAR_P.get((stype, action), 0.0)
        p = min(0.99, base_p * _COMPOUND_WEAR_MULT[compound] * (2.0 if weather == RAIN else 1.0))
        if p > 0 and np.random.random() < p:
            return compound, tire - 1
        return compound, tire

    @staticmethod
    def _tire_outcomes(tire, compound, action, stype, weather):
        """Returns [(prob, next_compound, next_tire), ...]"""
        if action in _PIT_ACTIONS:
            return [(1.0, _PIT_COMPOUND[action], FRESH)]
        if tire == DEGRADED:
            return [(1.0, compound, DEGRADED)]
        base_p = _WEAR_P.get((stype, action), 0.0)
        p = min(0.99, base_p * _COMPOUND_WEAR_MULT[compound] * (2.0 if weather == RAIN else 1.0))
        if p == 0.0:
            return [(1.0, compound, tire)]
        return [(1.0 - p, compound, tire), (p, compound, tire - 1)]

    # ------------------------------------------------------------------
    # Weather helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_weather(weather):
        p = _WEATHER_CHANGE_P[weather]
        return _WEATHER_NEXT[weather] if np.random.random() < p else weather

    @staticmethod
    def _weather_outcomes(weather):
        p = _WEATHER_CHANGE_P[weather]
        nw = _WEATHER_NEXT[weather]
        return [(1.0 - p, weather), (p, nw)]

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_reward(battery, compound, tire, weather, action, stype, section):
        if action in _PIT_ACTIONS:
            return -5.0
        cost = _BASE_TIME[(stype, action)]
        if action == PUSH and section in DRS_ZONES:
            cost -= DRS_TIME_BONUS
        if compound == SOFT:
            cost -= 0.15
        elif compound == HARD:
            cost += 0.10
        if weather == RAIN:
            cost += 1.0
        if tire    == DEGRADED:  cost += 0.5
        if battery == CRITICAL:  cost += 0.3
        return -cost
