"""Functional cortico-cerebellar demonstration, NumPy only.

M1-like echo-state RNN: fixed recurrent connections, supervised fitted readout.
Shared fixed feature maps and supervised base readout. Adaptation is defined
in distributed_simulation.py. Cerebellar feedback enters the RNN, not the plant.
This functional model does not reproduce biological cells or source experiments.
"""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import hashlib
import numpy as np

DT, DURATION, DISTANCE = .01, .8, .1
SPEED_SCALE = DISTANCE / DURATION
N = round(DURATION / DT)
PREP = 10
K = 5.0
HIDDEN, FEATURES = 64, 48


def rotation(angle):
    a = np.deg2rad(angle)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def reference(direction):
    t = np.linspace(0, 1, N + 1)
    position = (10*t**3 - 15*t**4 + 6*t**5)[:, None] * direction
    velocity = np.diff(position, axis=0) * DURATION / DT
    return position, velocity


class CoupledModel:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.recurrent = self.rng.normal(0, 1/np.sqrt(HIDDEN), (HIDDEN, HIDDEN))
        self.recurrent *= .45 / np.max(np.abs(np.linalg.eigvals(self.recurrent)))
        self.input = self.rng.normal(0, .12, (HIDDEN, 8))
        self.bias = self.rng.normal(0, .015, HIDDEN)
        self.readout = np.zeros((2*HIDDEN+1, 2))
        self.expansion = self.rng.normal(0, .8, (FEATURES, 7))
        self.expansion_bias = self.rng.normal(0, .3, FEATURES)
        self.hidden_projection = self.rng.normal(0, 1/np.sqrt(HIDDEN), (2, HIDDEN))
        self.cerebellar = np.zeros((FEATURES, 2))

    def cortex_step(self, hidden, planned_velocity, feedback_error, direction, phase, go):
        inp = np.r_[planned_velocity, feedback_error, direction, phase, go]
        new = .3*hidden + .7*np.tanh(self.recurrent@hidden + self.input@inp + self.bias)
        feature = np.r_[new, hidden, 1.0]
        return new, (feature@self.readout)*go, feature

    def cb_features(self, hidden, direction, velocity, phase):
        # MF-like contextual input includes a low-dimensional copy of cortical state.
        context = np.r_[direction, velocity, phase, self.hidden_projection@hidden]
        fixed = np.tanh(self.expansion@context + self.expansion_bias)/np.sqrt(FEATURES)
        # Gating by intended speed keeps stationary output zero; a preview is
        # supplied during preparation, while the plant is held by the Go gate.
        return fixed*np.linalg.norm(velocity)

    def train_cortex(self, examples=360):
        gram = np.zeros((2*HIDDEN+1, 2*HIDDEN+1))
        rhs = np.zeros((2*HIDDEN+1, 2))
        # Train on a range of target directions, changing planned commands and
        # smooth delayed-error inputs. Teacher is a conventional kinematic controller.
        # This is readout-only supervised learning, NOT BPTT/recurrent plasticity.
        for example in range(examples):
            angle = self.rng.uniform(-np.pi, np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])
            _, velocities = reference(direction)
            hidden = np.zeros(HIDDEN)
            error = np.zeros(2)
            correction = self.rng.normal(0, .4, 2)
            seq_f, seq_y = [], []
            for i in range(-PREP, N):
                go = float(i >= 0)
                v = velocities[i] if go else direction*.8
                phase = max(i, 0)/N
                error = .8*error + self.rng.normal(0, .03, 2) if go else np.zeros(2)
                correction = .85*correction + self.rng.normal(0, .09, 2)
                planned = v+correction
                hidden, _, feature = self.cortex_step(hidden, planned, error, direction, phase, go)
                seq_f.append(feature)
                # Desired normalized velocity before Go gating.
                seq_y.append(planned+K*DURATION*error)
            X, Y = np.array(seq_f), np.array(seq_y)
            gram += X.T@X
            rhs += X.T@Y
        self.readout = np.linalg.solve(gram+1e-4*np.eye(gram.shape[0]), rhs)
        return self.validate_cortex()

    def validate_cortex(self):
        errors = []
        rng = np.random.default_rng(self.seed+90000)
        for _ in range(30):
            direction = rotation(rng.uniform(-180, 180))@np.array([1., 0.])
            _, vv = reference(direction)
            h = np.zeros(HIDDEN)
            e = np.zeros(2)
            for i in range(-PREP, N):
                go = float(i >= 0)
                v = vv[i] if go else direction*.8
                e = .8*e+rng.normal(0, .03, 2) if go else np.zeros(2)
                c = rng.normal(0, .15, 2)
                h, output, _ = self.cortex_step(h, v+c, e, direction, max(i, 0)/N, go)
                if go:
                    errors.append(output-(v+c+K*DURATION*e))
        return float(np.sqrt(np.mean(np.array(errors)**2))*SPEED_SCALE*1000)


def digest(arr):
    return hashlib.sha256(arr.tobytes()).hexdigest()
