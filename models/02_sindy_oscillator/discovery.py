"""Teaching implementation of sequential thresholded least squares (STLSQ)."""
import math
from functools import lru_cache
import numpy as np
from oscillator import state_at as linear_state_at


NAMES = ['1', 'x', 'v', 'x²', 'xv', 'v²', 'x³', 'x²v', 'xv²', 'v³']
POWERS = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]


def library(x, v, degree):
    x, v = np.asarray(x), np.asarray(v)
    count = {1: 3, 2: 6, 3: 10}[degree]
    with np.errstate(over='ignore', invalid='ignore'):
        return np.stack([x**a * v**b for a, b in POWERS[:count]], axis=-1)


def records(duration, dt, noise, seed, params):
    t = np.arange(int(np.floor(duration / dt + 1e-9)) + 1) * dt
    states = np.array([true_state_at(float(ti), *params) for ti in t])
    a = -(params[2] * states[:, 1] + params[1] * states[:, 0] + (params[5] if len(params)>5 else 0.0)*states[:,0]**2) / params[0]
    observed = states[:, 0] + np.random.default_rng(seed).normal(0, noise, len(t))
    return t, states[:, 0], states[:, 1], a, observed


def derivative_data(t, x, v, a, observed, split, mode, window):
    # Only training data enter derivative estimation. No boundary leakage.
    if mode == 'ideal':
        return t[:split], x[:split], v[:split], a[:split], 0
    dt = float(t[1] - t[0])
    y = np.asarray(observed[:split])
    if mode == 'difference':
        return t[1:split-1], y[1:-1], (y[2:]-y[:-2])/(2*dt), (y[2:]-2*y[1:-1]+y[:-2])/dt**2, 1
    if len(y) < window:
        raise ValueError(f'平滑化には学習区間に{window}点以上が必要です。記録を増やすか窓幅を小さくしてください。')
    half = window // 2
    u = np.arange(-half, half+1) * dt
    # Symmetric cubic local polynomial regression, evaluating at each center.
    matrix = np.column_stack([np.ones(window), u, u**2, u**3])
    transform = np.linalg.pinv(matrix)
    coeff = np.array([transform @ y[i-half:i+half+1] for i in range(half, len(y)-half)])
    return t[half:split-half], coeff[:, 0], coeff[:, 1], 2*coeff[:, 2], half


def stlsq(x, v, a, degree, threshold):
    theta = library(x, v, degree)
    n, count = theta.shape
    if n < count + 3:
        raise ValueError(f'有効な学習点が{n}点です。{count+3}点以上を用意するか、候補項を減らしてください。')
    rms = np.sqrt(np.mean(theta**2, axis=0))
    target_scale = float(np.sqrt(np.mean(a**2)))
    if target_scale < 1e-12:
        raise ValueError('加速度がほぼ0で、運動の関係を推定する情報が不足しています。初期変位や初期速度を変えてください。')
    active = rms > 1e-12
    safe_scale = np.where(active, rms, 1.0)
    X, y = theta / safe_scale, np.asarray(a) / target_scale
    initial_rank = int(np.linalg.matrix_rank(X[:, active]))
    cond = float(np.linalg.cond(X[:, active]))
    coef = np.zeros(count)
    coef[active] = np.linalg.lstsq(X[:, active], y, rcond=None)[0]
    history = [coef.copy()]
    for _ in range(count + 1):
        keep = active & (np.abs(coef) >= threshold)
        if np.array_equal(keep, active):
            break
        active = keep
        coef[:] = 0
        if active.any():
            coef[active] = np.linalg.lstsq(X[:, active], y, rcond=None)[0]
        history.append(coef.copy())
    physical = coef * target_scale / safe_scale
    return dict(coef=physical, scaled=coef, history=history,
                physical_history=[h * target_scale / safe_scale for h in history],
                rank=initial_rank, condition=cond, columns=count,
                fit_rmse=float(np.sqrt(np.mean((theta @ physical-a)**2))))


def rollout(times, initial, coef, degree):
    """RK4 with substeps; bound runaway models rather than silently clipping."""
    output = np.full((len(times), 2), np.nan)
    output[0] = initial
    def rhs(state):
        return np.array([state[1], float(library(state[0], state[1], degree) @ coef)])
    for i in range(1, len(times)):
        state = output[i-1].copy()
        steps = max(1, math.ceil(float(times[i]-times[i-1]) / .005))
        h = float(times[i]-times[i-1]) / steps
        for _ in range(steps):
            with np.errstate(over='ignore', invalid='ignore'):
                k1 = rhs(state)
                k2 = rhs(state+h*k1/2)
                k3 = rhs(state+h*k2/2)
                k4 = rhs(state+h*k3)
                state += h*(k1+2*k2+2*k3+k4)/6
            if not np.all(np.isfinite(state)) or np.max(np.abs(state)) > 1e6:
                return output, False
        output[i] = state
    return output, True


@lru_cache(maxsize=16)
def true_path(params):
    m,k,c,x0,v0,q,duration = params
    times = np.linspace(0,duration,math.ceil(duration/.005)+1)
    if q == 0:
        states = np.array([linear_state_at(float(t),m,k,c,x0,v0) for t in times])
    else:
        coef = np.array([0.,-k/m,-c/m,-q/m,0.,0.])
        states,success = rollout(times,[x0,v0],coef,2)
        if not success:
            raise ValueError('この条件では真の二次ばねモデルの運動が非有界となり、計算を継続できません。二次係数・初期変位・初期速度を小さくするか、ばね定数を大きくしてください。')
    return times,states


def true_state_at(t,*params):
    if len(params)==5 or params[5]==0:
        return linear_state_at(t,*params[:5])
    times,states = true_path(tuple(params))
    return float(np.interp(t,times,states[:,0])),float(np.interp(t,times,states[:,1]))
