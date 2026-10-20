"""OLS autoregression with a common fitting window across candidate orders."""
import numpy as np

from oscillator import state_at


def observe(duration, interval, noise, seed, params):
    t = np.arange(int(np.floor(duration / interval + 1e-9)) + 1) * interval
    truth = np.array([state_at(float(ti), *params)[0] for ti in t])
    return t, truth, truth + np.random.default_rng(seed).normal(0, noise, len(t))


def design(y, p, start, stop, intercept):
    columns = [y[start - j:stop - j] for j in range(1, p + 1)]
    if intercept:
        columns.insert(0, np.ones(stop - start))
    return np.column_stack(columns) if columns else np.empty((stop - start, 0))


def rmse(y, pred):
    if not np.all(np.isfinite(pred)):
        return float('inf')
    with np.errstate(over='ignore', invalid='ignore'):
        return float(np.sqrt(np.mean((np.asarray(y) - pred) ** 2)))


def fit_orders(y, fraction, max_order, intercept=False):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 12:
        raise ValueError('少なくとも12点の記録が必要です。再生を進めるか、記録周期を短くしてください。')
    split = min(n - 3, max(6, int(n * fraction)))
    # Leave residual degrees of freedom even for the largest candidate.
    pmax = min(max_order, (split - int(intercept) - 2) // 2)
    if pmax < 1:
        raise ValueError('学習用の記録が不足しています。')
    start = pmax
    target = y[start:split]
    rows = []
    for p in range(pmax + 1):
        X = design(y, p, start, split, intercept)
        coef, _, rank, _ = np.linalg.lstsq(X, target, rcond=None)
        if rank < X.shape[1]:
            continue
        fitted = X @ coef
        mse = max(float(np.mean((target - fitted) ** 2)), 1e-24)
        # Conditional Gaussian IC, omitting constants common to every candidate.
        # Count the innovation variance as one parameter.
        count, num = len(target), len(coef) + 1
        aic = count * np.log(mse) + 2 * num
        bic = count * np.log(mse) + np.log(count) * num
        one_step = design(y, p, split, n, intercept) @ coef
        history = list(y[:split])
        bias = coef[0] if intercept else 0.0
        weights = coef[1:] if intercept else coef
        recursive = []
        for _ in range(split, n):
            with np.errstate(over='ignore', invalid='ignore'):
                value = float(bias + np.dot(weights, history[-p:][::-1])) if p else float(bias)
            if not np.isfinite(value) or abs(value) > 1e100:
                recursive.extend([float('nan')] * (n - len(history)))
                break
            recursive.append(value)
            history.append(value)
        recursive = np.asarray(recursive)
        rows.append(dict(p=p, coef=coef, fitted=fitted, one_step=one_step,
                         recursive=recursive, train_rmse=rmse(target, fitted),
                         validation_rmse=rmse(y[split:], one_step),
                         recursive_rmse=rmse(y[split:], recursive),
                         aic=float(aic), bic=float(bic)))
    if not rows:
        raise ValueError('記録の情報が不足し、候補モデルを推定できませんでした。')
    return dict(rows=rows, split=split, start=start, pmax=pmax, n=n)
