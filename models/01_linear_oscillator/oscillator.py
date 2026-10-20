"""Exact solution of m*x'' + c*x' + k*x = 0 (standard library only)."""
import math


def state_at(t, m, k, c, x0, v0):
    a = c / (2 * m)
    w2 = k / m
    d = a * a - w2
    if abs(d) <= 1e-12 * max(a * a, w2):
        b = v0 + a * x0
        e = math.exp(-a * t)
        return e * (x0 + b * t), e * (v0 - a * b * t)
    if d < 0:
        w = math.sqrt(-d)
        b = (v0 + a * x0) / w
        co, si, e = math.cos(w * t), math.sin(w * t), math.exp(-a * t)
        q = x0 * co + b * si
        return e * q, e * (-a * q - x0 * w * si + b * w * co)
    s = math.sqrt(d)
    # This expression avoids subtracting two nearly equal numbers.
    r1, r2 = -w2 / (a + s), -a - s
    b1 = (v0 - r2 * x0) / (r1 - r2)
    b2 = x0 - b1
    e1, e2 = math.exp(r1 * t), math.exp(r2 * t)
    return b1 * e1 + b2 * e2, r1 * b1 * e1 + r2 * b2 * e2
