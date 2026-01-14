from __future__ import annotations
import numpy as np


# ---------- 1) Membership functions ----------
def tri(x: np.ndarray | float, a: float, b: float, c: float) -> np.ndarray | float:
    """
    Triangular membership function.
    a <= b <= c
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)

    # Rising edge
    rising = (a < x) & (x < b)
    y[rising] = (x[rising] - a) / (b - a) if b != a else 0.0

    # Peak
    y[x == b] = 1.0

    # Falling edge
    falling = (b < x) & (x < c)
    y[falling] = (c - x[falling]) / (c - b) if c != b else 0.0

    # Inclusive ends (optional)
    y[(x <= a) | (x >= c)] = 0.0
    return y


def trap(x: np.ndarray | float, a: float, b: float, c: float, d: float) -> np.ndarray | float:
    """
    Trapezoidal membership function.
    a <= b <= c <= d
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)

    # Rising
    rising = (a < x) & (x < b)
    y[rising] = (x[rising] - a) / (b - a) if b != a else 0.0

    # Plateau
    plateau = (b <= x) & (x <= c)
    y[plateau] = 1.0

    # Falling
    falling = (c < x) & (x < d)
    y[falling] = (d - x[falling]) / (d - c) if d != c else 0.0

    y[(x <= a) | (x >= d)] = 0.0
    return y


# ---------- 2) Define fuzzy sets ----------
def temp_memberships(temp_c: float) -> dict[str, float]:
    # Universe temp: 0..40
    return {
        "cold": float(trap(temp_c, 0, 0, 12, 18)),      # 0-12 full, fades to 18
        "warm": float(tri(temp_c, 16, 22, 28)),         # peak at 22
        "hot":  float(trap(temp_c, 26, 30, 40, 40)),    # starts at 26, full after 30
    }


def hum_memberships(hum_pct: float) -> dict[str, float]:
    # Universe humidity: 0..100
    return {
        "dry":   float(trap(hum_pct, 0, 0, 30, 45)),
        "ok":    float(tri(hum_pct, 35, 50, 65)),
        "humid": float(trap(hum_pct, 55, 70, 100, 100)),
    }


# Output universe: fan power 0..100
fan_x = np.linspace(0, 100, 1001)

def fan_sets(x: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "low":    trap(x, 0, 0, 20, 40),
        "medium": tri(x, 30, 50, 70),
        "high":   trap(x, 60, 80, 100, 100),
    }


# ---------- 3) Rule base (Mamdani: min for AND, max for OR) ----------
def infer_fan(temp_c: float, hum_pct: float) -> tuple[float, dict]:
    T = temp_memberships(temp_c)
    H = hum_memberships(hum_pct)

    out = fan_sets(fan_x)

    # Rules (examples):
    # R1: IF temp is cold THEN fan is low
    r1 = T["cold"]

    # R2: IF temp is warm AND humidity is ok THEN fan is medium
    r2 = min(T["warm"], H["ok"])

    # R3: IF temp is warm AND humidity is humid THEN fan is high
    r3 = min(T["warm"], H["humid"])

    # R4: IF temp is hot THEN fan is high
    r4 = T["hot"]

    # R5: IF humidity is dry AND temp is warm THEN fan is low (comfort: don't blast)
    r5 = min(H["dry"], T["warm"])

    # Aggregate outputs (clip each consequent set by rule strength, then max)
    agg_low = np.maximum.reduce([
        np.minimum(r1, out["low"]),
        np.minimum(r5, out["low"]),
    ])

    agg_med = np.minimum(r2, out["medium"])

    agg_high = np.maximum.reduce([
        np.minimum(r3, out["high"]),
        np.minimum(r4, out["high"]),
    ])

    aggregated = np.maximum.reduce([agg_low, agg_med, agg_high])

    # ---------- 4) Defuzzification: centroid ----------
    if np.sum(aggregated) == 0:
        crisp = 0.0
    else:
        crisp = float(np.sum(fan_x * aggregated) / np.sum(aggregated))

    debug = {
        "T": T, "H": H,
        "rules": {"r1": r1, "r2": r2, "r3": r3, "r4": r4, "r5": r5},
        "crisp": crisp,
    }
    return crisp, debug


if __name__ == "__main__":
    # Example inputs:
    examples = [
        (18, 40),  # cool-ish, ok humidity
        (24, 80),  # warm and humid
        (33, 50),  # hot
        (22, 20),  # warm but dry
    ]

    for t, h in examples:
        power, dbg = infer_fan(t, h)
        print(f"Temp={t}°C  Hum={h}%  => Fan power ~ {power:.1f}%")
        print("  memberships:", dbg["T"], dbg["H"])
        print("  rules:", dbg["rules"])
        print()
