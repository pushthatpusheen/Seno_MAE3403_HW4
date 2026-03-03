"""
MAE 3403 HW4 - Problem 2
Find intersection (meeting point) of:
  circle:   (y - y1)^2 + (x - x1)^2 = R^2
  parabola: y = a*(x - x0)^2 + y0    (default a=0.5, x0=1, y0=1)

Uses scipy.optimize.fsolve.
Plots x,y from -10 to 10.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def system(vars_, x1, y1, R, a, x0, y0):
    x, y = vars_
    eq1 = (y - y1) ** 2 + (x - x1) ** 2 - R ** 2
    eq2 = y - (a * (x - x0) ** 2 + y0)
    return [eq1, eq2]


def get_float(prompt: str, default: float) -> float:
    s = input(f"{prompt} ({default}): ").strip()
    return default if s == "" else float(s)


def main():
    # Test case requirement: center at (x1=1, y1=0)
    x1 = get_float("Circle center x1", 1.0)
    y1 = get_float("Circle center y1", 0.0)
    R = get_float("Circle radius R", 4.0)

    # Parabola defaults based on prompt: y = 0.5*x^2 + 1, with options for width & offset
    a = get_float("Parabola width a (y = a*(x-x0)^2 + y0)", 0.5)
    x0 = get_float("Parabola horizontal shift x0", 0.0)
    y0 = get_float("Parabola vertical offset y0", 1.0)

    # initial guess (can change if it converges to a different intersection)
    x_guess = get_float("Initial guess x", 1.0)
    y_guess = get_float("Initial guess y", 1.0)

    sol = fsolve(system, x0=[x_guess, y_guess], args=(x1, y1, R, a, x0, y0), xtol=1e-12, maxfev=200)
    xi, yi = float(sol[0]), float(sol[1])

    print(f"\nIntersection point:")
    print(f"x = {xi:.8f}")
    print(f"y = {yi:.8f}")

    # Plot
    xs = np.linspace(-10, 10, 800)
    y_par = a * (xs - x0) ** 2 + y0

    # Circle (upper/lower branches where defined)
    inside = R**2 - (xs - x1)**2
    mask = inside >= 0
    y_c_up = np.full_like(xs, np.nan, dtype=float)
    y_c_dn = np.full_like(xs, np.nan, dtype=float)
    y_c_up[mask] = y1 + np.sqrt(inside[mask])
    y_c_dn[mask] = y1 - np.sqrt(inside[mask])

    plt.figure()
    plt.plot(xs, y_par, label="Parabola")
    plt.plot(xs, y_c_up, label="Circle (upper)")
    plt.plot(xs, y_c_dn, label="Circle (lower)")
    plt.plot([xi], [yi], marker="o", linestyle="None", label="Intersection")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True)
    plt.legend()
    plt.title("Circle–Parabola Intersection (fsolve)")
    plt.show()


if __name__ == "__main__":
    main()