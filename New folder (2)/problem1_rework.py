"""
MAE 3403 HW4 - Problem 1 (Rework)
- Uses scipy.integrate.quad instead of Simpson
- Uses scipy.optimize.fsolve instead of Secant

Assumptions (from prompt):
- Rock diameters D are modeled as log-normal: ln(D) ~ N(mu, sigma)
- Sieving truncates sizes to [Dmin, Dmax]
- We sample via inverse-CDF: pick P~U(0,1), solve CDF_trunc(D)=P for D
"""

import math
from random import random as rnd
from scipy.integrate import quad
from scipy.optimize import fsolve


# -----------------------------
# PDFs and CDFs
# -----------------------------
def ln_pdf(D: float, mu: float, sig: float) -> float:
    """Standard log-normal PDF f(D)."""
    if D <= 0.0:
        return 0.0
    pref = 1.0 / (D * sig * math.sqrt(2.0 * math.pi))
    expo = -((math.log(D) - mu) ** 2) / (2.0 * sig ** 2)
    return pref * math.exp(expo)


def ln_cdf(D: float, mu: float, sig: float) -> float:
    """CDF F(D) = integral_0^D f(x) dx using quad."""
    if D <= 0.0:
        return 0.0
    val, _ = quad(lambda x: ln_pdf(x, mu, sig), 0.0, D)
    return val


def trunc_cdf(D: float, mu: float, sig: float, Dmin: float, Dmax: float, Fmin: float, Fmax: float) -> float:
    """
    Truncated CDF on [Dmin, Dmax]:
      F_trunc(D) = (F(D) - F(Dmin)) / (F(Dmax) - F(Dmin))
    """
    if D <= Dmin:
        return 0.0
    if D >= Dmax:
        return 1.0
    FD = ln_cdf(D, mu, sig)
    return (FD - Fmin) / (Fmax - Fmin)


# -----------------------------
# Sampling + stats
# -----------------------------
def inverse_trunc_sample(mu: float, sig: float, Dmin: float, Dmax: float, Fmin: float, Fmax: float, P: float) -> float:
    """Solve F_trunc(D)=P for D using fsolve."""
    def eqn(D):
        # fsolve passes an array; we return scalar array-like
        d = float(D[0])
        return trunc_cdf(d, mu, sig, Dmin, Dmax, Fmin, Fmax) - P

    # good initial guess in the middle of the interval
    x0 = 0.5 * (Dmin + Dmax)
    sol = fsolve(eqn, x0=[x0], xtol=1e-10, maxfev=200)
    d = float(sol[0])

    # keep it physically inside bounds (numerical nudges happen)
    if d < Dmin:
        d = Dmin
    if d > Dmax:
        d = Dmax
    return d


def make_sample(mu: float, sig: float, Dmin: float, Dmax: float, Fmin: float, Fmax: float, N: int = 100):
    """Generate N rock diameters from truncated log-normal."""
    probs = [rnd() for _ in range(N)]
    return [inverse_trunc_sample(mu, sig, Dmin, Dmax, Fmin, Fmax, p) for p in probs]


def sample_stats(values):
    """Return (mean, sample variance with N-1 in denominator)."""
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, var


# -----------------------------
# CLI inputs (defaults like your file)
# -----------------------------
def get_float(prompt: str, default: float) -> float:
    s = input(f"{prompt} ({default}): ").strip()
    return default if s == "" else float(s)


def main():
    # Suggested defaults (same spirit as your provided file)
    mu_default = math.log(2.0)   # ln(inches)
    sig_default = 1.0
    Dmax_default = 1.0
    Dmin_default = 3.0 / 8.0
    nsamples_default = 11
    nper_default = 100

    mu = get_float("Mean mu of ln(D)", mu_default)
    sig = get_float("Std dev sigma of ln(D)", sig_default)
    Dmax = get_float("Large aperture size Dmax", Dmax_default)
    Dmin = get_float("Small aperture size Dmin", Dmin_default)
    nsamples = int(get_float("How many samples?", nsamples_default))
    nper = int(get_float("How many rocks per sample?", nper_default))

    if Dmin <= 0 or Dmax <= 0 or Dmin >= Dmax:
        raise ValueError("Need 0 < Dmin < Dmax.")

    # Precompute F(Dmin), F(Dmax) for normalization
    Fmin = ln_cdf(Dmin, mu, sig)
    Fmax = ln_cdf(Dmax, mu, sig)

    means = []
    for k in range(1, nsamples + 1):
        sample = make_sample(mu, sig, Dmin, Dmax, Fmin, Fmax, N=nper)
        m, v = sample_stats(sample)
        means.append(m)
        print(f"Sample {k:02d}: mean = {m:.6f}, var = {v:.6f}")

    m_means, v_means = sample_stats(means)
    print(f"\nMean of sampling mean = {m_means:.6f}")
    print(f"Variance of sampling mean = {v_means:.10f}")


if __name__ == "__main__":
    main()