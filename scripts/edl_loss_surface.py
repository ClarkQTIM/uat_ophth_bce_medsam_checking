"""
Loss-surface check for the K=3 EDL loss on PAPILA soft targets.

Loss per pixel (as in the IK dirichlet_loss, generalised to K=3):
    L = sum_k [(q_k - p_k)^2 + p_k(1-p_k)/(S+1)] + lam * KL[Dir(a_tilde) || Dir(1)]
    a_tilde = q + (1-q) * alpha

Question: at a disagreement target such as q=(0.5,0.5,0), does the optimum
have LOWER total evidence S than at an agreement target such as q=(1,0,0)?
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, digamma

TARGETS = {
    "BG/BG     (1,0,0)":     np.array([1.0, 0.0, 0.0]),
    "BG/rim    (.5,.5,0)":   np.array([0.5, 0.5, 0.0]),
    "rim/rim   (0,1,0)":     np.array([0.0, 1.0, 0.0]),
    "rim/cup   (0,.5,.5)":   np.array([0.0, 0.5, 0.5]),
}
LAMBDAS = [0.0, 1e-4 / 3, 1e-3, 1e-2, 1e-1, 1.0]
LOG_A_MAX = np.log(1e6)  # cap so "unbounded" optima show up as hitting the cap


def kl_to_uniform(a):
    a0 = a.sum()
    K = len(a)
    return (gammaln(a0) - gammaln(K) - gammaln(a).sum()
            + ((a - 1.0) * (digamma(a) - digamma(a0))).sum())


def edl_loss(log_a, q, lam):
    a = np.exp(log_a)
    S = a.sum()
    p = a / S
    mse = ((q - p) ** 2 + p * (1 - p) / (S + 1)).sum()
    a_t = q + (1 - q) * a
    return mse + lam * kl_to_uniform(a_t)


def optimum(q, lam):
    best = None
    for start in [np.zeros(3), np.log([5, 5, 5]), np.log(q * 50 + 1)]:
        r = minimize(edl_loss, start, args=(q, lam), method="L-BFGS-B",
                     bounds=[(np.log(1e-4), LOG_A_MAX)] * 3)
        if best is None or r.fun < best.fun:
            best = r
    a = np.exp(best.x)
    return a, a.sum(), a / a.sum(), best.fun


if __name__ == "__main__":
    print(f"{'lambda':>9} | {'target':<20} | {'S*':>10} | {'p*':<22} | capped?")
    print("-" * 80)
    for lam in LAMBDAS:
        for name, q in TARGETS.items():
            a, S, p, _ = optimum(q, lam)
            capped = "yes" if np.any(np.log(a) > LOG_A_MAX - 1e-3) else ""
            print(f"{lam:9.2e} | {name:<20} | {S:10.3g} | "
                  f"({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})    | {capped}")
        print("-" * 80)

    # Gradient of the variance term w.r.t. S at p = q (no KL):
    print("\nVariance-term value at p=q, as a function of S (lam=0):")
    for name, q in TARGETS.items():
        vals = [(q * (1 - q)).sum() / (S + 1) for S in (3, 10, 100, 1000)]
        print(f"  {name:<20} S=3,10,100,1000 -> " + ", ".join(f"{v:.4f}" for v in vals))
