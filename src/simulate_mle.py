"""Simulated demonstration of the distributional MLE in paper/mle_proposal.tex.

Simulates a bank-quarter panel from a known sinh-arcsinh (SHASH) location-scale-shape
model in which an adverse oil shock shifts the mean, widens the spread, skews the
distribution left, and fattens the tails, with the effect varying by bank exposure.
Then fits (1) a Gaussian heteroskedastic model and (2) the SHASH model by maximum
likelihood, computes time-clustered robust standard errors, and compares the fits.

Outputs go to output/simulated_mle/.
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import optimize, stats
from main import OUTPUT

OUT = OUTPUT / "simulated_mle"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1504)

T, N = 80, 250                       # quarters, banks per quarter
COVS = ["const", "shock", "exposure", "size", "shock*exposure", "shock*size"]
K = len(COVS)
PARAMS = ["mu", "log_sigma", "eps", "log_delta"]

# ---- True data-generating process (coefficients in COVS order) -------------
TRUE = {
    "mu":        [1.0, -0.6, 0.2, 0.3, -0.5, 0.0],
    "log_sigma": [0.0, 0.25, 0.10, -0.15, 0.15, 0.0],
    "eps":       [0.0, -0.35, 0.0, 0.10, -0.25, 0.0],   # negative eps = left skew
    "log_delta": [0.0, -0.15, 0.0, 0.0, -0.05, 0.0],    # negative = heavier tails
}
TRUE_VEC = np.concatenate([TRUE[p] for p in PARAMS])


def design(s, E, w):
    return np.column_stack([np.ones_like(s), s, E, w, s * E, s * w])


def unpack(theta):
    return [theta[k * K:(k + 1) * K] for k in range(4)]


def shash_ll(theta, X, y, obs=False):
    bm, bs, be, bd = unpack(theta)
    mu, sig = X @ bm, np.exp(np.clip(X @ bs, -8, 8))
    eps, dl = X @ be, np.exp(np.clip(X @ bd, -4, 4))
    z = (y - mu) / sig
    r = dl * np.arcsinh(z) - eps
    ll = (np.log(dl) + np.log(np.cosh(np.clip(r, -300, 300))) - 0.5 * np.log(2 * np.pi)
          - 0.5 * np.log1p(z ** 2) - 0.5 * np.sinh(np.clip(r, -300, 300)) ** 2 - np.log(sig))
    return ll if obs else ll.sum()


def gauss_ll(theta, X, y, obs=False):
    bm, bs = theta[:K], theta[K:]
    mu, ls = X @ bm, np.clip(X @ bs, -8, 8)
    ll = -ls - 0.5 * np.log(2 * np.pi) - 0.5 * ((y - mu) / np.exp(ls)) ** 2
    return ll if obs else ll.sum()


def shash_cdf(y, mu, sig, eps, dl):
    return stats.norm.cdf(np.sinh(dl * np.arcsinh((y - mu) / sig) - eps))


def shash_quantile(p, mu, sig, eps, dl):
    return mu + sig * np.sinh((np.arcsinh(stats.norm.ppf(p)) + eps) / dl)


# ---- Simulate ---------------------------------------------------------------
s_t = rng.standard_normal(T)                         # adverse oil shock (std. units)
E_i = rng.standard_normal(N * 3)                     # producer-side exposure (fixed)
bank = rng.integers(0, len(E_i), size=(T, N))
s = np.repeat(s_t, N)
t_id = np.repeat(np.arange(T), N)
E = E_i[bank.ravel()]
w = rng.standard_normal(T * N)
X = design(s, E, w)
bm, bs, be, bd = (X @ np.array(TRUE[p]) for p in PARAMS)
z = np.sinh((np.arcsinh(rng.standard_normal(T * N)) + be) / np.exp(bd))
y = bm + np.exp(bs) * z
print(f"Simulated {T*N:,} bank-quarters ({T} quarters x {N} banks)")

# ---- Estimate ---------------------------------------------------------------
beta0 = np.linalg.lstsq(X, y, rcond=None)[0]
resid = y - X @ beta0
gamma0 = np.linalg.lstsq(X, np.log(np.abs(resid) + 1e-3), rcond=None)[0]
th_g0 = np.concatenate([beta0, gamma0])
g = optimize.minimize(lambda t: -gauss_ll(t, X, y), th_g0, method="BFGS")
th_g = g.x
th_s0 = np.concatenate([th_g[:K], th_g[K:], np.zeros(K), np.zeros(K)])
sh = optimize.minimize(lambda t: -shash_ll(t, X, y), th_s0, method="L-BFGS-B",
                       options={"maxiter": 3000, "maxfun": 200000})
sh = optimize.minimize(lambda t: -shash_ll(t, X, y), sh.x, method="BFGS")
th_s = sh.x
print(f"Gaussian  logL = {-g.fun:,.1f}  | SHASH logL = {-sh.fun:,.1f}")


def cluster_se(ll_fn, theta, X, y, cl, h=1e-4):
    """Time-clustered sandwich: H^-1 (sum_t g_t g_t') H^-1 with numerical derivatives."""
    p = len(theta)
    scores = np.empty((len(y), p))
    for j in range(p):
        d = np.zeros(p); d[j] = h
        scores[:, j] = (ll_fn(theta + d, X, y, True) - ll_fn(theta - d, X, y, True)) / (2 * h)
    H = np.empty((p, p))
    for j in range(p):
        d = np.zeros(p); d[j] = h
        gp = np.array([(ll_fn(theta + d + e, X, y) - ll_fn(theta + d - e, X, y)) / (2 * h)
                       for e in np.eye(p) * h])
        gm = np.array([(ll_fn(theta - d + e, X, y) - ll_fn(theta - d - e, X, y)) / (2 * h)
                       for e in np.eye(p) * h])
        H[j] = (gp - gm) / (2 * h)
    H = (H + H.T) / 2
    G = pd.DataFrame(scores).groupby(cl).sum().to_numpy()
    Hinv = np.linalg.inv(-H)
    V = Hinv @ (G.T @ G) @ Hinv * len(G) / (len(G) - 1)
    return np.sqrt(np.diag(V))


se_s = cluster_se(shash_ll, th_s, X, y, t_id)
se_g = cluster_se(gauss_ll, th_g, X, y, t_id)

# ---- Coefficient table ------------------------------------------------------
rows = []
for k, name in enumerate(PARAMS):
    for j, c in enumerate(COVS):
        i = k * K + j
        rows.append(dict(parameter=name, covariate=c, truth=TRUE_VEC[i], estimate=th_s[i],
                         cluster_se=se_s[i], t_stat=(th_s[i] - TRUE_VEC[i]) / se_s[i],
                         wald_z_vs_zero=th_s[i] / se_s[i]))
coef = pd.DataFrame(rows)
coef.to_csv(OUT / "shash_coefficients.csv", index=False)

# ---- Model comparison ----------------------------------------------------------
n = len(y)
cmp = pd.DataFrame({
    "model": ["Gaussian (hetero)", "SHASH"],
    "n_params": [2 * K, 4 * K],
    "logL": [-g.fun, -sh.fun],
})
cmp["AIC"] = 2 * cmp.n_params - 2 * cmp.logL
cmp["BIC"] = np.log(n) * cmp.n_params - 2 * cmp.logL
cmp.to_csv(OUT / "model_comparison.csv", index=False)
LR = 2 * (-sh.fun + g.fun)
# Shock-effect test on shape parameters (Wald, clustered): shock terms in eps and log_delta
idx = [2 * K + 1, 2 * K + 4, 3 * K + 1, 3 * K + 4]
print(f"LR (SHASH vs Gaussian) = {LR:,.1f}")

# ---- Mean-only / true-vs-fitted quantile effects -------------------------------
def fitted_quantiles(model, Xnew, ps):
    if model == "true":
        m, sg, e, d = (Xnew @ np.array(TRUE[p]) for p in PARAMS)
        sg, d = np.exp(sg), np.exp(d)
    elif model == "shash":
        b = unpack(th_s); m, sg, e, d = Xnew @ b[0], np.exp(Xnew @ b[1]), Xnew @ b[2], np.exp(Xnew @ b[3])
    else:
        m, sg = Xnew @ th_g[:K], np.exp(Xnew @ th_g[K:]); e, d = 0 * m, 1 + 0 * m
    return np.array([shash_quantile(p, m, sg, e, d) for p in ps])


ps = [0.05, 0.25, 0.5, 0.75, 0.95]
grid = np.linspace(-2.5, 2.5, 51)
fig, ax = plt.subplots(2, 2, figsize=(12, 9))

# Panel A: quantile fan, high-exposure bank, true vs SHASH vs Gaussian
Xg = design(grid, np.ones_like(grid), np.zeros_like(grid))
qt, qs, qg = (fitted_quantiles(m, Xg, ps) for m in ("true", "shash", "gauss"))
cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(ps)))
for k, p in enumerate(ps):
    ax[0, 0].plot(grid, qt[k], color=cols[k], lw=2, label=f"true q{int(p*100)}")
    ax[0, 0].plot(grid, qs[k], color=cols[k], ls="--", lw=1.5)
    ax[0, 0].plot(grid, qg[k], color=cols[k], ls=":", lw=1.5)
ax[0, 0].set(title="A. Conditional quantiles vs shock (exposure = +1 SD)\nsolid = truth, dashed = SHASH, dotted = Gaussian",
             xlabel="adverse oil shock (SD)", ylabel="outcome")
ax[0, 0].legend(fontsize=7, ncol=2)

# Panel B: conditional density under mild vs severe shock
yy = np.linspace(-8, 8, 600)
for sv, c in [(-2, "tab:green"), (0, "tab:gray"), (2, "tab:red")]:
    x1 = design(np.array([sv]), np.array([1.0]), np.array([0.0]))
    b = unpack(th_s)
    m, sg, e, d = x1 @ b[0], np.exp(x1 @ b[1]), x1 @ b[2], np.exp(x1 @ b[3])
    zz = (yy - m) / sg; r = d * np.arcsinh(zz) - e
    dens = d * np.cosh(r) / np.sqrt(2 * np.pi * (1 + zz ** 2)) * np.exp(-0.5 * np.sinh(r) ** 2) / sg
    ax[0, 1].plot(yy, dens, color=c, label=f"shock = {sv:+d} SD")
ax[0, 1].set(title="B. Fitted SHASH density by shock size (exposure = +1 SD)", xlabel="outcome", ylabel="density")
ax[0, 1].legend()

# Panel C: PIT histograms
b = unpack(th_s)
pit_s = shash_cdf(y, X @ b[0], np.exp(X @ b[1]), X @ b[2], np.exp(X @ b[3]))
pit_g = stats.norm.cdf((y - X @ th_g[:K]) / np.exp(X @ th_g[K:]))
ax[1, 0].hist(pit_g, bins=20, density=True, alpha=0.55, label="Gaussian")
ax[1, 0].hist(pit_s, bins=20, density=True, alpha=0.55, label="SHASH")
ax[1, 0].axhline(1, color="k", lw=1)
ax[1, 0].set(title="C. PIT histograms (uniform = well calibrated)", xlabel="PIT", ylabel="density")
ax[1, 0].legend()

# Panel D: coefficient recovery with 95% CI
ax[1, 1].errorbar(TRUE_VEC, th_s, yerr=1.96 * se_s, fmt="o", ms=4, alpha=0.8)
lim = [min(TRUE_VEC.min(), th_s.min()) - 0.1, max(TRUE_VEC.max(), th_s.max()) + 0.1]
ax[1, 1].plot(lim, lim, "k--", lw=1)
ax[1, 1].set(title="D. Estimated vs true coefficients (95% clustered CI)", xlabel="true", ylabel="estimated")
fig.tight_layout()
fig.savefig(OUT / "simulated_mle_diagnostics.png", dpi=150)

# ---- Distributional effects: shock moves what part of the distribution? ------------
x_lo = design(np.array([0.0]), np.array([1.0]), np.array([0.0]))
x_hi = design(np.array([2.0]), np.array([1.0]), np.array([0.0]))
eff = pd.DataFrame({
    "quantile": ps,
    "true_effect": (fitted_quantiles("true", x_hi, ps) - fitted_quantiles("true", x_lo, ps)).ravel(),
    "shash_effect": (fitted_quantiles("shash", x_hi, ps) - fitted_quantiles("shash", x_lo, ps)).ravel(),
    "gaussian_effect": (fitted_quantiles("gauss", x_hi, ps) - fitted_quantiles("gauss", x_lo, ps)).ravel(),
})
eff.to_csv(OUT / "quantile_effects_2sd_shock.csv", index=False)

# ---- Text report ---------------------------------------------------------------------
pd.set_option("display.width", 140, "display.float_format", "{:.3f}".format)
report = f"""SIMULATED MLE DEMONSTRATION  (seed 1504, {n:,} bank-quarters)

1. MODEL COMPARISON
{cmp.to_string(index=False)}
LR statistic (SHASH vs Gaussian, 12 restrictions) = {LR:,.1f}

2. SHASH COEFFICIENTS (truth vs estimate, time-clustered SEs)
{coef.to_string(index=False)}

3. EFFECT OF A 2-SD ADVERSE SHOCK ON QUANTILES (exposure = +1 SD, size = 0)
{eff.to_string(index=False)}

4. GAUSSIAN SHOCK COEFFICIENTS (mean, log-sigma) for reference
mean shock={th_g[1]:.3f} (se {se_g[1]:.3f}); log-sigma shock={th_g[K+1]:.3f} (se {se_g[K+1]:.3f})
"""
(OUT / "simulated_mle_results.txt").write_text(report)
print(report)
