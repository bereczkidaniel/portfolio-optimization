"""Portfolio optimization on four Budapest Stock Exchange stocks with real price data.

Data: BSE historical price export (historikus_2025_05_20__2026_05_20_.xlsx).
If that file is not present, the script falls back to the processed daily
log returns in log_returns.csv, so the analysis can be reproduced from the
repository alone.

Setup: short selling allowed (unbounded weights), common target return r_bar,
alpha = 0.95, two consecutive six-month periods.
"""
import json
import os

import cvxpy as cp
import numpy as np
import pandas as pd

XLSX = "historikus_2025_05_20__2026_05_20_.xlsx"   # raw BSE price export
CSV = "log_returns.csv"                             # processed log returns
STOCKS = ["OTP", "MTELEKOM", "RICHTER", "WABERERS"]
n = 4
RBAR = 0.002                          # common target daily return (above the GMV return in both periods)
ALPHA = 0.95                          # confidence level for VaR and CVaR
SPLIT = pd.Timestamp("2025-11-20")    # boundary between the two six-month periods

# ---- load data ----------------------------------------------------------------
if os.path.exists(XLSX):
    # The BSE export is in Hungarian: sheet "Adatok" with columns
    # name, date (YYYY.MM.DD.), closing price, volume.
    df = pd.read_excel(XLSX, sheet_name="Adatok", header=None, skiprows=3)
    df.columns = ["Name", "Date", "Price", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d.")
    px = df.pivot(index="Date", columns="Name", values="Price")[STOCKS].sort_index()
    ret = np.log(px / px.shift(1)).dropna()           # daily log returns
    ret.index.name = "Date"
    ret.to_csv(CSV)
else:
    ret = pd.read_csv(CSV, index_col="Date", parse_dates=True)[STOCKS]

R1 = ret[ret.index < SPLIT].values                    # period 1
R2 = ret[ret.index >= SPLIT].values                   # period 2
d1 = ret[ret.index < SPLIT].index
d2 = ret[ret.index >= SPLIT].index
mu1, mu2 = R1.mean(axis=0), R2.mean(axis=0)
C1 = np.cov(R1.T, bias=True)
C2 = np.cov(R2.T, bias=True)
print(f"Period 1: {len(R1)} days ({d1[0].date()} - {d1[-1].date()})")
print(f"Period 2: {len(R2)} days ({d2[0].date()} - {d2[-1].date()})")


# ---- models -------------------------------------------------------------------
def markowitz(R, mu, C, r):
    """Minimum-variance portfolio at target return r (quadratic program)."""
    x = cp.Variable(n)
    p = cp.Problem(cp.Minimize(cp.quad_form(x, cp.psd_wrap(C))),
                   [mu @ x == r, cp.sum(x) == 1])
    p.solve(solver=cp.CLARABEL)
    return x.value, p.value


def mad(R, mu, r):
    """Mean absolute deviation model (linear program)."""
    Tt = R.shape[0]
    x = cp.Variable(n)
    d = cp.Variable(Tt, nonneg=True)
    f = cp.Variable(Tt, nonneg=True)
    p = cp.Problem(cp.Minimize(cp.sum(d + f)),
                   [R @ x + d - f == r, mu @ x == r, cp.sum(x) == 1])
    p.solve(solver=cp.HIGHS)
    return x.value, p.value


def var_milp(R, mu, r):
    """Value-at-Risk model (mixed-integer linear program).

    Binary b_t marks a day allowed to fall below the VaR level; at most
    (1 - alpha) * T such days. Daily returns are small, so a big-M of 1 suffices.
    """
    Tt = R.shape[0]
    x = cp.Variable(n)
    V = cp.Variable()
    b = cp.Variable(Tt, boolean=True)
    p = cp.Problem(cp.Maximize(V),
                   [R @ x - V + b >= 0,
                    cp.sum(b) <= (1 - ALPHA) * Tt,
                    mu @ x == r, cp.sum(x) == 1])
    p.solve(solver=cp.HIGHS)
    return x.value, p.value


def cvar(R, mu, r):
    """Conditional Value-at-Risk model, Rockafellar-Uryasev formulation (LP)."""
    Tt = R.shape[0]
    x = cp.Variable(n)
    Z = cp.Variable()
    U = cp.Variable(Tt, nonneg=True)
    p = cp.Problem(cp.Minimize(Z + 1.0 / ((1 - ALPHA) * Tt) * cp.sum(U)),
                   [R @ x + Z + U >= 0, mu @ x == r, cp.sum(x) == 1])
    p.solve(solver=cp.HIGHS)
    return x.value, p.value


MODELS = ["Markowitz", "MAD", "VaR", "CVaR"]


def solve_all(R, mu, C, r):
    return {"Markowitz": markowitz(R, mu, C, r), "MAD": mad(R, mu, r),
            "VaR": var_milp(R, mu, r), "CVaR": cvar(R, mu, r)}


# ---- risk measures evaluated on a return sample --------------------------------
def ev_std(x, C):
    return float(np.sqrt(x @ C @ x))


def ev_mad(x, R):
    pr = R @ x
    return float(np.mean(np.abs(pr - pr.mean())))


def ev_var(x, R):
    pr = R @ x
    return float(-np.percentile(pr, 5))


def ev_cvar(x, R):
    pr = R @ x
    q = np.percentile(pr, 5)
    return float(-pr[pr <= q].mean())


def gmv_ret(mu, C):
    """Expected return of the global minimum-variance portfolio."""
    one = np.ones(n)
    Ci = np.linalg.inv(C)
    w = Ci @ one / (one @ Ci @ one)
    return float(mu @ w)


print("GMV return  period 1:", round(gmv_ret(mu1, C1), 6),
      " period 2:", round(gmv_ret(mu2, C2), 6), " | r_bar =", RBAR)

# ---- optimal portfolios in both periods ----------------------------------------
sol1 = solve_all(R1, mu1, C1, RBAR)
sol2 = solve_all(R2, mu2, C2, RBAR)
W1 = {m: sol1[m][0] for m in MODELS}
W2 = {m: sol2[m][0] for m in MODELS}

# ---- efficient frontiers on period 2 -------------------------------------------
r_grid = np.linspace(-0.0005, 0.0050, 45)
r_grid_var = np.linspace(-0.0005, 0.0050, 18)        # coarser grid: the MILP is slower
frontier = {m: {"risk": [], "ret": []} for m in MODELS}
for r in r_grid:
    x, _ = markowitz(R2, mu2, C2, r)
    frontier["Markowitz"]["risk"].append(ev_std(x, C2)); frontier["Markowitz"]["ret"].append(r)
    x, _ = mad(R2, mu2, r)
    frontier["MAD"]["risk"].append(ev_mad(x, R2));        frontier["MAD"]["ret"].append(r)
    x, _ = cvar(R2, mu2, r)
    frontier["CVaR"]["risk"].append(ev_cvar(x, R2));      frontier["CVaR"]["ret"].append(r)
for r in r_grid_var:
    x, _ = var_milp(R2, mu2, r)
    frontier["VaR"]["risk"].append(ev_var(x, R2));        frontier["VaR"]["ret"].append(r)

# ---- all eight optima evaluated on period 2, once per risk measure -------------
risk_ev = {"Markowitz": lambda x: ev_std(x, C2), "MAD": lambda x: ev_mad(x, R2),
           "VaR": lambda x: ev_var(x, R2),       "CVaR": lambda x: ev_cvar(x, R2)}
points = {p: [] for p in MODELS}
for panel in MODELS:
    rf = risk_ev[panel]
    for src in MODELS:
        for per, W in [(1, W1), (2, W2)]:
            x = W[src]
            points[panel].append(dict(model=src, period=per,
                                      risk=rf(x), ret=float(mu2 @ x)))


def metrics(x, R, C):
    pr = R @ x
    return dict(ret=float(pr.mean()), std=ev_std(x, C),
                mad=ev_mad(x, R), var=ev_var(x, R), cvar=ev_cvar(x, R))


m1 = {m: metrics(W1[m], R1, C1) for m in MODELS}
m2 = {m: metrics(W2[m], R2, C2) for m in MODELS}


def desc(R):
    """Annualized mean and standard deviation (252 trading days)."""
    return dict(mu_ann=(R.mean(0) * 252).tolist(),
                sd_ann=(R.std(0, ddof=0) * np.sqrt(252)).tolist())


# ---- save results ---------------------------------------------------------------
np.save("R1.npy", R1)
np.save("R2.npy", R2)
out = dict(stocks=STOCKS, rbar=RBAR, alpha=ALPHA, T1=len(R1), T2=len(R2),
           dates=dict(p1=[str(d1[0].date()), str(d1[-1].date())],
                      p2=[str(d2[0].date()), str(d2[-1].date())]),
           gmv1=gmv_ret(mu1, C1), gmv2=gmv_ret(mu2, C2),
           mu1=mu1.tolist(), mu2=mu2.tolist(),
           desc1=desc(R1), desc2=desc(R2),
           corr1=np.corrcoef(R1.T).tolist(), corr2=np.corrcoef(R2.T).tolist(),
           W1={m: W1[m].tolist() for m in MODELS},
           W2={m: W2[m].tolist() for m in MODELS},
           obj1={m: float(sol1[m][1]) for m in MODELS},
           obj2={m: float(sol2[m][1]) for m in MODELS},
           metrics1=m1, metrics2=m2,
           frontier=frontier, points=points)
with open("results.json", "w") as fh:
    json.dump(out, fh, indent=1)

print("\n--- Period 1 optima ---")
for m in MODELS:
    print(f"{m:10s}", np.round(W1[m], 4), " obj=", round(sol1[m][1], 6))
print("--- Period 2 optima ---")
for m in MODELS:
    print(f"{m:10s}", np.round(W2[m], 4), " obj=", round(sol2[m][1], 6))
print("\nDone: results.json, R1.npy, R2.npy")
