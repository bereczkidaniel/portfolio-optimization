"""Export the four period 1 models as LP files for an independent cross-check.

MAD1 / VaR1 / CVaR1 -> CPLEX LP format as exported by GUSEK (GLPK), including
its tab padding after section keywords, so the files open in GUSEK unchanged.
markowitz1 -> CPLEX LP format with a quadratic (QP) objective.

Run gen.py first: this script reads R1.npy and results.json.
"""
import json
import os

import numpy as np

R1 = np.load("R1.npy")
with open("results.json") as fh:
    D = json.load(fh)
STOCKS = D["stocks"]
mu1 = np.array(D["mu1"])
RBAR = D["rbar"]
ALPHA = D["alpha"]
T, n = R1.shape


def num(v):
    """Number at full precision."""
    return f"{v:.15g}"


def signed(v):
    return ("+" if v >= 0 else "-") + num(abs(v))


def ret_terms(coefs):
    """Linear term over the stocks, e.g. "+c1OTP\t-c2MTELEKOM\t..."."""
    return "\t".join(signed(c) + s for c, s in zip(coefs, STOCKS))


# GUSEK export style: trailing tabs after the "subject to" keyword
PAD = "\t" * (n + 1)

# ----------------------------------------------------- mean absolute deviation
with open("MAD1.lp", "w", newline="\r\n") as f:
    f.write("min\n\n")
    for t in range(1, T + 1):
        f.write(f"+D{t}+F{t}\n\n")
    f.write(f"subject to{PAD}\n\n")
    for t in range(T):
        f.write(ret_terms(R1[t]) + f"\t\t+D{t+1}-F{t+1}={num(RBAR)}\n\n")
    f.write(ret_terms(mu1) + f"\t\t={num(RBAR)}\n\n")              # target return
    f.write("\t".join("+" + s for s in STOCKS) + "\t\t=1\n\n\n")    # budget
    f.write("bounds\t\n\n")
    for s in STOCKS:
        f.write(f"{s}\tfree\n\n")
    f.write("end\t\n")

# ----------------------------------------------------------------- VaR (MILP)
with open("VaR1.lp", "w", newline="\r\n") as f:
    f.write(f"max\nVaR95\nsubject to{PAD}\n\n")
    for t in range(T):
        f.write(ret_terms(R1[t]) + f"\t\t-VaR95+B{t+1}>=0\n\n")
    f.write(ret_terms(mu1) + f"\t\t={num(RBAR)}\n\n")              # target return
    f.write("\t".join("+" + s for s in STOCKS) + "\t\t=1\n")        # budget
    for t in range(1, T + 1):                                       # at most (1 - alpha) T tail days
        tail = f"\t<={round((1 - ALPHA) * T, 4):g}" if t == T else "\t"
        if t == T:
            f.write(f"+B{t}{tail}\n")
        else:
            f.write(f"+B{t}{tail}\n\n")
    f.write("bound\nVaR95 free\n")
    f.write(f"{STOCKS[0]}\tfree\n")                                 # first variable without a blank line before it
    for s in STOCKS[1:]:
        f.write(f"\n{s}\tfree\n")
    f.write("binary\n")
    for t in range(1, T + 1):
        f.write(f"B{t}\n")
        if t < T:
            f.write("\n")
    f.write("end\n")

# ------------------------------------------------------------------------ CVaR
w = 1.0 / ((1 - ALPHA) * T)
with open("CVaR1.lp", "w", newline="\r\n") as f:
    f.write("min Z\n\n")
    for t in range(1, T + 1):
        f.write(f"+{num(w)}U{t}\n\n")
    f.write(f"subject to{PAD}\n\n")
    for t in range(T):
        f.write(ret_terms(R1[t]) + f"\t+Z+U{t+1}>=0\n\n")
    f.write(ret_terms(mu1) + f"\t={num(RBAR)}\n\n")                 # target return
    f.write("\t".join("+" + s for s in STOCKS) + "\t=1\n")          # budget
    f.write("bounds\t\n\n")
    f.write("Z\tfree\n\n")
    for s in STOCKS:
        f.write(f"{s}\tfree\n\n")
    f.write("end\t\n")

# ------------------------------------------------- Markowitz QP (CPLEX LP format)
C1 = np.cov(R1.T, bias=True)
with open("markowitz1.lp", "w", newline="\n") as f:
    f.write("\\* Markowitz portfolio model - period 1 *\\\n")
    f.write("\\* min x^T C x   s.t.  mu^T x = r_bar, sum x = 1, x free *\\\n")
    f.write("Minimize\n obj: [ ")
    terms = []
    for i in range(n):
        terms.append(f"+ {2 * C1[i, i]:.12g} {STOCKS[i]}^2")
    for i in range(n):
        for j in range(i + 1, n):
            terms.append(f"+ {4 * C1[i, j]:.12g} {STOCKS[i]} * {STOCKS[j]}")
    line = ""
    for tk in terms:
        if len(line) + len(tk) > 78:
            f.write(line + "\n   ")
            line = ""
        line += tk + " "
    f.write(line + "] / 2\n")
    f.write("Subject To\n")
    f.write(" budget: " + " + ".join(STOCKS) + " = 1\n")
    f.write(" expret: " + " + ".join(f"{mu1[i]:.12g} {STOCKS[i]}"
                                      for i in range(n)) + f" = {RBAR}\n")
    f.write("Bounds\n")
    for s in STOCKS:
        f.write(f" {s} free\n")
    f.write("End\n")

print("LP files written:")
for fn in ["MAD1.lp", "VaR1.lp", "CVaR1.lp", "markowitz1.lp"]:
    print(f"  {fn:16s} {os.path.getsize(fn):>7d} B")
