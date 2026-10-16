import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

d = np.load("data.npz"); X, labels = d["X"], d["labels"]
N, P0, P1 = len(labels), 0.35, 0.65
n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())

m0 = np.full(4, -0.5)
C0 = 0.25 * np.array([[2, -0.5, 0.3, 0], [-0.5, 1, -0.5, 0],
                      [0.3, -0.5, 1, 0], [0, 0, 0, 2]])
m1 = np.ones(4)
C1 = np.array([[1, 0.3, -0.2, 0], [0.3, 2, 0.3, 0],
               [-0.2, 0.3, 1, 0], [0, 0, 0, 3]])

# Naive Bayes model: true means, diagonal covariances (true variances, zero off-diagonals)
C0nb, C1nb = np.diag(np.diag(C0)), np.diag(np.diag(C1))

def roc(s):
    order = np.argsort(s); ss, ls = s[order], labels[order]
    c1 = np.concatenate([[0], np.cumsum(ls)]); c0 = np.concatenate([[0], np.cumsum(1 - ls)])
    tpr, fpr = (n1 - c1) / n1, (n0 - c0) / n0
    perr = fpr * P0 + (1 - tpr) * P1
    lg = np.concatenate([[-np.inf], 0.5 * (ss[:-1] + ss[1:]), [np.inf]])
    return ss, tpr, fpr, perr, lg

s_true = mvn.logpdf(X, m1, C1) - mvn.logpdf(X, m0, C0)
s_nb   = mvn.logpdf(X, m1, C1nb) - mvn.logpdf(X, m0, C0nb)

gth = P0 / P1
res = {}
for name, s in [("True (Part A)", s_true), ("Naive Bayes (Part B)", s_nb)]:
    ss, tpr, fpr, perr, lg = roc(s)
    ke = np.argmin(perr); kt = np.searchsorted(ss, np.log(gth), side="right")
    res[name] = (tpr, fpr, perr, lg, ke, kt)
    print(f"{name}:")
    print(f"  empirical: gamma={np.exp(lg[ke]):.4f} FPR={fpr[ke]:.4f} TPR={tpr[ke]:.4f} P(err)={perr[ke]:.4f}")
    print(f"  theory   : gamma={gth:.4f} FPR={fpr[kt]:.4f} TPR={tpr[kt]:.4f} P(err)={perr[kt]:.4f}")

# AUC comparison
for name, (tpr, fpr, *_ ) in res.items():
    auc = abs(np.trapezoid(tpr, fpr)); print(f"AUC {name}: {auc:.4f}")

tpr, fpr, perr, lg, ke, kt = res["Naive Bayes (Part B)"]
np.savez("roc_values_nb.npz", gamma=np.exp(lg), TPR=tpr, FPR=fpr,
         P_D0_given_L1=1 - tpr, P_D1_given_L0=fpr, perr=perr)

plt.figure(figsize=(6.5, 6))
tA, fA, pA, lA, kA, ktA = res["True (Part A)"]
plt.plot(fA, tA, "b-", lw=1.8, label="Part A: true pdf")
plt.plot(fpr, tpr, "r--", lw=1.8, label="Part B: Naive Bayes")
plt.plot(fA[kA], tA[kA], "bo", ms=10, mfc="none", mew=2, label=f"A empirical min, P(err)={pA[kA]:.4f}")
plt.plot(fA[ktA], tA[ktA], "bx", ms=11, mew=2.5, label=f"A theory γ={gth:.3f}, P(err)={pA[ktA]:.4f}")
plt.plot(fpr[ke], tpr[ke], "ro", ms=10, mfc="none", mew=2, label=f"B empirical min, P(err)={perr[ke]:.4f}")
plt.plot(fpr[kt], tpr[kt], "rx", ms=11, mew=2.5, label=f"B theory γ={gth:.3f}, P(err)={perr[kt]:.4f}")
plt.xlabel("P(D=1 | L=0)  (false positive)"); plt.ylabel("P(D=1 | L=1)  (true positive)")
plt.title("ROC: true-pdf ERM vs Naive Bayes ERM")
plt.grid(alpha=.3); plt.legend(loc="lower right", fontsize=8); plt.xlim(0, 1); plt.ylim(0, 1.02)
plt.tight_layout(); plt.savefig("roc_curve_nb.png", dpi=150)
plt.show()
