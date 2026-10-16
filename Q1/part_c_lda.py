import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.stats import multivariate_normal as mvn

d = np.load("data.npz"); X, labels = d["X"], d["labels"]
N, P0, P1 = len(labels), 0.35, 0.65
n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())
X0, X1 = X[labels == 0], X[labels == 1]

# ---- Sample-average estimates of class means and covariances ----
mu0, mu1 = X0.mean(axis=0), X1.mean(axis=0)
S0, S1 = np.cov(X0, rowvar=False), np.cov(X1, rowvar=False)   # unbiased (1/(n-1))
print("mu0_hat =", np.round(mu0, 4)); print("mu1_hat =", np.round(mu1, 4))
print("Sigma0_hat =\n", np.round(S0, 4)); print("Sigma1_hat =\n", np.round(S1, 4))

# ---- Scatter matrices (equal weights for the two classes) ----
Sb = np.outer(mu1 - mu0, mu1 - mu0)
Sw = S0 + S1
# Generalized eigendecomposition: Sb w = lambda Sw w ; w = eigenvector of largest eigenvalue
vals, vecs = eigh(Sb, Sw)
w = vecs[:, np.argmax(vals)]
if w @ (mu1 - mu0) < 0: w = -w            # orient so class 1 projects higher
w = w / np.linalg.norm(w)
print("w_LDA =", np.round(w, 4), " largest eigenvalue =", round(vals.max(), 4))

# ---- ROC by sweeping tau ----
y = X @ w
order = np.argsort(y); ys, ls = y[order], labels[order]
c1 = np.concatenate([[0], np.cumsum(ls)]); c0 = np.concatenate([[0], np.cumsum(1 - ls)])
tpr, fpr = (n1 - c1) / n1, (n0 - c0) / n0            # D=1 if w^T x > tau
perr = fpr * P0 + (1 - tpr) * P1                      # priors from sample counts: n0/N, n1/N
perr_cnt = fpr * (n0 / N) + (1 - tpr) * (n1 / N)      # sample-count version
tau = np.concatenate([[-np.inf], 0.5 * (ys[:-1] + ys[1:]), [np.inf]])
k = np.argmin(perr_cnt)
print(f"Min P(error) (sample-count priors) = {perr_cnt[k]:.4f} at tau = {tau[k]:.4f}")
print(f"  FPR={fpr[k]:.4f} TPR={tpr[k]:.4f}   (with true priors: {perr[k]:.4f})")
print(f"  errors: {int(round(perr_cnt[k]*N))} of {N}")
print(f"AUC LDA = {abs(np.trapezoid(tpr, fpr)):.4f}")
np.savez("roc_values_lda.npz", tau=tau, TPR=tpr, FPR=fpr,
         P_D0_given_L1=1 - tpr, P_D1_given_L0=fpr, perr=perr_cnt, w=w)

# ---- Parts A and B for comparison ----
m0 = np.full(4, -0.5); C0 = 0.25*np.array([[2,-.5,.3,0],[-.5,1,-.5,0],[.3,-.5,1,0],[0,0,0,2]])
m1 = np.ones(4); C1 = np.array([[1,.3,-.2,0],[.3,2,.3,0],[-.2,.3,1,0],[0,0,0,3]])
def roc(s):
    o = np.argsort(s); l = labels[o]
    a = np.concatenate([[0], np.cumsum(l)]); b = np.concatenate([[0], np.cumsum(1 - l)])
    t, f = (n1 - a) / n1, (n0 - b) / n0
    return f, t, f * (n0 / N) + (1 - t) * (n1 / N)
fA, tA, pA = roc(mvn.logpdf(X, m1, C1) - mvn.logpdf(X, m0, C0))
fB, tB, pB = roc(mvn.logpdf(X, m1, np.diag(np.diag(C1))) - mvn.logpdf(X, m0, np.diag(np.diag(C0))))
for nm, f, t, p in [("A", fA, tA, pA), ("B", fB, tB, pB)]:
    print(f"Part {nm}: min P(err, sample-count priors) = {p.min():.4f}, AUC = {abs(np.trapezoid(t, f)):.4f}")

# ---- Plot ----
plt.figure(figsize=(6.5, 6))
plt.plot(fA, tA, "b-", lw=1.6, label="Part A: true pdf")
plt.plot(fB, tB, "r--", lw=1.6, label="Part B: Naive Bayes")
plt.plot(fpr, tpr, "g-.", lw=1.8, label="Part C: Fisher LDA")
plt.plot(fpr[k], tpr[k], "ko", ms=11, mfc="none", mew=2.2,
         label=f"LDA min P(error)={perr_cnt[k]:.4f}, τ={tau[k]:.3f}")
plt.xlabel("P(D=1 | L=0)  (false positive)"); plt.ylabel("P(D=1 | L=1)  (true positive)")
plt.title("ROC: Fisher LDA vs Parts A and B")
plt.grid(alpha=.3); plt.legend(loc="lower right", fontsize=9); plt.xlim(0, 1); plt.ylim(0, 1.02)
plt.tight_layout(); plt.savefig("roc_curve_lda.png", dpi=150)
plt.show()
