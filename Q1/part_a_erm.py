import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

rng = np.random.default_rng(0)
N, P0, P1 = 10000, 0.35, 0.65
m0 = np.full(4, -0.5)
C0 = 0.25 * np.array([[2, -0.5, 0.3, 0], [-0.5, 1, -0.5, 0],
                      [0.3, -0.5, 1, 0], [0, 0, 0, 2]])
m1 = np.ones(4)
C1 = np.array([[1, 0.3, -0.2, 0], [0.3, 2, 0.3, 0],
               [-0.2, 0.3, 1, 0], [0, 0, 0, 3]])

# Data generation 
labels = (rng.random(N) < P1).astype(int)
X = np.empty((N, 4))
n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())
X[labels == 0] = rng.multivariate_normal(m0, C0, size=n0)
X[labels == 1] = rng.multivariate_normal(m1, C1, size=n1)
np.savez("data.npz", X=X, labels=labels)

# log-likelihood ratio for each sample
s = mvn.logpdf(X, m1, C1) - mvn.logpdf(X, m0, C0)   # ln p(x|1)/p(x|0)

order = np.argsort(s)
s_sorted, l_sorted = s[order], labels[order]

# k = number of samples (lowest scores) assigned D=0; the rest get D=1
cum1 = np.concatenate([[0], np.cumsum(l_sorted)])
cum0 = np.concatenate([[0], np.cumsum(1 - l_sorted)])
TPR = (n1 - cum1) / n1         
FPR = (n0 - cum0) / n0         
perr = FPR * P0 + (1 - TPR) * P1

log_gamma = np.concatenate([[-np.inf], 0.5 * (s_sorted[:-1] + s_sorted[1:]), [np.inf]])
gamma = np.exp(log_gamma)      

# Find the empirical and theoretical min-P(error) points on the ROC curve
k_emp = np.argmin(perr)
gamma_emp = gamma[k_emp]
gamma_th = (P0 / P1)            # 0-1 loss
k_th = np.searchsorted(s_sorted, np.log(gamma_th), side="right")

print(f"n0={n0}, n1={n1}")
print(f"Theoretical gamma = P(L=0)/P(L=1) = {gamma_th:.4f}")
print(f"  ROC point: FPR={FPR[k_th]:.4f}, TPR={TPR[k_th]:.4f}, P(error)={perr[k_th]:.4f}")
print(f"Empirical min-P(error) gamma = {gamma_emp:.4f} (ln = {log_gamma[k_emp]:.4f})")
print(f"  ROC point: FPR={FPR[k_emp]:.4f}, TPR={TPR[k_emp]:.4f}, P(error)={perr[k_emp]:.4f}")


plt.figure(figsize=(6.5, 6))
plt.plot(FPR, TPR, "b-", lw=1.8, label="ERM ROC (empirical)")
plt.plot(FPR[k_emp], TPR[k_emp], "go", ms=10, mfc="none", mew=2,
         label=f"Empirical min P(error)={perr[k_emp]:.4f}, γ={gamma_emp:.3f}")
plt.plot(FPR[k_th], TPR[k_th], "rx", ms=11, mew=2.5,
         label=f"Theoretical γ={gamma_th:.3f}, P(error)={perr[k_th]:.4f}")
plt.xlabel("P(D=1 | L=0)  (false positive)")
plt.ylabel("P(D=1 | L=1)  (true positive)")
plt.title("ROC curve of minimum expected risk classifier")
plt.grid(alpha=.3); plt.legend(loc="lower right"); plt.xlim(0, 1); plt.ylim(0, 1.02)
plt.tight_layout(); plt.savefig("roc_curve.png", dpi=150)
plt.show()

np.savez("roc_values.npz", gamma=gamma, TPR=TPR, FPR=FPR,
         P_D0_given_L1=1 - TPR, P_D1_given_L0=FPR, perr=perr)
