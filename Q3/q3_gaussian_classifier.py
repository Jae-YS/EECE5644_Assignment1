"""
Run:
  python3 q3_gaussian_classifier_simple.py \
      --wine "wine+quality/winequality-white.csv" \
      --har  "human+activity+recognition+using+smartphones/UCI HAR Dataset" --alpha 0.01
"""
import argparse, os
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

# load data
def load_wine(path):
    D = np.genfromtxt(path, delimiter=";", skip_header=1)
    return D[:, :11], D[:, 11].astype(int)

def load_har(folder):
    rd = lambda *p: np.loadtxt(os.path.join(folder, *p))
    X = np.vstack([rd("train", "X_train.txt"), rd("test", "X_test.txt")])
    y = np.concatenate([rd("train", "y_train.txt"), rd("test", "y_test.txt")]).astype(int)
    return X, y

# Gaussian classifier
def fit(X, y, alpha):
    d, N = X.shape[1], len(y)
    model = {}
    for c in np.unique(y):
        Xc = X[y == c]
        mu, C = Xc.mean(axis=0), np.cov(Xc, rowvar=False)
        ev = np.linalg.eigvalsh(C)
        pos = ev[ev > ev.max() * d * np.finfo(float).eps]       
        lam = alpha * pos.mean()
        Creg = C + lam * np.eye(d)
        model[int(c)] = dict(mu=mu, L=cholesky(Creg, lower=True), logprior=np.log(len(Xc) / N),
                             n=len(Xc), rank=len(pos), lam=lam,
                             cond=(ev.max() / abs(ev.min()), np.linalg.cond(Creg)))
    return model

def classify(X, model):
    """MAP rule: argmax_j  ln P(L=j) - 0.5 (x-mu)' Creg^-1 (x-mu) - 0.5 ln|Creg|  (constant dropped)."""
    cls = sorted(model); S = []
    for c in cls:
        m = model[c]
        Z = solve_triangular(m["L"], (X - m["mu"]).T, lower=True)
        S.append(m["logprior"] - 0.5 * (Z ** 2).sum(axis=0) - np.log(np.diag(m["L"])).sum())
    return np.array(cls)[np.argmax(S, axis=0)], cls

def confusion(y, D, cls):
    M = np.array([[np.sum((D == i) & (y == j)) for j in cls] for i in cls])  
    return M, M / M.sum(axis=0)

# visualization 
def plot_pca(X, y, D, name, standardize):
    Z = X - X.mean(axis=0)
    if standardize:
        Z = Z / X.std(axis=0)
    _, s, Vt = np.linalg.svd(Z, full_matrices=False)
    P, var = Z @ Vt[:3].T, s[:3] ** 2 / (s ** 2).sum()
    ok = D == y
    fig = plt.figure(figsize=(18, 6)); fig.suptitle(name)
    ax = fig.add_subplot(1, 3, 1)
    for c in np.unique(y):
        ax.scatter(P[y == c, 0], P[y == c, 1], s=6, alpha=0.5, label=f"class {c}")
    ax.set_xlabel(f"PC1 ({var[0]:.1%})"); ax.set_ylabel(f"PC2 ({var[1]:.1%})")
    ax.set_title("true classes"); ax.legend(fontsize=7, markerscale=2)
    for k, (mask, col, ttl) in enumerate([(ok, "green", f"correct decisions ({ok.sum()})"),
                                          (~ok, "red", f"incorrect decisions ({(~ok).sum()})")]):
        ax = fig.add_subplot(1, 3, k + 2, projection="3d")
        ax.scatter(*P[mask].T, s=4, alpha=0.4, c=col)
        ax.set_xlim(P[:, 0].min(), P[:, 0].max()); ax.set_ylim(P[:, 1].min(), P[:, 1].max()); ax.set_zlim(P[:, 2].min(), P[:, 2].max())
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3"); ax.set_title(ttl)
    plt.tight_layout(); plt.savefig(f"q3_out/{name}_pca.png", dpi=140)

# main
def run(name, X, y, alpha, plot=True, standardize_pca=False):
    print("=" * 70); print(f"{name}: N={len(y)}, d={X.shape[1]}, alpha={alpha}")
    model = fit(X, y, alpha)
    for c, m in model.items():
        print(f"  class {c}: n={m['n']:>5}  prior={m['n']/len(y):.4f}  rank={m['rank']}/{X.shape[1]}  "
              f"cond {m['cond'][0]:.1e} -> {m['cond'][1]:.1e}  lambda={m['lam']:.3g}")
    D, cls = classify(X, model)
    print(f"\nP(error) = {np.mean(D != y):.4f}  ({np.sum(D != y)} errors out of {len(y)})")
    M, rate = confusion(y, D, cls)
    print("\nConfusion counts (rows = decision D, columns = true label L), classes", cls); print(M)
    print("\nP(D=i | L=j):"); print(np.round(rate, 3))
    print("\nTraining error vs alpha:")
    for a in [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        print(f"  alpha={a:<7g} P(error)={np.mean(classify(X, fit(X, y, a))[0] != y):.4f}")
    print("\nGaussianity check (Gaussian: skew = 0, excess kurtosis = 0):")
    for c in cls:
        Xc = X[y == c]
        print(f"  class {c}: mean|skew|={np.abs(skew(Xc, axis=0)).mean():.3f}  "
              f"mean excess kurtosis={kurtosis(Xc, axis=0).mean():.3f}")
    if plot:
        plot_pca(X, y, D, name, standardize_pca)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wine"); ap.add_argument("--har"); ap.add_argument("--alpha", type=float, default=0.01)
    a = ap.parse_args(); os.makedirs("q3_out", exist_ok=True)
    if a.wine:
        Xw, yw = load_wine(a.wine)
        run("wine_raw", Xw, yw, a.alpha, plot=False)                       
        run("wine_standardized", (Xw - Xw.mean(axis=0)) / Xw.std(axis=0), yw, a.alpha)   
    if a.har:
        run("har", *load_har(a.har), a.alpha)
    plt.show()
