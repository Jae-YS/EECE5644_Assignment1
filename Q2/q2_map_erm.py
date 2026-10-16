import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

rng = np.random.default_rng(7)
N = 10000
priors = np.array([0.3, 0.3, 0.4])

# ---- Four Gaussian components (3-D) ----
means = np.array([[0.0, 0.0, 0.0],     # component a -> class 1
                  [2.5, 0.0, 0.0],     # component b -> class 2
                  [1.25, 2.5, 0.0],    # component c -> half of class 3
                  [1.25, 1.0, 2.5]])   # component d -> half of class 3
covs = [np.array([[1.0, 0.3, 0.0], [0.3, 1.0, 0.2], [0.0, 0.2, 1.0]]),
        np.array([[1.2, -0.3, 0.0], [-0.3, 0.8, 0.0], [0.0, 0.0, 1.0]]),
        np.array([[0.9, 0.0, 0.3], [0.0, 1.1, 0.0], [0.3, 0.0, 1.0]]),
        np.array([[1.0, 0.2, -0.2], [0.2, 1.2, 0.0], [-0.2, 0.0, 0.8]])]

# check separation: pairwise mean distances / average std of the components
avg_std = np.mean([np.sqrt(np.diag(C)).mean() for C in covs])
print(f"average std of components = {avg_std:.3f}")
for i in range(4):
    for j in range(i + 1, 4):
        print(f"  dist(m{'abcd'[i]}, m{'abcd'[j]}) = {np.linalg.norm(means[i]-means[j]):.2f} = {np.linalg.norm(means[i]-means[j])/avg_std:.2f} std")

# ---- A.1 Generate data ----
labels = rng.choice([1, 2, 3], size=N, p=priors)          # true labels
comp = np.where(labels == 1, 0, np.where(labels == 2, 1, 2 + rng.integers(0, 2, N)))
X = np.empty((N, 3))
for k in range(4):
    idx = comp == k
    X[idx] = rng.multivariate_normal(means[k], covs[k], size=idx.sum())
np.savez("q2_data.npz", X=X, labels=labels)
print("class counts:", [int((labels == j).sum()) for j in (1, 2, 3)])

# ---- Class-conditional pdfs ----
pa, pb, pc, pd = [mvn.pdf(X, means[k], covs[k]) for k in range(4)]
px_given = np.column_stack([pa, pb, 0.5 * pc + 0.5 * pd])   # p(x|L=j), j=1..3
joint = px_given * priors                                    # p(x|L=j)P(L=j)
post = joint / joint.sum(axis=1, keepdims=True)              # P(L=j|x)

def confusion(D):   # entry (i,j) = P(D=i | L=j)
    M = np.zeros((3, 3))
    for j in (1, 2, 3):
        for i in (1, 2, 3):
            M[i-1, j-1] = np.mean(D[labels == j] == i)
    return M

def scatter3d(D, title, fname):
    fig = plt.figure(figsize=(8.5, 7)); ax = fig.add_subplot(111, projection="3d")
    shapes = {1: "o", 2: "^", 3: "s"}
    for ok, col in [(True, "green"), (False, "red")]:
        for j in (1, 2, 3):
            m = (labels == j) & ((D == j) == ok)
            ax.scatter(X[m, 0], X[m, 1], X[m, 2], marker=shapes[j], c=col, s=10 if ok else 18,
                       alpha=0.35 if ok else 0.9, linewidths=0.3,
                       label=f"Class {j} {'correct' if ok else 'incorrect'}")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3"); ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, markerscale=1.6)
    plt.tight_layout(); plt.savefig(fname, dpi=140); plt.close()

# ---- A.2 MAP (0-1 loss) classifier ----
D_map = np.argmax(post, axis=1) + 1
Cm = confusion(D_map)
print("\nMAP confusion matrix P(D=i|L=j) (rows i, cols j):\n", np.round(Cm, 4))
print("MAP P(error) =", round(np.mean(D_map != labels), 4))
scatter3d(D_map, "MAP classifier (0-1 loss): shape = true class, green = correct, red = wrong", "q2_scatter_map.png")

# ---- Part B: ERM with loss matrices ----
L10  = np.array([[0, 1, 10], [1, 0, 10], [1, 1, 0]], float)
L100 = np.array([[0, 1, 100], [1, 0, 100], [1, 1, 0]], float)
L01  = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)

def erm(L):
    risk = post @ L.T                    # risk[n,i] = sum_j L[i,j] P(L=j|x_n)
    return np.argmin(risk, axis=1) + 1

def emp_risk(D, L):
    return L[D - 1, labels - 1].mean()

summary = {}
for name, L, f in [("Lambda_10", L10, "q2_scatter_l10.png"), ("Lambda_100", L100, "q2_scatter_l100.png")]:
    D = erm(L)
    M = confusion(D)
    r = emp_risk(D, L); r_map = emp_risk(D_map, L)
    summary[name] = (M, r, r_map, D)
    print(f"\n{name}: confusion matrix P(D=i|L=j):\n", np.round(M, 4))
    print(f"  min expected risk (ERM) = {r:.4f}   |   MAP classifier under same loss = {r_map:.4f}")
    print("  decision counts (D=1,2,3):", [int((D == i).sum()) for i in (1, 2, 3)],
          "  P(error) =", round(np.mean(D != labels), 4))
    scatter3d(D, f"ERM with {name}: shape = true class, green = correct, red = wrong", f)

# MAP sanity: ERM with 0-1 loss should equal MAP
assert np.all(erm(L01) == D_map)

# ---- Confusion-matrix heat maps ----
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, (t, M) in zip(axs, [("MAP (0-1 loss)", Cm), ("ERM Λ10", summary["Lambda_10"][0]), ("ERM Λ100", summary["Lambda_100"][0])]):
    im = ax.imshow(M, vmin=0, vmax=1, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", color="white" if M[i,j] > .5 else "black")
    ax.set_xticks(range(3)); ax.set_xticklabels(["L=1", "L=2", "L=3"]); ax.set_yticks(range(3)); ax.set_yticklabels(["D=1", "D=2", "D=3"])
    ax.set_title(t); ax.set_xlabel("true label"); ax.set_ylabel("decision")
plt.tight_layout(); plt.savefig("q2_confusions.png", dpi=140)
np.savez("q2_results.npz", Cm=Cm, M10=summary["Lambda_10"][0], M100=summary["Lambda_100"][0])
