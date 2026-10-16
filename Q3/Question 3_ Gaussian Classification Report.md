# Question 3: Gaussian Minimum-P(error) Classification of the Wine Quality and HAR Datasets

*Name / Course / Date: \[fill in\]*

## 1. Model, Estimation and Decision Rule

**Assumptions.** Let **x** ∈ ℝ^d be the features and L the class label. Each class-conditional pdf is Gaussian with its own covariance, p(**x**|L=j) = g(**x** | **μ**\_j, **C**\_j) = (2π)^(−d/2) |**C**\_j|^(−1/2) exp(−½ (**x**−**μ**\_j)ᵀ **C**\_j⁻¹ (**x**−**μ**\_j)). Training samples are assumed independent, and the loss is 0-1.

**Estimation.** From the n_j samples of class j (N samples in total):

μ̂\_j = (1/n_j) Σ **x**, Ĉ_j = (1/(n_j−1)) Σ (**x**−μ̂\_j)(**x**−μ̂\_j)ᵀ, P̂(L=j) = n_j / N.

**Regularization.** Ĉ_j is singular when n_j ≤ d (wine class 9 has 5 samples in 11 dimensions, rank 4), and for HAR the covariances are rank-deficient (rank 466 to 469 of 561, condition number about 10¹⁶) because many features are correlated or nearly constant. I used

**C**\_reg,j = Ĉ_j + λ_j **I**, λ_j = α · (mean of the non-zero eigenvalues of Ĉ_j).

Since Ĉ_j is positive semi-definite, all eigenvalues of **C**\_reg,j are at least λ_j > 0, so it is invertible (the condition numbers fall to about 10⁴ for HAR). The main results use α = 0.01; Table 3 shows other values. Wine features have very different scales (variance about 10³ for total sulfur dioxide versus 10⁻⁶ for density), and λ**I** is not scale-invariant, so I report wine on **z-scored features** as the main result and on raw features for comparison. HAR features are already in \[−1, 1\].

**Decision rule.** With 0-1 loss the minimum-P(error) rule is the MAP rule,

D(**x**) = argmax_j \[ ln P̂(L=j) − ½ (**x**−μ̂\_j)ᵀ **C**\_reg,j⁻¹ (**x**−μ̂\_j) − ½ ln |**C**\_reg,j| \],

with quadratic decision boundaries because the covariances differ. It was applied to all N training samples, P̂(error) is the fraction with D ≠ L, and the confusion matrix estimates P(D=i | L=j) = #(D=i, L=j) / n_j.

## 2. Results

**Table 1: error probability estimates (training samples, α = 0.01)**

| Dataset | P̂(error) | Errors | Reference |
| --- | --- | --- | --- |
| Wine, standardized (main) | **0.5045** | 2471 / 4898 | always guessing class 6 gives 0.5512 |
| Wine, raw | 0.6933 | 3396 / 4898 |  |
| HAR | **0.0369** | 380 / 10299 |  |

**Table 2a: confusion matrix, wine standardized** (counts; rows = decision D, columns = true label L)

|  | L=3 | L=4 | L=5 | L=6 | L=7 | L=8 | L=9 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **D=3** | 8 | 3 | 10 | 2 | 0 | 0 | 0 |
| **D=4** | 2 | 39 | 48 | 28 | 2 | 0 | 0 |
| **D=5** | 3 | 70 | 804 | 461 | 27 | 3 | 0 |
| **D=6** | 4 | 39 | 436 | 909 | 192 | 47 | 0 |
| **D=7** | 2 | 11 | 158 | 782 | 647 | 110 | 0 |
| **D=8** | 1 | 1 | 1 | 16 | 12 | 15 | 0 |
| **D=9** | 0 | 0 | 0 | 0 | 0 | 0 | 5 |

Correct-decision rates P(D=j | L=j) for classes 3 to 9 are 0.40, 0.24, 0.55, 0.41, 0.74, 0.09 and 1.00. (For raw wine, 3275 of the 4898 samples are assigned to class 7.)

**Table 2b: confusion matrix, HAR** (counts; rows = decision D, columns = true label L)

|  | L=1 | L=2 | L=3 | L=4 | L=5 | L=6 |
| --- | --- | --- | --- | --- | --- | --- |
| **D=1** | 1721 | 0 | 0 | 0 | 0 | 0 |
| **D=2** | 1 | 1544 | 48 | 0 | 0 | 0 |
| **D=3** | 0 | 0 | 1358 | 0 | 0 | 0 |
| **D=4** | 0 | 0 | 0 | 1446 | 0 | 0 |
| **D=5** | 0 | 0 | 0 | 331 | 1906 | 0 |
| **D=6** | 0 | 0 | 0 | 0 | 0 | 1944 |

The 380 errors are 331 class-4 samples decided as class 5 (18.6% of class 4), 48 class-3 samples decided as class 2 (3.4%), and 1 class-1 sample decided as class 2. \[Write the activity names: classes 4 and 5 are expected to be sitting and standing, and classes 3 and 2 walking downstairs and upstairs; check `activity_labels.txt`.\]

**Table 3: training error versus α**

| α | 10⁻⁶ | 10⁻⁴ | 10⁻³ | 10⁻² | 10⁻¹ | 1 |
| --- | --- | --- | --- | --- | --- | --- |
| Wine, raw | 0.5102 | 0.5041 | 0.6002 | 0.6933 | 0.7556 | 0.7871 |
| Wine, standardized | 0.5002 | 0.5006 | 0.5031 | 0.5045 | 0.5159 | 0.5561 |
| HAR | 0.0166 | 0.0022 | 0.0083 | 0.0369 | 0.0898 | 0.2952 |

## 3. Visualization

Figure 1: HAR projected onto the first principal components

*Figure 1: HAR samples projected onto the first principal components. Left: PC1 vs PC2 colored by true class. Middle and right: first three PCs, with the 9919 correct decisions and the 380 incorrect decisions shown separately.*

The first principal component (62.2% of the variance) separates the six activities into two groups: the static activities (classes 4 to 6, left) and the moving activities (classes 1 to 3, right). Class 6 forms a compact cluster at the far left, classes 4 and 5 overlap heavily with each other, and classes 1, 2 and 3 overlap in an elongated band, with class 3 having the widest spread. The correctly classified samples form compact groups on the static side and an elongated band on the moving side. The 380 errors are not spread out but form two small localized clusters, one in the static region and one at the start of the moving band, which matches the confusion matrix (class 4 versus 5, and class 3 versus 2). The overlap in the 2-D view is larger than the classifier's actual error, because two components keep only about 67% of the variance and the classifier uses all 561 features.

Figure 2: Standardized wine projected onto the first principal components

*Figure 2: Standardized wine samples projected onto the first principal components (same layout as Figure 1), with the 2427 correct and 2471 incorrect decisions shown separately.*

For wine, all the quality classes lie in a single dense, overlapping cloud, and the first two components explain only 43.6% of the variance (29.3% + 14.3%). There is only a weak trend (class 7 sits slightly toward negative PC1 and class 5 toward positive PC1), and no class forms a separate cluster. The correct and the incorrect decisions occupy the same region with a similar spread, so the errors are not confined to any part of the cloud. This is consistent with the heavy class overlap and explains the high error rate (0.5045).

## 4. Discussion

**Are Gaussian models appropriate?** For a Gaussian, skewness and excess kurtosis are 0. Averaged over the features:

| Wine class | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mean abs. skew | 0.89 | 1.28 | 1.01 | 1.36 | 0.79 | 0.99 | 0.79 |
| mean excess kurtosis | 1.1 | 5.1 | 3.6 | 7.3 | 1.9 | 2.3 | −0.7 |

| HAR class | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| mean abs. skew | 1.25 | 1.26 | 1.31 | 5.53 | 4.86 | 6.27 |
| mean excess kurtosis | 5.5 | 4.8 | 5.7 | 79.7 | 65.2 | 98.6 |

*Wine:* the features are skewed and heavy-tailed (positive physicochemical measurements), the quality classes overlap heavily because neighboring scores have similar chemistry, and the class sizes range from 5 to 2198. The Gaussian model is a poor fit, and the standardized classifier (0.5045) improves only modestly on the 0.5512 of always guessing the majority class. *HAR:* the features are bounded in \[−1, 1\], so exact Gaussianity is impossible, and the static activities (classes 4 to 6) are far from Gaussian (skew about 5 to 6, kurtosis 65 to 99). The classes are nevertheless well separated relative to their spread, so the classifier is accurate (0.0369). Overlapping windows from only 30 subjects also mean the samples are not independent.

**How the model choice affected the results.**

- *Priors and overlap (wine).* The rule weights classes by their priors, so rare classes are chosen seldom (class 8 is correct for only 9% of its samples). Errors fall mostly between neighboring qualities, for example 782 of the 2198 class-6 samples are decided as class 7, because the Gaussian densities of adjacent classes overlap heavily. Class 9 (5 samples, rank-4 covariance) is classified 5 of 5 correctly, which is overfitting to training data rather than real skill.
- *Scaling and regularization (wine).* On raw features, λ is set by the large-variance features and swamps the small ones, which distorts the densities and gives an error (0.6933) worse than the majority-class baseline; standardizing reduces it to 0.5045.
- *HAR.* Almost all errors come from two physically similar pairs (class 4 vs 5, class 3 vs 2), while the other activities are nearly perfect. A class-specific 561×561 covariance has about 157,000 free parameters, so the estimates are noisy and depend on regularization. A very small α (10⁻⁶) leaves near-singular directions that add noise (0.0166), a moderate α (10⁻⁴) is best (0.0022), and a large α makes the classes look spherical and loses the correlation structure (0.2952 at α = 1), which is a bias-variance trade-off.
- *Caveat.* All errors are training errors and are therefore optimistic, and choosing α from the sweep on the same data favors small values. A held-out set or cross-validation would give an unbiased estimate and a principled α.

## 5. Conclusion

The Gaussian minimum-P(error) classifier works well on HAR (P̂(error) = 0.0369) despite non-Gaussian features, because the classes are well separated. On wine it performs poorly (0.5045 standardized, 0.6933 raw) because the features are skewed, the classes overlap heavily and are unbalanced, and the Gaussian model fits poorly. Regularization was essential, and its effect depends strongly on feature scaling and on α.

## References

1. Cortez, P., Cerdeira, A., Almeida, F., Matos, T., and Reis, J. Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47(4):547–553, 2009.
2. Anguita, D., Ghio, A., Oneto, L., Parra, X., and Reyes-Ortiz, J. L. A public domain dataset for human activity recognition using smartphones. *ESANN 2013*, Bruges, Belgium, 2013.