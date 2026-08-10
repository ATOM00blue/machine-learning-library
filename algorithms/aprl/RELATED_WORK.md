# Custom APRL Algorithm: Related Work and Novelty Boundary

APRL is a custom composition, not a claim that prototypes, local experts,
relevance-weighted distances, residual fitting, or distance-aware confidence are
individually new. This note records the closest conceptual families found during
the initial design review.

## Related foundations

1. **k-means++ initialization.** APRL uses distance-squared sampling to initialize
   its prototype centers before ordinary Lloyd iterations. This comes directly
   from Arthur and Vassilvitskii, [*k-means++: The Advantages of Careful
   Seeding*](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf) (2007).

2. **Radial-basis-function networks.** Soft Gaussian distance weights around
   learned centers are longstanding. Wettschereck and Dietterich discuss both
   interpretable centers and learning their locations in [*Improving the
   Performance of Radial Basis Function Networks by Learning Center
   Locations*](https://proceedings.neurips.cc/paper_files/paper/1991/file/97e8527feaf77a97fc38f34216141515-Paper.pdf)
   (1991).

3. **Mixtures of local experts.** Gated expert models that partition an input
   space and combine specialized predictors precede APRL. See Jordan and Jacobs,
   [*Hierarchies of Adaptive
   Experts*](https://papers.nips.cc/paper/1991/file/59b90e1005a220e2ebc542eb9d950b1e-Paper.pdf)
   (1991), and Nowlan and Hinton, [*Evaluation of Adaptive Mixtures of Competing
   Experts*](https://papers.nips.cc/paper_files/paper/1990/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html)
   (1990).

4. **Locally weighted polynomial regression.** Fitting polynomial models with
   weights that decrease with distance is the central idea of LOESS. See
   Cleveland, [*Robust Locally Weighted Regression and Smoothing
   Scatterplots*](https://doi.org/10.1080/01621459.1979.10481038) (1979).

5. **Prototype and relevance learning.** Learning vector quantization provides a
   major body of prototype-based classification work. Feature or matrix relevance
   has also been learned jointly with prototypes; see Schneider, Biehl, and
   Hammer, [*Adaptive Relevance Matrices in Learning Vector
   Quantization*](https://doi.org/10.1162/neco.2009.11-08-908) (2009), and Biehl,
   Ghosh, and Hammer, [*Dynamics and Generalization Ability of LVQ
   Algorithms*](https://www.jmlr.org/papers/v8/biehl07a.html) (2007).

6. **Prototype-based confidence and distribution mismatch.** Prototype agreement
   and distance have been used for interpretable confidence and mismatch
   detection. See Arik and Pfister,
   [*ProtoAttend*](https://www.jmlr.org/papers/v21/20-042.html) (2020).

7. **Calibrated uncertainty is a separate problem.** APRL's uncertainty output is
   heuristic. Distribution-free uncertainty requires an explicit calibration
   procedure; Angelopoulos and Bates provide an introduction in [*A Gentle
   Introduction to Conformal Prediction and Distribution-Free Uncertainty
   Quantification*](https://arxiv.org/abs/2107.07511) (2021).

## What is custom in APRL

The repository's design contribution is the particular end-to-end combination:

- marginal supervised relevance weights with a nonzero interaction-preserving floor;
- unsupervised prototypes learned in that weighted space;
- one stable global ridge prediction;
- per-prototype polynomial experts trained only on global residuals;
- soft prototype averaging;
- a quartic distance-and-density gate that suppresses local corrections away from data;
- a shared implementation for classification, regression, uncertainty, and exact batch updates.

That combination is useful as an experimental algorithm and educational reference.
It must not be described as a novel scientific discovery until a broader literature
review, ablation studies, statistical comparisons, and peer review establish that
claim.

## Evidence still needed

Before making a research contribution claim:

1. search patents and a wider set of local-model, RBF, mixture-of-experts,
   prototype-regression, and residual-model literature;
2. compare against tuned gradient boosting, random forests, kernel methods, GAMs,
   RBF networks, and modern tabular neural models;
3. run repeated nested cross-validation across a broad OpenML benchmark suite;
4. ablate relevance weighting, quadratic terms, residual learning, density gating,
   and uncertainty penalties;
5. measure calibration, coverage under shift, runtime, and memory;
6. publish negative results and sensitivity to every important hyperparameter.
