# PROJECT_CARD — Amazon Reviews '23

**Project track:** Track B — Amazon Reviews '23: Text, Structured Data, and Multimodal Analysis  


## Provisional project title

**Predicting Recorded Helpful Votes from Review Text and Structured Context: A Multimodal Study of Musical Instruments and Video Games**

## Real-world problem and decision supported

Review platforms and marketplace analysts need to understand which observable review characteristics are associated with the attention and helpfulness feedback that reviews receive. Being able to estimate *recorded helpful-vote counts*, and to identify the contribution of text versus structured context, may support review engagement analysis, research into review presentation or ranking, and the prioritization of reviews for further examination.

The project studies **associations and prediction**, not causal effects or the intrinsic truthfulness of a review. A high helpful-vote count is not evidence that a review is honest, authentic, unbiased, or factually correct. 

## Unit of analysis and unit of prediction

- **Unit of analysis:** One individual review record, associated with a reviewer and a product.
- **Unit of prediction:** One individual review record.
- **Observational keys:** `user_id` identifies the reviewer and `parent_asin`/`asin` identifies the product, subject to the source schema and audit results.
- **Inputs potentially available at review level:** `title`, `text`, `rating`, `verified_purchase`, `timestamp`, reviewer/product identifiers, and joined product metadata where available.
- **Task:** Non-negative **count prediction / regression**.

## Target population and scope

The primary scope is reviews in the `Musical_Instruments` and `Video_Games` categories represented in the selected Amazon Reviews '23 files. Results will be reported overall and separately by category.

The intended prediction setting is **estimating the helpful-vote count recorded in the available dataset snapshot for held-out reviews**, conditional on carefully specified features. It is **not** prediction of votes received during a future 7-day or 30-day period: individual helpful-vote timestamps and standardized exposure windows are not available in the chosen data.

Any conclusions about other categories, users, or periods will be explicitly treated as untested generalizations.

## Main analytical / predictive question

> **Can textual characteristics and structured information about a review, its reviewer, and its product predict the number of recorded helpful votes? Will combining review text and structured features outperform either information source alone?**

Two complementary objectives guide the project:

1. **Explanatory / exploratory:** Study how recorded vote counts vary with review text, star rating, verified purchase, publication time, product attributes, and observed reviewer/product activity.
2. **Predictive / experimental:** Compare a simple count baseline, text-only models, structured-only models, and a model combining both sources, using leakage-aware evaluation.

This question was selected over narrower alternatives focused only on text characteristics or reviewer/product activity because it has a measurable outcome and directly motivates meaningful comparisons across information sources.

## Target variable / desired output

- **Target column:** `helpful_vote`.
- **Data type:** Non-negative integer count (after validating and handling missing or invalid entries).
- **Model output:** An estimated non-negative count of **recorded** helpful votes per review; optionally a count distribution or uncertainty estimate if a suitable model supports it.


## Initial baseline

**Primary simple baseline: Training-set median predictor.** For each held-out review, predict the median `helpful_vote` count computed **only from the training data**. This gives an interpretable and robust reference for a skewed count target.


## Proposed evaluation metrics

**Primary metric:** Mean Absolute Error (**MAE**) on the held-out review counts, since it is expressed directly in votes and permits a clear comparison with the median baseline.

**Secondary metrics:**

- **RMSE:** Highlights particularly large count-prediction errors.
- **MAE on `log1p(helpful_vote)` or RMSLE (for non-negative predictions):** Assesses performance across the highly skewed count scale.
- **R²:** A supplementary measure of explained variability, interpreted cautiously for sparse, skewed counts.
- **Subgroup error:** MAE (and where useful RMSE) separately by product category, review-age band, and zero versus positive vote count.
- **Optional diagnostic:** Precision/recall/F1 for distinguishing zero from positive vote counts, **only as a secondary derived task**, not as a replacement for the regression objective.


## Initial training, validation, and test split plan

**Primary plan: Chronological holdout at the review level.** Sort usable reviews by publication timestamp and provisionally allocate the earliest **70%** to training, the next **15%** to validation, and the latest **15%** to testing. Exact date boundaries and proportions will be frozen before model tuning. If timestamps are missing or invalid, their treatment will be documented before splitting.

## Main information-leakage risks

| Risk | Proposed prevention |
| --- | --- |
| Using `helpful_vote` or a transformation of it as an input | Exclude the target and all direct derivatives from predictors |
| Fitting TF-IDF, imputation, scaling, feature selection, or encoders on the entire dataset | Fit each transformation on the training fold only, using pipelines |
| Repeated reviewers or products across partitions | Report what the chronological evaluation measures; add reviewer-/product-disjoint checks |
| Duplicate or near-identical review content across splits | Audit fingerprints and keep duplicates together or remove them before splitting |
| Reviewer/product activity calculated from all available records | Construct past-only, training-safe aggregates; do not include future/test reviews |
| Product average rating, rating count, or other snapshot aggregates incorporating later reviews | Exclude initially or reconstruct historical versions from valid past data |
| Product price or metadata recorded after the review date | Document timing and exclude any feature not available at the intended prediction time |
| Target-based category/user/product statistics using all partitions | Estimate statistics from training data only |
| Review timestamp encoding accumulated vote exposure | Analyze review-age bias and qualify the interpretation, even with a chronological split |

## Proposed unsupervised learning / pattern-discovery component

Represent review text using **TF-IDF** and, if feasible, pretrained text embeddings. Apply a dimensionality-reduction method such as **Truncated SVD** (for sparse TF-IDF) or PCA (for suitable dense features), followed by a clustering method such as **K-means**.

The analysis will ask whether distinct review-writing patterns or topics emerge and whether their distributions of recorded helpful votes differ. Clusters will be interpreted using representative reviews, salient terms, and category/rating distributions. The number of clusters and robustness will be assessed using appropriate criteria (such as silhouette score, stability across seeds/subsamples, and interpretability).


## Track-specific advanced component

**Primary extension: Meaningful multimodal fusion of text and structured context.**

The project will combine at least two distinct information sources:

1. **Textual:** Review title and body, represented with TF-IDF or a pretrained text embedding, potentially augmented with interpretable text features such as length and punctuation frequency.
2. **Structured/contextual:** Star rating, verified-purchase status, product category, permitted time features, available product metadata, and carefully constructed historical activity measures.

The central comparison is **text-only vs. structured-only vs. fused** under identical evaluation conditions. A documented early-fusion feature pipeline or late-fusion ensemble will be used. **Modality ablation** will remove one source at a time to quantify its contribution.

Further analyses may cover sentiment/rating disagreement, category-specific performance, temporal shift, and representative failure cases. Images are audited for availability in Phase 1 but are **not** assumed to be part of the main model unless their accessibility and analytical value are established. The analysis qualifies as multimodal through the deliberate combination and comparison of text and structured information—not merely by merging columns into one table.

## Ethical, privacy, and potential bias risks

- **Privacy:** Reviewer identifiers are pseudonymous but can still support linkage across records. Use them only where necessary for joins, leakage controls, and permitted historical features; do not publish identifying review-level outputs or attempt re-identification.
- **Exposure and selection bias:** Older, more visible, or popular-product reviews may accumulate more votes. Helpful-vote counts may reflect exposure and platform mechanisms rather than textual value alone.
- **Category and subgroup bias:** Reviews in `Video_Games` and `Musical_Instruments` may have different audiences, review conventions, and voting patterns. Performance must be reported separately to avoid hiding disparities.
- **Missingness and sampling bias:** Missing text, product metadata, prices, or images may be systematic. Reservoir sampling from two selected categories does not establish representativeness of Amazon as a whole.
- **Label ambiguity:** Helpful votes are a behavioral signal, **not ground truth for review accuracy, authenticity, or customer benefit**.
- **Misuse:** Model estimates should not automatically remove, penalize, or suppress reviews, nor be presented as proof of fraud, deception, or quality.
- **Causal overclaiming:** Observed associations (e.g., longer text correlating with more votes) do not demonstrate that changing that feature will cause a vote increase.
- **Dataset terms:** The dataset's license, redistribution conditions, and usage requirements will be checked before sharing any raw records, images, or derivative artifacts.

---

### Project-card maintenance

This document records the **initial plan**, not completed model results. Choices involving eligibility, feature availability, metadata timing, exact split cutoffs, metrics, and model design will be updated transparently as evidence becomes available. 


Mozhan Homavandi 
402100572