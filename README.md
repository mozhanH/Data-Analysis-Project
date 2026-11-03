# Amazon Reviews '23 — Phase 1

**Data Analysis Course Project | Track B: Text and Multimodal Data**

## Project Overview

This project explores whether **review text and structured information** can help predict the number of **recorded helpful votes** a product review receives. Phase 1 focuses on understanding the data, assessing its quality, and conducting exploratory data analysis (EDA). Predictive modeling is planned for later phases.

## Dataset

- **Source:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)
- **Categories:** `Musical_Instruments` and `Video_Games`
- **Sample:** Up to 40,000 reviews per category, sampled reproducibly (`SEED = 42`)
- **Data:** Review text, ratings, timestamps, verified-purchase status, helpful votes, and available product metadata
- **Target for future modeling:** `helpful_vote` (a count, **not** an indicator of review authenticity)

## Phase 1 Work

- Loaded and sampled review data from existing Kaggle inputs.
- Inspected dataset structure, metadata coverage, missing values, and duplicates.
- Explored distributions and relationships involving review length, ratings, helpful votes, time, and product/reviewer activity.
- Identified potential data leakage and outlined baselines and evaluation plans for subsequent phases.

## Repository Files

| File | Description |
| --- | --- |
| `phase1-da.ipynb` | Main Phase 1 analysis notebook |
| `phase1-da.html` | HTML version of the notebook |
| `compare-product-score.ipynb` | Supplementary category-comparison notebook |
| `Phase1_Report.docx` | Written Phase 1 report |
| `PROJECT_CARD.md` | Project definition and proposed methodology |

## Running the Notebook

1. Open `phase1-da.ipynb` in **Kaggle**.
2. Attach the review and metadata datasets for both categories via **Add Input**.
3. Run the cells in order. The notebook searches for files under `/kaggle/input`; adjust paths if needed.

## Next Steps

Build and compare **text-only**, **structured-only**, and **combined (multimodal)** models for helpful-vote prediction using leakage-aware train/validation/test splits.
