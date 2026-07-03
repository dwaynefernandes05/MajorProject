# OBJ3 - Explainable AI for RFP to SKU Matching

This folder contains the explainability layer for the RFP-to-SKU matching system. It turns the Objective 1 matcher and the Objective 2 robustness experiments into a single analysis pipeline that can be described in a research paper.

## Research Goal

OBJ3 answers three questions:

1. Why did an RFP and SKU fully match?
2. Why did they only partially match?
3. Why did they fail to match at all?

The notebook explains the matcher in human-readable terms, combines evidence from the robustness experiments, and exports paper-ready tables and figures.

## System Overview

The pipeline is organized into three layers:

1. Objective 1 provides the matching engine and score decomposition.
2. Objective 2 provides stress-test evidence from numeric perturbation, unit variation, long-document bias, missing parameters, noise injection, and positional bias.
3. Objective 3 converts the score signals into explainability outputs and summary reports.

In practice, OBJ3 does not retrain the matcher. It reuses the saved Objective 1 artifacts and interprets the resulting scores.

## Main Notebook

- [EXPLAINABLE_AI_PIPELINE.ipynb](EXPLAINABLE_AI_PIPELINE.ipynb)

## Data and Artifacts Used

- Objective 1 artifacts from `obj1_model_artifacts/`
- RFP source data from `datasets/rfp_specs_7000.csv`
- Product catalog from `datasets/product_catalog.csv`
- Objective 2 CSV results from `OBJ2/EXP1`, `OBJ2/EXP2`, `OBJ2/EXP5`, `OBJ2/EXP6`, and `OBJ2/EXP7`
- Notebook-only sources for `OBJ2/EXP3` and `OBJ2/EXP4` when exported CSVs are not available

The notebook also reconstructs the top-1 match table if `match_df.parquet` is unavailable.

## How The Pipeline Works

### 1. Load and prepare data

The notebook loads the RFP table, product table, embeddings, baseline results, and the saved match table. If parquet files are missing or unsupported, it falls back to CSV sources in `datasets/`.

### 2. Merge Objective 2 evidence

OBJ2 results are loaded into a unified summary table. This lets the notebook report how the matcher behaves under different perturbation regimes and group the results by theme.

### 3. Build explainability outputs

For each RFP, the notebook selects the top-ranked SKU and generates a compact explanation record. The explanation includes:

- the predicted category: matching, partially matching, or not matching
- the final compliance score
- the structured, semantic, and standards score components
- a SHAP-style attribution of those score components relative to the baseline
- a short natural-language explanation

The current explanation layer is based on score decomposition rather than a black-box post-hoc explainer over the raw text embeddings. The SHAP view is therefore best described as baseline-centered attribution over the final score components.

### 4. Export reports and figures

The notebook writes tables and plots to `OBJ3/results/` for later use in analysis, presentation, and paper writing.

## What The SHAP Layer Means Here

The notebook treats the final compliance score as an additive combination of the component scores. For the top-1 explanation, it compares each component against a baseline and produces a contribution value for:

- Structured score
- Semantic score
- Standards score

This makes the explanation easy to cite in a paper because the contributions map directly back to the matcher's score channels.

## Outputs

The notebook writes the following artifacts to `OBJ3/results/`:

- `obj3_top1_explanations.csv`
- `obj3_top1_summary.csv`
- `obj3_shap_summary.png`
- `obj3_sample_report.csv`
- `obj3_obj2_theme_summary.csv`
- `obj3_category_visualization.png`

These outputs are the main evidence tables and figures for a research paper.

## Suggested Paper Narrative

If you are writing a paper, the easiest structure is:

1. Problem statement: explain why automated RFP-to-SKU matching needs interpretability.
2. Matching pipeline: summarize Objective 1 as the scoring engine.
3. Robustness analysis: summarize Objective 2 as perturbation-based evaluation.
4. Explainability method: describe Objective 3 as score-level SHAP-style attribution plus rule-based narrative text.
5. Results: show the top-1 explanation table, category distribution, and SHAP summary figure.
6. Discussion: explain which signals dominate and how robust the matcher is under perturbation.

## Reproducibility Notes

- The notebook is designed to run from the saved artifacts in this workspace.
- If Exp 3 or Exp 4 result files are regenerated later, the notebook can load them automatically.
- The explanation outputs depend on the saved score components, so rerunning the notebook after updating Objective 1 artifacts will regenerate the reports.

## Current Status

- The notebook runs successfully end-to-end in the current workspace.
- The SHAP summary plot and top-1 explanation tables are generated without errors.
- The best-match explanations currently use the RFP `item_name` and product `product_name` fields from the source CSVs.
