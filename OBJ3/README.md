# OBJ3 - Explainable AI

This folder contains the explainability layer for the RFP to SKU matching system.

## What it uses
- Objective 1 artifacts from `obj1_model_artifacts/`
- Objective 2 result CSVs from `OBJ2/EXP1`, `OBJ2/EXP2`, `OBJ2/EXP5`, `OBJ2/EXP6`, and `OBJ2/EXP7`
- Notebook-only sources for `OBJ2/EXP3` and `OBJ2/EXP4` when result CSVs are not present

## Main notebook
- `EXPLAINABLE_AI_PIPELINE.ipynb`

## Outputs
The notebook writes explainability outputs to `OBJ3/results/`, including:
- top-1 pair explanations
- Obj2 theme summaries
- category visualizations
- sample report tables

## Notes
- Exp 3 and Exp 4 do not currently have exported result CSVs in the workspace.
- If those experiments are rerun later and outputs are saved, the notebook will pick them up automatically.
