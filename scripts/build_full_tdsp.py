import json
import copy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_NOTEBOOK = PROJECT_ROOT / "archive" / "notebooks" / "automl_notebook_2017_improved.ipynb"
OUTPUT_NOTEBOOK = PROJECT_ROOT / "notebooks" / "credit_score_automl_tdsp.ipynb"

if not SOURCE_NOTEBOOK.exists():
    # Fallback to current notebook if archive source is unavailable.
    SOURCE_NOTEBOOK = OUTPUT_NOTEBOOK

MOJIBAKE_REPLACEMENTS = {
    "âœ…": "[OK]",
    "âš ï¸": "[WARN]",
    "âŒ": "[ERROR]",
    "ðŸ“Š": "Dataset",
    "ðŸ¤–": "Model",
    "ðŸ“ˆ": "Performance",
    "ðŸ’¾": "Outputs",
}

def create_md_cell(source_lines, cell_id):
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source_lines
    }

def create_code_cell(source_lines, cell_id):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    }

def sanitize_cell(cell):
    """Normalize copied cells for readability and portability."""
    cleaned = copy.deepcopy(cell)

    if cleaned.get("cell_type") == "code":
        cleaned["execution_count"] = None
        cleaned["outputs"] = []

    source = cleaned.get("source", [])
    normalized = []
    for line in source:
        new_line = line
        for bad, good in MOJIBAKE_REPLACEMENTS.items():
            new_line = new_line.replace(bad, good)
        normalized.append(new_line)
    cleaned["source"] = normalized

    return cleaned

# Read the original notebook
with open(SOURCE_NOTEBOOK, 'r', encoding='utf-8') as f:
    original = json.load(f)

# Create new TDSP notebook
tdsp_notebook = {
    "cells": [],
    "metadata": original["metadata"],
    "nbformat": 4,
    "nbformat_minor": 5
}

# TDSP Header
tdsp_notebook["cells"].append(create_md_cell([
    "# Credit Score Classification - AutoML Pipeline\n",
    "## Microsoft Team Data Science Process (TDSP) Template\n",
    "\n",
    "**Project:** Credit Score Classification using AutoML  \n",
    "**Date:** 2026-05-20  \n",
    "**Version:** 2.0 (TDSP Format)\n",
    "\n",
    "---\n",
    "\n",
    "## Table of Contents\n",
    "1. [Business Understanding](#1-business-understanding)\n",
    "2. [Data Acquisition](#2-data-acquisition)\n",
    "3. [Data Understanding](#3-data-understanding)\n",
    "4. [Data Preparation](#4-data-preparation)\n",
    "5. [Modeling](#5-modeling)\n",
    "6. [Evaluation](#6-evaluation)\n",
    "7. [Deployment](#7-deployment)\n",
    "\n",
    "---"
], "tdsp-header"))

# 1. Business Understanding
tdsp_notebook["cells"].append(create_md_cell([
    "## 1. Business Understanding\n",
    "\n",
    "### 1.1 Project Objectives\n",
    "The goal of this project is to develop an automated machine learning solution for credit score classification. This system will:\n",
    "- Classify customers into credit score categories (Poor, Standard, Good)\n",
    "- Provide probability scores for risk assessment\n",
    "- Enable data-driven lending decisions\n",
    "- Reduce manual review time and costs\n",
    "\n",
    "### 1.2 Success Criteria\n",
    "- **Accuracy:** Achieve >70% classification accuracy\n",
    "- **Precision:** Minimize false positives for high-risk categories\n",
    "- **Recall:** Identify majority of poor credit scores\n",
    "- **F1-Score:** Balanced performance across all classes\n",
    "- **Deployment:** Model registered and ready for production use\n",
    "\n",
    "### 1.3 Business Impact\n",
    "- Improved risk assessment accuracy\n",
    "- Faster loan approval process\n",
    "- Reduced default rates\n",
    "- Better customer segmentation\n",
    "\n",
    "### 1.4 Key Stakeholders\n",
    "- **Business:** Credit Risk Management Team\n",
    "- **Technical:** Data Science & ML Engineering Teams\n",
    "- **End Users:** Loan Officers & Credit Analysts"
], "business-understanding"))

# 2. Data Acquisition
tdsp_notebook["cells"].append(create_md_cell([
    "## 2. Data Acquisition\n",
    "\n",
    "### 2.1 Data Sources\n",
    "- **Source:** Delta Lake (train_silver/)\n",
    "- **Format:** Parquet files\n",
    "- **Size:** 100,000 samples\n",
    "- **Features:** 28 columns including demographic, financial, and credit history data"
], "data-acquisition"))

# Add configuration cell from original
for cell in original["cells"]:
    if cell.get("id") == "config-cell":
        tdsp_notebook["cells"].append(sanitize_cell(cell))
        break

# Add logging setup
for cell in original["cells"]:
    if cell.get("id") == "setup-logging":
        tdsp_notebook["cells"].append(sanitize_cell(cell))
        break

# Add package installation
for cell in original["cells"]:
    if cell.get("id") == "install-packages":
        tdsp_notebook["cells"].append(sanitize_cell(cell))
        break

# Add data loading
for cell in original["cells"]:
    if cell.get("id") == "load-data":
        tdsp_notebook["cells"].append(sanitize_cell(cell))
        break

# 3. Data Understanding
tdsp_notebook["cells"].append(create_md_cell([
    "## 3. Data Understanding\n",
    "\n",
    "### 3.1 Data Quality Assessment\n",
    "Comprehensive validation of data integrity, completeness, and distribution."
], "data-understanding"))

# Add validation cells
for cell in original["cells"]:
    if cell.get("id") in ["data-validation", "class-balance"]:
        tdsp_notebook["cells"].append(sanitize_cell(cell))

# 4. Data Preparation
tdsp_notebook["cells"].append(create_md_cell([
    "## 4. Data Preparation\n",
    "\n",
    "### 4.1 Feature Engineering\n",
    "Creating derived features and preparing data for modeling."
], "data-preparation"))

# Add feature engineering, split, and preprocessing cells
for cell in original["cells"]:
    if cell.get("id") in ["feature-engineering", "train-test-split", "preprocessing"]:
        tdsp_notebook["cells"].append(sanitize_cell(cell))

# 5. Modeling
tdsp_notebook["cells"].append(create_md_cell([
    "## 5. Modeling\n",
    "\n",
    "### 5.1 Model Selection & Training\n",
    "Using FLAML AutoML for automated model selection and hyperparameter tuning."
], "modeling"))

# Add MLflow and training cells
for cell in original["cells"]:
    if cell.get("id") in ["mlflow-setup", "automl-training"]:
        tdsp_notebook["cells"].append(sanitize_cell(cell))

# 6. Evaluation
tdsp_notebook["cells"].append(create_md_cell([
    "## 6. Evaluation\n",
    "\n",
    "### 6.1 Model Performance Assessment\n",
    "Comprehensive evaluation using multiple metrics and visualizations."
], "evaluation"))

# Add evaluation cells
for cell in original["cells"]:
    if cell.get("id") in ["model-evaluation", "feature-importance"]:
        tdsp_notebook["cells"].append(sanitize_cell(cell))

# 7. Deployment
tdsp_notebook["cells"].append(create_md_cell([
    "## 7. Deployment\n",
    "\n",
    "### 7.1 Model Registration & Deployment\n",
    "Registering the model to MLflow Model Registry for production deployment."
], "deployment"))

# Add deployment cells
for cell in original["cells"]:
    if cell.get("id") in ["save-model", "generate-predictions", "save-predictions", "pipeline-summary"]:
        tdsp_notebook["cells"].append(sanitize_cell(cell))

# Add conclusion
tdsp_notebook["cells"].append(create_md_cell([
    "## Conclusion\n",
    "\n",
    "### Project Summary\n",
    "This TDSP-structured notebook demonstrates a complete machine learning pipeline for credit score classification:\n",
    "\n",
    "**Key Achievements:**\n",
    "- Structured approach following Microsoft TDSP methodology\n",
    "- Automated model selection using FLAML\n",
    "- Comprehensive evaluation and validation\n",
    "- Production-ready model deployment\n",
    "\n",
    "**Next Steps:**\n",
    "1. Monitor model performance in production\n",
    "2. Implement A/B testing framework\n",
    "3. Schedule periodic model retraining\n",
    "4. Gather feedback from stakeholders\n",
    "\n",
    "---\n",
    "\n",
    "**Documentation:**\n",
    "- [Microsoft TDSP](https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/)\n",
    "- [FLAML Documentation](https://microsoft.github.io/FLAML/)\n",
    "- [MLflow Documentation](https://mlflow.org/)"
], "conclusion"))

# Save the TDSP notebook
with open(OUTPUT_NOTEBOOK, 'w', encoding='utf-8') as f:
    json.dump(tdsp_notebook, f, indent=2, ensure_ascii=False)

print("TDSP notebook created successfully!")
print(f"Total cells: {len(tdsp_notebook['cells'])}")
print(f"File: {OUTPUT_NOTEBOOK.relative_to(PROJECT_ROOT)}")
print("Structure: Business Understanding -> Data Acquisition -> Data Understanding")
print("           -> Data Preparation -> Modeling -> Evaluation -> Deployment")

# Made with Bob
