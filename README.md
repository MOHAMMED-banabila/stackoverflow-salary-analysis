# StackOverflow Developer Salary Analysis

A data science project exploring what factors drive developer compensation, using the **StackOverflow Developer Survey 2023** dataset as part of the Udacity Data Scientist Nanodegree (Project 1).

---

## Motivation

Understanding what shapes a developer's salary matters — whether you are negotiating your next offer, planning a career pivot, or simply curious about the industry. Three questions guide this analysis:

1. How much does professional coding experience affect salary?
2. Does country of employment outweigh formal education as a salary predictor?
3. Do developers who use AI coding tools earn more than those who do not?

---

## File Structure

```
stackoverflow-analysis/
├── notebook.ipynb      # Main analysis: EDA, visualisations, modelling
├── helpers.py          # Reusable, tested Python functions
├── requirements.txt    # Python package dependencies
└── data/
    ├── results.csv     # StackOverflow Developer Survey 2023 responses
    └── schema.csv      # Column descriptions from the original survey
```

---

## Installation and Usage

### 1. Clone / download the project

```bash
git clone <repository-url>
cd stackoverflow-analysis
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the notebook

```bash
jupyter notebook notebook.ipynb
```

Run all cells top-to-bottom; the notebook imports helpers from `helpers.py` and reads data from `data/results.csv`.

---

## Summary of Findings

### 1. Experience is the #1 salary driver

Professional coding experience (`YearsCodePro`) is by far the most influential feature in the salary model, with a feature importance score of **0.549** — over 5× larger than the next predictor. Each additional year of professional experience is associated with a meaningful pay increase, making it the clearest lever a developer can pull.

### 2. Country matters more than education

The correlation between a respondent's country and their salary is **0.67**, compared to only **0.06** for education level. Where you work dominates the compensation equation; a Master's degree does not offset a large geographic pay gap. Developers in the United States, Switzerland, and Israel consistently report the highest median salaries.

### 3. AI tool users do not necessarily earn more

Counter-intuitively, developers currently using AI coding tools report a **lower** median salary ($68,740) than those not using them ($75,000). This likely reflects adoption patterns: AI tools are disproportionately used by junior or mid-level developers who are actively learning, whereas more senior (and higher-paid) developers may already have efficient workflows and are less reliant on AI assistance.

---

## Model Performance

A **Gradient Boosting Regressor** was trained on cleaned survey responses with features including professional experience, country, education level, developer type, organisation size, and remote-work status.

| Metric | Value |
|--------|-------|
| Algorithm | Gradient Boosting Regressor |
| R² (test set) | **0.5678** |
| RMSE (test set) | **$33,020** |

The model explains roughly 57% of salary variance, with a typical prediction error of ±$33K — reasonable given the high noise inherent in self-reported global compensation data.

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, cleaning, and aggregation |
| `numpy` | Numerical operations |
| `matplotlib` | Base plotting |
| `seaborn` | Statistical visualisations |
| `scikit-learn` | Machine learning (GradientBoostingRegressor, metrics, preprocessing) |
| `jupyter` | Interactive notebook environment |
| `ipykernel` | Jupyter kernel for the virtual environment |

---

## Acknowledgements

- **StackOverflow** — for making the [Developer Survey 2023](https://survey.stackoverflow.co/2023/) dataset publicly available.
- **Udacity** — Data Scientist Nanodegree curriculum and project guidelines.
