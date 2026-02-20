# Tabular Classification — FIFA 21 Player Position Prediction

Classifying **~18,500 football players** into four position categories (Forward, Midfielder, Defender, Goalkeeper) using **Random Forest**, **SVM**, and **XGBoost** with GridSearchCV hyperparameter tuning. Built on the [FIFA 21 Complete Player Dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-21-complete-player-dataset) from Kaggle, using ~40 in-game skill attributes as features.

---

## Table of Contents

- [Task Overview](#task-overview)
- [Dataset](#dataset)
- [Target Engineering](#target-engineering)
- [Feature Set](#feature-set)
- [Data Preprocessing](#data-preprocessing)
- [Model Architectures](#model-architectures)
  - [Model 1: Random Forest](#model-1-random-forest)
  - [Model 2: Support Vector Machine (SVM)](#model-2-support-vector-machine-svm)
  - [Model 3: XGBoost](#model-3-xgboost)
- [Optimisation Techniques](#optimisation-techniques)
- [Evaluation Metrics](#evaluation-metrics)
- [Prediction & Inference](#prediction--inference)
- [Getting Started](#getting-started)
- [Notebook Structure](#notebook-structure)
- [Outputs](#outputs)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Task Overview

```
┌───────────────────────┐      ┌──────────────────────┐      ┌──────────────────┐
│   Player Attributes   │─────▶│  RF / SVM / XGBoost  │─────▶│  Position Class  │
│  (~40 skill ratings)  │      │  with GridSearchCV   │      │  (4 categories)  │
└───────────────────────┘      └──────────────────────┘      └──────────────────┘
```

- **Input:** ~40 numerical player attributes (pace, shooting, defending, goalkeeping, etc.)
- **Output:** One of four position categories — Forward, Midfielder, Defender, or Goalkeeper

**Real-world applications:**
- Recommending optimal playing positions based on skill profiles
- Auto-tagging players in scouting databases
- Talent identification and squad-building analytics
- Content categorisation for sports platforms

---

## Dataset

**[FIFA 21 Complete Player Dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-21-complete-player-dataset)** — web-scraped from sofifa.com.

| Property | Value |
|----------|-------|
| Total players | ~18,500 |
| Original columns | 92 |
| Features used | ~40 relevant skill attributes |
| Target | `position_category` (engineered from `player_positions`) |
| Source | sofifa.com (FIFA 21 video game data) |

The dataset is downloaded automatically via the `opendatasets` library using Kaggle API credentials.

---

## Target Engineering

Raw positions from `player_positions` (e.g., "ST, CF", "CB, RB") are mapped to four simplified categories using the primary (first-listed) position:

| Category | Positions Included |
|----------|-------------------|
| **Forward** | ST, CF, LW, RW, LF, RF |
| **Midfielder** | CM, CAM, CDM, LM, RM, LAM, RAM, LCM, RCM, LDM, RDM |
| **Defender** | CB, LB, RB, LWB, RWB, LCB, RCB |
| **Goalkeeper** | GK |

Edge cases (unrecognised positions) default to Midfielder.

---

## Feature Set

Features span eight attribute categories from the FIFA 21 player model:

| Category | Features |
|----------|----------|
| **Basic** | `overall`, `potential`, `age`, `height_cm`, `weight_kg` |
| **Main Skills** | `pace`, `shooting`, `passing`, `dribbling`, `defending`, `physic` |
| **Attacking** | `attacking_crossing`, `attacking_finishing`, `attacking_heading_accuracy`, `attacking_short_passing`, `attacking_volleys` |
| **Skill** | `skill_dribbling`, `skill_curve`, `skill_fk_accuracy`, `skill_long_passing`, `skill_ball_control` |
| **Movement** | `movement_acceleration`, `movement_sprint_speed`, `movement_agility`, `movement_reactions`, `movement_balance` |
| **Power** | `power_shot_power`, `power_jumping`, `power_stamina`, `power_strength`, `power_long_shots` |
| **Mentality** | `mentality_aggression`, `mentality_interceptions`, `mentality_positioning`, `mentality_vision`, `mentality_penalties` |
| **Defending** | `defending_marking_awareness`, `defending_standing_tackle`, `defending_sliding_tackle` |
| **Goalkeeping** | `goalkeeping_diving`, `goalkeeping_handling`, `goalkeeping_kicking`, `goalkeeping_positioning`, `goalkeeping_reflexes` |

Only columns that actually exist in the downloaded dataset are used — missing naming variants are automatically filtered out.

---

## Data Preprocessing

1. **Column filtering** — candidate features are checked against the actual dataset; missing columns are silently dropped
2. **All-NaN column removal** — columns with no valid data (e.g., `defending_marking` if only `defending_marking_awareness` exists) are excluded
3. **Median imputation** — remaining missing values in numerical features are filled with column medians via `SimpleImputer`
4. **Target cleaning** — rows with no valid position category are dropped
5. **Label encoding** — position categories encoded to integers for model training
6. **Stratified train/test split** — 80/20 with `random_state=42`, preserving class proportions
7. **Feature scaling** — `StandardScaler` (zero mean, unit variance) — critical for SVM, beneficial for model comparability

---

## Model Architectures

All three models are trained with **GridSearchCV** (5-fold cross-validation, scoring by accuracy).

### Model 1: Random Forest

| Parameter | Search Space |
|-----------|-------------|
| `n_estimators` | 100, 200 |
| `max_depth` | 10, 20, None |
| `min_samples_split` | 2, 5 |
| `min_samples_leaf` | 1, 2 |

An ensemble of decision trees that reduces overfitting through bagging and random feature subsets.

### Model 2: Support Vector Machine (SVM)

| Parameter | Search Space |
|-----------|-------------|
| `C` | 0.1, 1, 10 |
| `gamma` | scale, auto |
| `kernel` | RBF |

Uses the radial basis function kernel to find non-linear decision boundaries in the scaled feature space. Probability calibration is enabled for AUC-ROC computation.

### Model 3: XGBoost

| Parameter | Search Space |
|-----------|-------------|
| `n_estimators` | 100, 200 |
| `max_depth` | 3, 5, 7 |
| `learning_rate` | 0.01, 0.1, 0.2 |
| `subsample` | 0.8, 1.0 |

A gradient-boosted tree ensemble with stochastic subsampling and regularised learning. Uses `mlogloss` as the evaluation metric for multiclass classification.

---

## Optimisation Techniques

| # | Technique | Configuration | Purpose |
|---|-----------|---------------|---------|
| 1 | **Feature Scaling** | `StandardScaler` (zero mean, unit variance) | Essential for SVM; ensures fair model comparison |
| 2 | **Hyperparameter Tuning** | `GridSearchCV` with 5-fold CV for all three models | Systematic search for optimal configurations |
| 3 | **Random Forest Regularisation** | `max_depth`, `min_samples_split`, `min_samples_leaf` | Controls tree complexity to prevent overfitting |
| 4 | **SVM Kernel Tuning** | `C` (regularisation), `gamma` (kernel width) | Balances margin maximisation with classification error |
| 5 | **XGBoost Boosting Control** | `learning_rate`, `max_depth`, `subsample` | Step-size shrinkage and stochastic gradient boosting |
| 6 | **Missing Value Handling** | Median imputation + all-NaN column removal | Robust preprocessing without data leakage |
| 7 | **Stratified Splitting** | `stratify=y_encoded` in train/test split | Preserves class proportions across splits |

---

## Evaluation Metrics

All three models are evaluated on the same 20% held-out test set with five metrics:

| Metric | Averaging | Description |
|--------|-----------|-------------|
| **Accuracy** | — | Overall fraction of correct predictions |
| **Precision** | Weighted | Proportion of positive predictions that are correct, per class |
| **Recall** | Weighted | Proportion of actual positives correctly identified, per class |
| **F1-Score** | Weighted | Harmonic mean of precision and recall |
| **AUC-ROC** | Weighted, one-vs-rest | Area under the ROC curve for multiclass (binarised) |

### Visualisations

The notebook produces:

- **Class distribution** — bar chart of the four position categories
- **Model comparison** — grouped bar chart of all metrics across three models
- **Confusion matrices** — side-by-side heatmaps for Random Forest, SVM, and XGBoost
- **Feature importance** — top-20 features from XGBoost's feature importances
- **Per-class classification report** — precision, recall, F1 for each position category
- **Individual player predictions** — correctly classified and misclassified player examples with names

---

## Prediction & Inference

The notebook includes a reusable function for predicting any player's optimal position:

```python
predicted_position, probabilities = predict_player_position(
    player_stats={
        'overall': 85, 'potential': 88, 'age': 27,
        'pace': 90, 'shooting': 88, 'passing': 72,
        'dribbling': 85, 'defending': 35, 'physic': 78,
        'attacking_finishing': 92, 'movement_sprint_speed': 91,
        # ... remaining attributes
    },
    model=xgb_best,
    scaler=scaler,
    label_encoder=label_encoder,
    attribute_cols=attribute_cols,
)
# Output: *** PREDICTED POSITION: Forward ***
#   Forward     : ██████████████████████████ 87.3%
#   Midfielder  : ███                        9.1%
#   Defender    : █                          2.8%
#   Goalkeeper  :                            0.8%
```

The notebook demonstrates three player profiles:
1. **Striker profile** — high shooting, pace, finishing → predicts Forward
2. **Centre-back profile** — high defending, tackling, strength → predicts Defender
3. **Goalkeeper profile** — high goalkeeping stats → predicts Goalkeeper

---

## Getting Started

### Requirements

- **Hardware:** CPU is sufficient; GPU optional
- **Python:** 3.8+

### Installation

```bash
pip install scikit-learn xgboost pandas numpy matplotlib seaborn opendatasets
```

### Kaggle API Setup

The notebook downloads the dataset via `opendatasets`. You'll need Kaggle credentials:

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **Create New Token**
2. When prompted by the notebook, enter your Kaggle username and API key

### Running

1. Open `Tabular_Classification_FIXED.ipynb` in Jupyter or Google Colab.
2. Run all cells sequentially — the notebook handles dataset download, preprocessing, training, evaluation, and inference.

---

## Notebook Structure

| Cell(s) | Section | Description |
|---------|---------|-------------|
| 0 | Introduction | Dataset description, feature categories, models overview |
| 1–2 | Setup | Install dependencies, import libraries |
| 3–6 | Data Loading | Download from Kaggle, load CSV, inspect columns and sample rows |
| 7 | Feature Selection | Filter ~40 candidate attributes against actual dataset columns |
| 8 | Target Engineering | Map raw positions to 4 categories (Forward/Midfielder/Defender/Goalkeeper) |
| 9 | Data Cleaning | Remove all-NaN columns, median imputation, drop invalid rows |
| 10 | Class Distribution | Bar chart of position categories |
| 11–12 | Split & Scale | Label encoding, stratified 80/20 split, `StandardScaler` |
| 13 | Evaluation Function | `evaluate_classifier()` — accuracy, precision, recall, F1, AUC |
| 14 | Model 1: Random Forest | GridSearchCV training and evaluation |
| 15 | Model 2: SVM | GridSearchCV training and evaluation |
| 16 | Model 3: XGBoost | GridSearchCV training and evaluation |
| 17–18 | Results Comparison | Summary table, grouped bar chart |
| 19 | Confusion Matrices | Side-by-side heatmaps for all three models |
| 20 | Feature Importance | Top-20 XGBoost feature importances |
| 21 | Classification Report | Per-class precision/recall/F1 for XGBoost |
| 22–23 | Summary | Optimisation techniques recap, final results |
| 24–27 | Interpretation | Individual player predictions — correct and misclassified examples |
| 28–31 | Inference | `predict_player_position()` function + striker, defender, goalkeeper examples |
| 32–33 | Final Summary | Feature importance chart and concluding metrics |

---

## Outputs

| Artifact | Description |
|----------|-------------|
| Model comparison table | Side-by-side metrics for RF, SVM, XGBoost |
| Confusion matrices | 3-panel heatmap for all models |
| Feature importance chart | Top-20 attributes ranked by XGBoost importance |
| Classification report | Per-class precision, recall, F1 for best model |
| Player prediction examples | Named players with predicted vs actual positions |

---

## Tech Stack

| Library | Role |
|---------|------|
| [scikit-learn](https://scikit-learn.org/) | Random Forest, SVM, preprocessing, GridSearchCV, evaluation metrics |
| [XGBoost](https://xgboost.readthedocs.io/) | Gradient-boosted tree classifier |
| [opendatasets](https://github.com/JovianHQ/opendatasets) | Kaggle dataset download |
| [Matplotlib / Seaborn](https://matplotlib.org/) | Class distribution, confusion matrices, feature importance, model comparison |
| [NumPy / Pandas](https://numpy.org/) | Data manipulation and feature engineering |

---

## License

This project is for educational and research purposes.
