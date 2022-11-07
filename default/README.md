# Default prediction
Test task from Sber

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19iL1R9lIav6XVzDLOeo480CG0tthB-Dr?usp=sharing)

## Data
* (32 395, 35) shape
* Imbalanced target: 6.454% of +

## Models
 * Logistic Regression (feature selection, baseline)
 * CatBoost

## Estimation
 * Cohen's kappa
 * ROC-AUC
 * Accuracy (private leaderboard)

## Processing
* Distribution analysis
* NaNs
* Oversampling/Undersampling
* Calibration

## Results
* Leaderboard score: 0.565 (accuracy)
* Validation:
  * Cohen's kappa: 0.164503
  * ROC-AUC: 0.6956

## Dataset link: see [notebook](https://github.com/necroshine0/ds-notebooks/blob/main/default/sber_default.ipynb)
