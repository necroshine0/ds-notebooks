# Default prediction
Test task from Sber

## Colab

* v1: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19iL1R9lIav6XVzDLOeo480CG0tthB-Dr?usp=sharing)
* v2:
  * EDA: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oTewlFcNLnLGURcg1a_xvfJ8uoYevcxG?usp=sharing)
  * Modeling: TODO

## Data
* (32 395, 35) shape
* Imbalanced target: 6.454% of +

## Models
* v1:
  * Logistic Regression (feature selection, baseline)
  * CatBoost
* v2:
  * TODO: Kernel SVM, KNN, metric learning

## Estimation
 * Cohen's kappa, Matthew's corr coeff
 * ROC-AUC, PR-AUC
 * Accuracy (private leaderboard)

## Processing
* Distribution analysis
* NaNs
* Binning (v2 only)
* Oversampling/Undersampling (v1 only)
* Calibration (v1 only)

## Results
* v1:
  * Leaderboard score: 0.565 (accuracy)
  * Validation:
    * Cohen's kappa: 0.164503
    * ROC-AUC: 0.6956
* v2: no info yet

## Dataset link: see [notebook](https://github.com/necroshine0/ds-notebooks/blob/main/default/sber-default-v1.ipynb)
