import numpy as np
import pandas as pd
from scipy.stats import iqr
from statsmodels.stats.stattools import _medcouple_1d


def medcouple(feature_vals):
    vals = np.asarray(feature_vals)
    if vals.shape[0] > 10000:
        vals = np.random.choice(vals, 10000, replace=False)
    return _medcouple_1d(vals)


def iqr_segment(x: pd.Series, _type='standard', coeff=1.5):
    '''
        computes iqr segment
        _type: standard [Q_1 - 1.5 * IQR; Q_3 + 1.5 * IQR]
               adjbox acc. to doi.org/10.1016/j.csda.2007.11.008 (1)
    '''
    IQR = iqr(x, nan_policy='omit')
    Q1, Q3 = x.quantile(0.25), x.quantile(0.75)
    
    if _type == 'standard':
        return [Q1 - coeff * IQR, Q3 + coeff * IQR]
    
    if _type != 'adjbox':
        raise ValueError('Unknown type of segment given')
    
    mc = medcouple(x)
    
    if mc >= 0.0: # right-skewed
        return [Q1 - coeff * np.exp(-3.5 * mc) * IQR, Q3 + coeff * np.exp(4.0 * mc) * IQR]
    else: # left-skewed
        return [Q1 - coeff * np.exp(4 * mc) * IQR, Q3 + coeff * np.exp(-3.5 * mc) * IQR]
