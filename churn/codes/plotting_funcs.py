import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import auc, roc_curve

sns.set(style='whitegrid')
plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = 9, 4

from codes.iqrs import iqr_segment


def plot_heatmap(df, method='Spearman', _type='', triangle=True, colormap='inferno', figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.title('{} Correlation of {} Features'.format(method, _type), y=1.05, size=18)
    corrs = df.corr(method=method.lower())
    
    mask = np.triu(np.ones_like(corrs, dtype=bool)) if triangle else None
    
    sns.heatmap(corrs, mask=mask,
                linewidths=0.1, vmax=1.0, fmt='.2f',
                square=True, cmap=colormap, linecolor='white', annot=True)
    plt.show()


def plot_features_dists(df, features: list, log=False, kwargs={'bins': 70, 'edgecolor': 'k'}, div=2):
    '''
        log -- dict of {name: func}, e.g. {'feature_name': np.log1p} or bool
    '''

    if len(features) == 1:
        feature = features[0]
        plt.xlabel('{} value'.format(feature))
        if type(log) == bool:
            plt.hist(np.log1p(df[feature].values), **kwargs) if log else plt.hist(df[feature].values, **kwargs)
            plt.xlabel('log {} value'.format(feature))
        else:
            plt.hist(log(df[feature].values), **kwargs)

        plt.ylabel('number of records')
        plt.title('Log {} Distribution'.format(feature))
        plt.show()
        return

    l = len(features)
    assert l % div == 0

    fsz = plt.rcParams['figure.figsize']
    w, h = 0.85 * fsz[0] * div, 0.8 * fsz[1] * l // div + 1.5 * l
    fontsize = [20, 16] if div == 4 else [15, 12]
    fig, axes = plt.subplots(ncols=div, nrows=l // div, squeeze=False, figsize=(w, h))
    unlogged = {} # dict of unlogged features alike {name: min_value}
    for i, feature in enumerate(features):
        ii, jj = i // div, i % div
        axes[ii][jj].set_xlabel('{} value'.format(feature))

        values = df[feature].dropna().values
        min_values = np.min(values)

        if type(log) == bool:
            if log and min_values >= 0:
                axes[ii][jj].hist(np.log1p(values), **kwargs)
                axes[ii][jj].set_xlabel('log {} value'.format(feature), fontsize=fontsize[1])
            else:
                axes[ii][jj].hist(df[feature].values, **kwargs)
                unlogged[feature] = min_values
        elif type(log) == dict:
            if feature in log and min_values >= 0:
                axes[ii][jj].hist(log[feature](df[feature].values), **kwargs)
                axes[ii][jj].set_xlabel('log {} value'.format(feature), fontsize=fontsize[1])
            else:
                axes[ii][jj].hist(df[feature].values, **kwargs)

        axes[ii][jj].set_title('{} Distribution'.format(feature), fontsize=fontsize[0])
        axes[ii][0].set_ylabel('number of records', fontsize=fontsize[1])
    plt.show()

    if type(log) == bool and log:
        return unlogged if len(unlogged) > 0 else None
    else:
        return None


def plot_boxplots(df, features: list, kwargs: dict = None, div=4, segment_type: str = 'standard') -> dict:
    '''
        plot boxplots for features
        returns IQR-segments for every feature
    '''

    segments = {}

    l = len(features)
    fsz = plt.rcParams['figure.figsize']
    w, h = 0.7 * fsz[0] * div, fsz[1] * l // div + 1.25 * l
    fig, axes = plt.subplots(ncols=div, nrows=l // div, squeeze=False, figsize=(w, h))

    for i, feature in enumerate(features):
        ii, jj = i // div, i % div
        ax = axes[ii][jj]

        values = df[feature].dropna().values
        if kwargs is not None:
            sns.boxplot(data=values, orient='v', **kwargs, ax=ax)
        else:
            sns.boxplot(data=values, orient='v', ax=ax)

        ax.set_title('{} boxplot'.format(feature), fontsize=15)
        ax.set_ylabel('{} value'.format(feature), fontsize=10)

        segments[feature] = iqr_segment(df[feature], _type=segment_type)
        left, right = segments[feature]
        ax.text(0.08, 0.9, 'seg: [{:.2f}; {:.2f}]'.format(left, right), transform=ax.transAxes,
                bbox=dict(boxstyle='square', facecolor='gray', alpha=0.15))

    plt.show()
    return segments


def plot_eda_features(df, features, figsize=(7.5, 4.5), hue=None, norm=False):
    l = len(features)
    if figsize is None:
        figsize = plt.rcParams['figure.figsize']
    w, h = 2 * figsize[0], l * figsize[1]

    fig, axes = plt.subplots(ncols=2, nrows=l, squeeze=False, figsize=(w, h))

    for i, feature in enumerate(features):
        p1 = sns.boxplot(data=df, y=feature, x='churn', hue=hue, ax=axes[i][0])
        sns.kdeplot(data=df, x=feature, hue='churn', linewidth=1.35,
                                    common_norm=norm, fill=True, ax=axes[i][1])
        if i != l - 1:
            p1.set(xticklabels=[], xlabel=None)
    plt.show()
    

def plot_corr_target(df_train, nums: list, figsize=(6, 3)):
    correlations = df_train[nums].corrwith(df_train.churn).sort_values(ascending=False)
    corrs = correlations.index.tolist()
    sns.barplot(y=corrs, x=correlations)
    plt.title('Pearson Correlation of Features', y=1.05, size=18)
    plt.show()

    
def plot_countplots(df, features: list, figsize=None):
    l = len(features)
    assert l % 2 == 0
    
    fsz = plt.rcParams['figure.figsize'] if figsize is None else figsize
    w, h = 2 * fsz[0], l // 2 * fsz[1] + l
    
    fig, axes = plt.subplots(ncols=2, nrows=l // 2, squeeze=False, figsize=(w, h))
    
    for i, feature in enumerate(features):
        ax = axes[i // 2][i % 2]
        sns.countplot(data=df, x=feature, hue='churn', ax=ax)
        ax.set_title('{} Countplot'.format(feature))
    plt.show()
    
    
def draw_ROC(fpr, tpr, title, ax=None):
    if ax is None:
        ax = plt.gca()
        
    ax.plot(fpr, tpr, color='royalblue', label='roc-curve', linewidth=2)
    ax.set_ylabel('TPR')
    ax.set_xlabel('FPR')
    ax.set_title('{} ROC Curve'.format(title))
    return ax


def plot_curves(models, X, y, figsize=(8, 8)):
    l = len(models)
    ncols = l if l <= 3 else round(l / 2)
    nrows = 1 if l <= 3 else l // 2
    figsize = (12, 4) if l <= 3 else figsize
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=figsize)
    
    for i, m in enumerate(models):
        probas = m.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, probas)
        
        ax = axes[i // 2][i % 2] if l > 3 else axes[0][i]
        draw_ROC(fpr, tpr, m.__class__.__name__, ax=ax)
        ax.plot([0,1], [0,1], color='lime', linestyle='dashed', linewidth=2)
        auc_roc = auc(fpr, tpr)
        ax.text(0.08, 0.9, 'AUC: {:.4f}'.format(auc_roc), transform=ax.transAxes,
                bbox=dict(boxstyle='square', facecolor='gray', alpha=0.16))
        
    fig.tight_layout()
    plt.show()
    
    
def plot_calibration_curve(y_test, preds, label='', color='b', ax=None):
    if ax is None:
        ax = plt.gca()
    
    bin_middle_points = []
    bin_real_ratios = []
    n_bins = 10
    for i in range(n_bins):
        l = 1.0 / n_bins * i
        r = 1.0 / n_bins * (i + 1)
        bin_middle_points.append((l + r) / 2) # Здесь был -
        bin_real_ratios.append(np.mean(y_test[(preds >= l) & (preds < r)] == 1))
        
    ax.plot(bin_middle_points, bin_real_ratios, color=color, label=label)
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Segment center')
    ax.set_ylabel('Fracture of positive predictions')
    ax.set_title('Calibration Curve')
    ax.grid()
    if label != '':
        ax.legend(shadow=False, fontsize=14)
    return ax
    