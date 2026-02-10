# ===============================
# model.py
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, log_loss, roc_curve, auc as sklearn_auc
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier



def predictions_summary(df, X_train, X_test, model,
                        col_train="xG_model_train",
                        col_test="xG_model_test"):
    """
    Store predicted probabilities (xG) in df and print summary statistics.
    """
    df.loc[X_train.index, col_train] = model.predict_proba(X_train)[:, 1]
    df.loc[X_test.index, col_test] = model.predict_proba(X_test)[:, 1]

    print("Training set predictions:")
    print(df.loc[X_train.index, col_train].describe())
    print("\nTest set predictions:")
    print(df.loc[X_test.index, col_test].describe())

    return df


def mean_xg_by_outcome(df, X_index, col_pred):
    """
    Compute mean predicted xG for goals and non-goals.
    """
    df_sub = df.loc[X_index]
    mean_goal = df_sub.loc[df_sub["Outcome"] == 1, col_pred].mean()
    mean_nongoal = df_sub.loc[df_sub["Outcome"] == 0, col_pred].mean()
    return mean_goal, mean_nongoal

def calibration_table(y_true, y_pred, n_bins=20):
    """
    Print calibration table based on n_bins quantiles.
    """
    sorted_idx = np.argsort(y_pred)
    y_pred_sorted = y_pred[sorted_idx]
    y_true_sorted = y_true[sorted_idx]
    bins = np.array_split(range(len(y_pred_sorted)), n_bins)

    for i, b in enumerate(bins):
        print(
            f"BIN {i}: mean xG={y_pred_sorted[b].mean():.3f}, "
            f"real goals={y_true_sorted[b].mean():.3f}, "
            f"n_shots={len(b)}"
        )

def plot_calibration(df, X_index, col_pred, n_bins=20, title="Calibration Curve"):
    y_pred = df.loc[X_index, col_pred].values
    y_true = df.loc[X_index, "Outcome"].values

    sorted_idx = np.argsort(y_pred)
    y_pred_sorted = y_pred[sorted_idx]
    y_true_sorted = y_true[sorted_idx]

    bins = np.array_split(range(len(y_pred_sorted)), n_bins)
    prob_pred = [y_pred_sorted[b].mean() for b in bins]
    prob_true = [y_true_sorted[b].mean() for b in bins]

    plt.figure(figsize=(7, 7))
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Mean predicted xG per bin')
    plt.ylabel('Observed goal rate per bin')
    plt.title(title)
    plt.grid(True)
    plt.show()


# ============================================================
# Model vs StatsBomb xG
# ============================================================
def correlation_with_statsbomb(df, X_index, col_model, col_statsbomb="shot_statsbomb_xg"):
    """
    Compute correlation between model xG and StatsBomb xG.
    """
    return df.loc[X_index, [col_model, col_statsbomb]].corr().iloc[0,1]

# ============================================================
# ROC Curve & AUC
# ============================================================
def plot_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc_value = sklearn_auc(fpr, tpr)

    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc_value:.3f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random model')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    return roc_auc_value

def compute_log_loss(y_true, y_score):
    """
    Compute log loss for predicted probabilities.
    """
    return log_loss(y_true, y_score)


def real_vs_expected_goals(df, X_index, col_pred):
    """
    Compute total real goals and expected goals (sum of predicted probabilities).
    """
    df_sub = df.loc[X_index]
    real_goals = df_sub["Outcome"].sum()
    expected_goals = df_sub[col_pred].sum()
    return real_goals, expected_goals


# ============================================================
# Random Forest xG Model
# ============================================================
def train_rf_model(X_train, y_train, n_estimators=500, max_depth=10, random_state=42):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    rf.fit(X_train, y_train)
    return rf

def rf_predictions(df, rf_model, X_train, X_test, col_train="xG_rf_train", col_test="xG_rf_test"):
    df.loc[X_train.index, col_train] = rf_model.predict_proba(X_train)[:, 1]
    df.loc[X_test.index, col_test] = rf_model.predict_proba(X_test)[:, 1]
    return df

def auc_logloss(df, X_index, col_pred):
    y_true = df.loc[X_index, "Outcome"].values
    y_score = df.loc[X_index, col_pred].values
    return roc_auc_score(y_true, y_score), log_loss(y_true, y_score)


def train_xgb_model(X_train, y_train, n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42):
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state
    )
    xgb.fit(X_train, y_train)
    return xgb

def xgb_predictions(df, xgb_model, X_train, X_test, col_train="xG_xgb_train", col_test="xG_xgb_test"):
    df.loc[X_train.index, col_train] = xgb_model.predict_proba(X_train)[:, 1]
    df.loc[X_test.index, col_test] = xgb_model.predict_proba(X_test)[:, 1]
    return df