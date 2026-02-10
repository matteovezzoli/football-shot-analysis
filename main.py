# ============================================================
# main.py
# Expected Goals (xG) Modeling – Full Analysis 
# ============================================================

# ===============================
# Imports
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    roc_curve,
    auc as sklearn_auc
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from matplotlib.patches import Rectangle
from tabulate import tabulate

from preprocessing import (
    load_data,
    add_outcome,
    convert_bool_columns,
    add_distance_and_angle,
    add_body_part_dummies,
    remove_penalties,
    add_categorical_dummies,
    add_macro_play_pattern,
    add_macro_position,
    add_macro_position_dummies,
    add_opening_angle,
    select_shots_on_target
)

from eda import (
    penalty_summary,
    dummy_correlation,
    plot_outcome_distribution,
    plot_correlation_heatmap,
    plot_boxplots,
    compute_vif,
    plot_shot_distance_distribution,
    plot_shot_angle_distribution,
    plot_goal_prob_by_angle,
    plot_goal_prob_by_distance,
    mean_shot_angle_by_role,
    mean_shot_distance_by_role,
    assign_shot_zones,
    plot_shot_zones_field,
    plot_goal_heatmaps_by_shot_zone,
    plot_xg_vs_distance,
    goal_prob_by_role,
    plot_distance_vs_opening_angle,
    goal_prob_by_distance_pressure,
    goal_prob_by_binary_vars,
    plot_goal_prob_binary_vars,
    plot_adaptive_goal_grid,
    top_zones_table
)

from model import (
    calibration_table,
    plot_calibration,
    correlation_with_statsbomb,
    plot_roc_curve,
    compute_log_loss,
    real_vs_expected_goals,
    train_rf_model,
    rf_predictions,
    mean_xg_by_outcome,
    train_xgb_model,
    xgb_predictions
)

# ============================================================
# 1. Load dataset
# ============================================================
df = load_data("data/xg_dataset.xlsx")

# Binary outcome: Goal vs No Goal
df = add_outcome(df)

# ============================================================
# 2. Penalty summary
# ============================================================
pen_summary = penalty_summary(df)
print("Penalty summary:")
print(pen_summary)

# ============================================================
# 3. Convert boolean columns
# ============================================================
bool_cols = [
    "shot_aerial_won", "shot_first_time", "shot_one_on_one",
    "under_pressure", "shot_deflected", "shot_open_goal",
    "shot_redirect", "shot_saved_off_target", "shot_saved_to_post",
    "shot_follows_dribble"
]
df = convert_bool_columns(df, bool_cols)

# ============================================================
# 4. Distance and angle computation
# ============================================================
df = add_distance_and_angle(df)
df = add_opening_angle(df)

# ============================================================
# 5. Body part dummies
# ============================================================
df, body_part_cols = add_body_part_dummies(df)
print("Body part dummy columns:", body_part_cols)

# ============================================================
# 6. Remove penalties
# ============================================================
df = remove_penalties(df)

# ============================================================
# 7. Categorical dummies (shot type & technique)
# ============================================================
cat_cols = ["shot_type", "shot_technique"]
df = add_categorical_dummies(df, cat_cols, drop_first=False)

# Check
for col in cat_cols:
    print(f"\nDummies created for {col}:")
    print([c for c in df.columns if c.startswith(col + "_")])

# ============================================================
# 8. Correlation: shot technique dummies vs Outcome
# ============================================================
dummy_cols = [col for col in df.columns if col.startswith("shot_technique_")]
corr_cols = dummy_cols + ["Outcome"]

corr_outcome_dummy = (
    df[corr_cols]
    .corr(numeric_only=True)["Outcome"]
    .sort_values(ascending=False)
)

print("\nCorrelation of dummy variables with Outcome:")
print(corr_outcome_dummy)

# ============================================================
# 9. Shot type distribution
# ============================================================
shot_type_dummies = [col for col in df.columns if col.startswith("shot_type_")]

print("\n=== Shot Type ===")
for col in shot_type_dummies:
    count = df[col].sum()
    perc = count / len(df) * 100
    print(f"{col}: {count} shots ({perc:.2f}%)")

print("Decision: drop 'shot_type' because 'Open Play' dominates (~94.78%)")

# ============================================================
# 10. Macro Play Pattern
# ============================================================
df, macro_cols = add_macro_play_pattern(df)
print("Macro play pattern dummy columns:", macro_cols)

corr_macro_outcome = dummy_correlation(
    df,
    dummy_prefixes=["macro_play_pattern_"],
    target="Outcome"
)

print("\nCorrelation of macro play patterns with Outcome:")
print(corr_macro_outcome)

# ============================================================
# 11. Shot Technique summary
# ============================================================
shot_technique_dummies = [
    col for col in df.columns if col.startswith("shot_technique_")
]

print("=== Shot Technique ===")
for col in shot_technique_dummies:
    count = df[col].sum()
    perc = count / len(df) * 100
    print(f"{col}: {count} shots ({perc:.2f}%)")

print("Decision: drop shot_technique because ~77% are Normal shots")

# ============================================================
# 12. Macro Position grouping
# ============================================================
df = add_macro_position(df)
df, macro_cols = add_macro_position_dummies(df)

corr_macro_outcome = dummy_correlation(
    df,
    dummy_prefixes=["macro_position_"],
    target="Outcome"
)

print("\nCorrelation of macro positions with Outcome:")
print(corr_macro_outcome)

# ============================================================
# 13. Outcome distribution
# ============================================================
plot_outcome_distribution(df, target="Outcome")

# ============================================================
# 14. Correlation analysis
# ============================================================
corr_outcome = (
    df.corr(numeric_only=True)["Outcome"]
    .sort_values(ascending=False)
)

print("\nCorrelation of variables with Outcome:")
print(corr_outcome)

plot_correlation_heatmap(df, target="Outcome")

# ============================================================
# 15. Feature selection based on correlation
# ============================================================
drop_cols = [
    "shot_body_part_Left Foot", "shot_body_part_Other", "shot_body_part_Right Foot",
    "shot_type_Free Kick", "shot_type_Corner", "shot_type_Open Play",
    "shot_technique_Lob", "shot_technique_Diving Header", "shot_technique_Volley",
    "shot_technique_Backheel", "shot_technique_Half Volley",
    "shot_technique_Overhead Kick", "shot_technique_Normal"
]

df1 = df.drop(columns=drop_cols)

# ============================================================
# 16. Exploratory boxplots
# ============================================================
numeric_cols = ["distance", "minute"]
plot_boxplots(df, numeric_cols=numeric_cols, target="Outcome")

# ============================================================
# 17. Correlation after feature reduction
# ============================================================
corr1 = df1.corr(numeric_only=True)
corr_outcome1 = corr1["Outcome"].sort_values(ascending=False)

print("\nCorrelation with Outcome after feature reduction:")
print(corr_outcome1)

plot_correlation_heatmap(df1, target="Outcome")

# ============================================================
# 18. Final feature matrix and target 
# ============================================================
exclude_cols = [
    "Outcome", "shot_statsbomb_xg",
    "x_end", "y_end", "z_end",
    "shot_saved_off_target", "shot_saved_to_post",
    "shot_deflected", "period",
    "x", "y", "position", "play_pattern", "shot_outcome"
]

# Exclude post-shot information, outcome proxies and redundant raw coordinates
# to avoid data leakage and ensure causal xG estimation

X = df1.drop(columns=exclude_cols)
y = df1["Outcome"]

# Convert boolean features to numeric format for model compatibility
bool_cols = X.select_dtypes(include=["bool"]).columns
X[bool_cols] = X[bool_cols].astype(int)

# Stratified split to preserve goal / no-goal distribution
# across train and test sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
 

# ============================================================
# 19. Multicollinearity check (VIF)  
# ============================================================
vif_data = compute_vif(X_train)
print("\nVIF check:")
print(vif_data)

# ============================================================
# 20. Logistic Regression (Baseline xG Model)
# ============================================================

logit_model = sm.Logit(y_train, X_train).fit()
print(logit_model.summary())

# ============================================================
# 21. Logistic Regression – Predictions (Train & Test)
# ============================================================

df1.loc[X_train.index, "xG_logit_train"] = logit_model.predict(X_train)
df1.loc[X_test.index, "xG_logit_test"] = logit_model.predict(X_test)

# ============================================================
# 22. Logistic Regression – Mean xG by Outcome
# ============================================================

mean_goal_train, mean_nongoal_train = mean_xg_by_outcome(
    df1, X_train.index, "xG_logit_train"
)
mean_goal_test, mean_nongoal_test = mean_xg_by_outcome(
    df1, X_test.index, "xG_logit_test"
)

print("LOGIT TRAIN – Mean xG GOAL:", mean_goal_train)  #0.22832773398298883
print("LOGIT TRAIN – Mean xG NO GOAL:", mean_nongoal_train) #0.08769740477246339
print("LOGIT TEST  – Mean xG GOAL:", mean_goal_test) #0.23135290930472424
print("LOGIT TEST  – Mean xG NO GOAL:", mean_nongoal_test) #0.08774322362282434

# ============================================================
# 23. Logistic Regression – Calibration
# ============================================================

y_true_train = df1.loc[X_train.index, "Outcome"].values
y_pred_train = df1.loc[X_train.index, "xG_logit_train"].values

calibration_table(y_true_train, y_pred_train, n_bins=20)

plot_calibration(
    df1,
    X_train.index,
    "xG_logit_train",
    n_bins=20,
    title="Calibration Curve (Training – Logit)"
)

# ============================================================
# 24. Logistic Regression – Comparison with StatsBomb xG
# ============================================================

print("\nStatsBomb xG summary:")
print(df1["shot_statsbomb_xg"].describe())

corr_train = correlation_with_statsbomb(df1, X_train.index, "xG_logit_train") #0.8210505777940112
corr_test = correlation_with_statsbomb(df1, X_test.index, "xG_logit_test") #0.8160653665408322

print("LOGIT vs StatsBomb (Train):", corr_train)
print("LOGIT vs StatsBomb (Test):", corr_test)

# ============================================================
# 25. Logistic Regression – ROC & Log Loss (Test)
# ============================================================

y_score_test = df1.loc[X_test.index, "xG_logit_test"].values

roc_auc_logit = plot_roc_curve(y_test.values, y_score_test)   #0.7768497818004465
logloss_logit = compute_log_loss(y_test.values, y_score_test) #0.2758317961394967

print("LOGIT Test AUC:", roc_auc_logit)
print("LOGIT Test Log Loss:", logloss_logit)

# ============================================================
# 26. Logistic Regression – Real vs Expected Goals
# ============================================================

real_train, expected_train = real_vs_expected_goals(
    df1, X_train.index, "xG_logit_train"
)
real_test, expected_test = real_vs_expected_goals(
    df1, X_test.index, "xG_logit_test"
)

print("LOGIT TRAIN – Real:", real_train, "Expected:", expected_train) #4836.497336805767
print("LOGIT TEST  – Real:", real_test, "Expected:", expected_test) #2080.0066616512236

# ============================================================
# 27. Random Forest xG Model
# ============================================================

rf_model = train_rf_model(
    X_train,
    y_train,
    n_estimators=500,
    max_depth=10,
    random_state=42
)

df1 = rf_predictions(
    df1,
    rf_model,
    X_train,
    X_test,
    col_train="xG_rf_train",
    col_test="xG_rf_test"
)

# ============================================================
# 28. Random Forest – Mean xG by Outcome
# ============================================================

mean_goal_train, mean_nongoal_train = mean_xg_by_outcome(
    df1, X_train.index, "xG_rf_train"
)
mean_goal_test, mean_nongoal_test = mean_xg_by_outcome(
    df1, X_test.index, "xG_rf_test"
)

print("RF TRAIN – Mean xG GOAL:", mean_goal_train)  #0.2584953749046375
print("RF TRAIN – Mean xG NO GOAL:", mean_nongoal_train) #0.08321142630998328
print("RF TEST  – Mean xG GOAL:", mean_goal_test) #0.22447129916908495
print("RF TEST  – Mean xG NO GOAL:", mean_nongoal_test) #0.08761155147892016

# ============================================================
# 29. Random Forest – Calibration
# ============================================================

plot_calibration(
    df1,
    X_train.index,
    "xG_rf_train",
    n_bins=20,
    title="Calibration Curve (Training – Random Forest)"
)

# ============================================================
# 30. Random Forest – Comparison with StatsBomb xG
# ============================================================

corr_test_rf = correlation_with_statsbomb(df1, X_test.index, "xG_rf_test") #0.8322253263486368
print("RF vs StatsBomb (Test):", corr_test_rf)

# ============================================================
# 31. Random Forest – ROC & Log Loss (Test)
# ============================================================

y_score_test_rf = df1.loc[X_test.index, "xG_rf_test"].values

roc_auc_rf = plot_roc_curve(y_test.values, y_score_test_rf)
logloss_rf = compute_log_loss(y_test.values, y_score_test_rf)

print("RF Test AUC:", roc_auc_rf)  #0.7790027425032754
print("RF Test Log Loss:", logloss_rf) #0.2746741793825855

# ============================================================
# 32. Random Forest – Real vs Expected Goals
# ============================================================

real_train_rf, expected_train_rf = real_vs_expected_goals(
    df1, X_train.index, "xG_rf_train"
)
real_test_rf, expected_test_rf = real_vs_expected_goals(
    df1, X_test.index, "xG_rf_test"
)

print("RF TRAIN – Real:", real_train_rf, "Expected:", expected_train_rf) #4790.2143553704445
print("RF TEST  – Real:", real_test_rf, "Expected:", expected_test_rf) #2063.424368014102

# ============================================================
# 33. XGBoost xG Model
# ============================================================

xgb_model = train_xgb_model(X_train, y_train)

df1 = xgb_predictions(
    df1,
    xgb_model,
    X_train,
    X_test,
    col_train="xG_xgb_train",
    col_test="xG_xgb_test"
)

# ============================================================
# 34. XGBoost – Mean xG by Outcome
# ============================================================

mean_goal_train, mean_nongoal_train = mean_xg_by_outcome(
    df1, X_train.index, "xG_xgb_train"
)
mean_goal_test, mean_nongoal_test = mean_xg_by_outcome(
    df1, X_test.index, "xG_xgb_test"
)

print("XGB TRAIN – Mean xG GOAL:", mean_goal_train)  #0.2490781
print("XGB TRAIN – Mean xG NO GOAL:", mean_nongoal_train) #0.08455846
print("XGB TEST  – Mean xG GOAL:", mean_goal_test) #0.23622712
print("XGB TEST  – Mean xG NO GOAL:", mean_nongoal_test) #0.08695456

# ============================================================
# 35. XGBoost – Calibration
# ============================================================

plot_calibration(
    df1,
    X_train.index,
    "xG_xgb_train",
    n_bins=20,
    title="Calibration Curve (Training – XGBoost)"
)

# ============================================================
# 36. XGBoost – Comparison with StatsBomb xG
# ============================================================

corr_test_xgb = correlation_with_statsbomb(df1, X_test.index, "xG_xgb_test")
print("XGB vs StatsBomb (Test):", corr_test_xgb) #0.8286699092859471

# ============================================================
# 37. XGBoost – ROC & Log Loss (Test)
# ============================================================

y_score_test_xgb = df1.loc[X_test.index, "xG_xgb_test"].values

roc_auc_xgb = plot_roc_curve(y_test.values, y_score_test_xgb)
logloss_xgb = compute_log_loss(y_test.values, y_score_test_xgb)

print("XGB Test AUC:", roc_auc_xgb) #0.7794097481593564
print("XGB Test Log Loss:", logloss_xgb) #0.27469748416072065

# ============================================================
# 38. XGBoost – Real vs Expected Goals
# ============================================================

real_train_xgb, expected_train_xgb = real_vs_expected_goals(
    df1, X_train.index, "xG_xgb_train"
)
real_test_xgb, expected_test_xgb = real_vs_expected_goals(
    df1, X_test.index, "xG_xgb_test"
)

print("XGB TRAIN – Real:", real_train_xgb, "Expected:", expected_train_xgb) #4802.388
print("XGB TEST  – Real:", real_test_xgb, "Expected:", expected_test_xgb)  #2075.6355

# ============================================================
# 39. XGBoost – Feature Importance
# ============================================================

importance = (
    pd.DataFrame({
        "feature": X_train.columns,
        "importance": xgb_model.feature_importances_
    })
    .sort_values(by="importance", ascending=False)
)

print("Top XGBoost features:")
print(importance.head(15))

# ============================================================
# 40. XGBoost – Learned xG Surface (Distance Diagnostic)
# ============================================================

plot_xg_vs_distance(
    df=df1,
    X_test_index=X_test.index,
    xg_col="xG_xgb_test",
    dist_col="distance",
    outcome_col="Outcome"
)



# ============================================================
# 41. Exploratory Data Analysis (EDA) – Shot Geometry
# ============================================================
# Analysis of core geometric variables underlying xG:
# - shot distance
# - shot angle
# - binned goal probability relationships

# Distribution of shot distance
plot_shot_distance_distribution(X, "distance")

# Distribution of shot angle
plot_shot_angle_distribution(X, "angle")

plot_shot_angle_distribution(X, "opening_angle")

# Goal probability vs angle (binned)
plot_goal_prob_by_angle(
    df1,
    angle_col="angle",
    outcome_col="Outcome",
    n_bins=10
)

plot_goal_prob_by_angle(
    df1,
    angle_col="opening_angle",
    outcome_col="Outcome",
    n_bins=10
)

# Goal probability vs distance (binned)
plot_goal_prob_by_distance(
    df1,
    dist_col="distance",
    outcome_col="Outcome",
    n_bins=10
)

# ============================================================
# 42. EDA – Role-based Shot Characteristics
# ============================================================
# Comparison of shot geometry across macro player roles

roles = [
    "macro_position_Central Forward",
    "macro_position_Winger",
    "macro_position_Midfielder",
    "macro_position_Fullback"
]

# Average shot angle by role (baseline: Center Back) 
mean_shot_angle_by_role(
    df1,
    roles,
    angle_col="angle",
    baseline_role="Center Back"
)

mean_shot_angle_by_role(
    df1,
    roles,
    angle_col="opening_angle",
    baseline_role="Center Back"
)

# Average shot distance by role (baseline: Center Back)
mean_shot_distance_by_role(
    df1,
    roles,
    dist_col="distance",
    baseline_role="Center Back"
)

# ============================================================
# 43. EDA – Goal Probability by Role
# ============================================================
# Empirical scoring probability conditioned on player role

goal_rate_by_role_dict = goal_prob_by_role(
    df,
    roles,
    baseline_role="Center Back",
    outcome_col="Outcome"
)

print("Goal probability by role:")
print(goal_rate_by_role_dict)


# ============================================================
# 44. EDA – Shot Geometry and Goal Distribution
# ============================================================
# Shot-level relationship between opening angle and distance.
# Points are colored by observed outcome (Goal / No Goal).

plot_distance_vs_opening_angle(
    df=df1,
    x_col="opening_angle",
    y_col="distance",
    outcome_col="Outcome"
)

# ============================================================
# 45. EDA – Defensive Pressure Effect by Distance
# ============================================================
# Analysis of how defensive pressure modifies scoring probability
# across distance bins

pressure_effect = goal_prob_by_distance_pressure(
    df,
    n_bins=10
)

print("Goal probability under pressure by distance bin:")
print(pressure_effect)

# ============================================================
# 46. EDA – Binary Contextual Variables
# ============================================================
# Goal probability conditioned on binary shot context variables

binary_vars = [
    "shot_aerial_won",
    "shot_first_time",
    "shot_one_on_one",
    "under_pressure",
    "shot_open_goal",
    "shot_redirect",
    "shot_follows_dribble"
]

goal_rate_binary = goal_prob_by_binary_vars(
    df,
    binary_vars,
    outcome_col="Outcome"
)

print("Goal probability for binary variables:")
print(goal_rate_binary)

plot_goal_prob_binary_vars(df, binary_vars)

# ============================================================
# 47. EDA – Pitch-level Spatial Analysis (Last 30m)
# ============================================================

# Adaptive grid (penalty area vs outside box)
pitch_stats = plot_adaptive_goal_grid(
    df,
    x_thresh_box=103.5,
    outcome_col="Outcome"
)

# Most dangerous zones by volume and conversion
top_zones_table(
    pitch_stats,
    min_shots=100
)

# ============================================================
# 48. EDA – Shots on Target & Goal Frame Analysis
# ============================================================
# Spatial distribution of shots on target and goal-mouth targeting

# Select shots on target in the final 30 meters
df_on = select_shots_on_target(df, min_x=90)

print("=== SHOTS ON TARGET CHECK ===")
print(f"Total shots on target (last 30m): {len(df_on)}")

# Assign shot zones
df_on = assign_shot_zones(df_on)

# Visualize shot zones on pitch
plot_shot_zones_field(df_on)

# ------------------------------------------------------------
# Goal frame discretization (5 × 3 = 15 bins)
# ------------------------------------------------------------

GOAL_Y_MIN = 40 - 7.32 / 2
GOAL_Y_MAX = 40 + 7.32 / 2

y_bins_goal = np.linspace(GOAL_Y_MIN, GOAL_Y_MAX, 6)
z_bins_goal = np.linspace(0, 2.44, 4)

# Goal-mouth heatmaps by shot zone
plot_goal_heatmaps_by_shot_zone(
    df_on,
    y_bins_goal=y_bins_goal,
    z_bins_goal=z_bins_goal,
    min_shots=10
)