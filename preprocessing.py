# ===============================
# preprocessing.py
# ===============================

import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_excel(path)

def add_outcome(df):
    df = df.copy()
    df["Outcome"] = (df["shot_outcome"] == "Goal").astype(int)
    return df

def convert_bool_columns(df, bool_cols):
    df = df.copy()
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool).astype(int)
    return df

def add_distance_and_angle(df):
    df = df.copy()
    df["distance"] = ((120 - df["x"])**2 + (40 - df["y"])**2)**0.5 
    angle_rad = np.arctan2(np.abs(df["y"] - 40), 120 - df["x"])
    df["angle"] = np.degrees(angle_rad)   
    return df

def add_body_part_dummies(df):
    df = pd.get_dummies(df, columns=["shot_body_part"], drop_first=True)
    body_part_cols = [c for c in df.columns if c.startswith("shot_body_part_")]
    return df, body_part_cols

def remove_penalties(df):
    return df[df["shot_type"] != "Penalty"].copy()

def add_categorical_dummies(df, cat_cols, drop_first=False):
    df = pd.get_dummies(df, columns=cat_cols, drop_first=drop_first)
    return df

def group_play_pattern(pattern):
    if pattern == "Regular Play":
        return "Open Play"
    elif pattern in ["From Free Kick", "From Corner", "From Goal Kick"]:
        return "Set Piece"
    elif pattern == "From Throw In":
        return "Throw In"
    elif pattern == "From Counter":
        return "Counter"
    elif pattern in ["From Keeper", "From Kick Off", "Other"]:
        return "Other"
    else:
        return "Other"


def add_macro_play_pattern(df):
    df = df.copy()
    df["macro_play_pattern"] = df["play_pattern"].apply(group_play_pattern)
    df = pd.get_dummies(df, columns=["macro_play_pattern"], drop_first=True)
    macro_cols = [c for c in df.columns if c.startswith("macro_play_pattern_")]
    return df, macro_cols

def group_position(pos):
    if pos in [
        "Center Forward", "Left Center Forward",
        "Right Center Forward", "Secondary Striker"
    ]:
        return "Central Forward"

    elif pos in [
        "Right Wing", "Left Wing",
        "Left Attacking Midfield", "Right Attacking Midfield",
        "Center Attacking Midfield"
    ]:
        return "Winger"

    elif pos in [
        "Center Midfield", "Left Midfield", "Right Midfield",
        "Center Defensive Midfield", "Left Defensive Midfield",
        "Right Defensive Midfield", "Left Center Midfield",
        "Right Center Midfield"
    ]:
        return "Midfielder"

    elif pos in [
        "Right Back", "Left Back",
        "Right Wing Back", "Left Wing Back"
    ]:
        return "Fullback"

    elif pos in [
        "Right Center Back", "Center Back",
        "Left Center Back", "Goalkeeper"
    ]:
        return "Center Back"

    else:
        return "Other"


def add_macro_position(df):
    df = df.copy()
    df["macro_position"] = df["position"].apply(group_position)
    return df

def add_macro_position_dummies(df):
    df = df.copy()
    df = pd.get_dummies(df, columns=["macro_position"], drop_first=True)
    macro_cols = [col for col in df.columns if col.startswith("macro_position_")]
    return df, macro_cols

def select_shots_on_target(df, min_x=90):
    """
    Returns shots on target in the last `min_x` meters.
    """
    df_on = df[
        (df["x"] >= min_x) &
        (df["shot_outcome"].isin(["Goal", "Saved", "Saved To Post"]))
    ].copy()
    return df_on

def add_opening_angle(df):
    """
    Compute opening angle to goal (degrees).
    Represents the visible goal mouth from the shooting location.
    """
    df = df.copy()

    x_goal = 120
    y_goal_center = 40
    goal_width = 7.32

    y_left = y_goal_center + goal_width / 2
    y_right = y_goal_center - goal_width / 2

    dx = x_goal - df["x"]

    angle_left = np.arctan2(y_left - df["y"], dx)
    angle_right = np.arctan2(y_right - df["y"], dx)

    opening_angle_rad = np.abs(angle_left - angle_right)
    df["opening_angle"] = np.degrees(opening_angle_rad)

    # Handle shots on the goal line (x == 120)
    df.loc[df["x"] == 120, "opening_angle"] = 180

    return df