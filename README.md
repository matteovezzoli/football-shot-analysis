# ⚽ Football Shot Analysis & Expected Goals Modeling

This repository contains my **first complete football analytics project**, developed as an **end-to-end data science pipeline in Python** on football shot data.

The analysis integrates **exploratory data analysis (EDA)**, **feature engineering**, and **Expected Goals (xG) modeling** to study how shot geometry, player roles, spatial location, and contextual factors influence goal-scoring probability.

The entire workflow is designed to be **fully reproducible and methodologically transparent**, with all analyses executed programmatically.

---

## 🎯 Project Goals

The main objectives of this project are to:

- Perform **in-depth exploratory data analysis** of football shot events  
- Analyze the relationship between **shot geometry and goal probability**
- Study how **player roles, spatial zones, and contextual variables** affect scoring
- Build and compare multiple **Expected Goals (xG) models**
- Evaluate models using **probabilistic, discriminative, and football-level metrics**
- Compare model outputs with **StatsBomb xG** as an external benchmark

---

## 📊 Data Source

The data used in this project comes from **StatsBomb Open Data**, publicly available for research and educational purposes.

After some initial preprocessing, the dataset contains 68,803 shot-level observations. 

The dataset provides detailed shot-level information, including:
- Shot location and geometry
- Body part and shot technique
- Player position
- Play pattern
- Contextual binary variables (e.g. defensive pressure, one-on-one situations)

---

## 🔍 Exploratory Data Analysis (EDA)

A substantial portion of the project is dedicated to **structured exploratory data analysis**, aimed at understanding the mechanisms underlying goal scoring.

Exploratory analyses are performed using **engineered geometric and contextual variables** (e.g. distance, angles, player roles) in order to study relationships in the data before any modeling decisions are made.

---

### Shot Geometry

The geometric foundations of xG are explored through:

- Distribution of **shot distance**
- Distribution of **shot angle** and **opening angle**
- Empirical goal probability as a function of:
  - Distance (binned)
  - Angle (binned)
  - Opening angle (binned)

These analyses highlight the **strongly non-linear relationship** between shot geometry and scoring probability.

---

### Role-Based Shot Characteristics

Shots are analyzed by **macro player role**, including:
- Central Forwards
- Wingers
- Midfielders
- Fullbacks  
(with Center Backs used as baseline)

For each role, the analysis compares:
- Average shot distance
- Average shot angle
- Average opening angle
- Empirical goal probability

This provides insight into **how different roles generate and convert scoring chances**.

---

### Contextual Effects

The influence of **contextual and tactical variables** is examined, including:
- Defensive pressure (analyzed across distance bins)
- First-time shots
- One-on-one situations
- Aerial duels
- Open-goal situations
- Shots following a dribble

Goal probabilities are computed and compared across these binary conditions.

---

### Spatial & Pitch-Level Analysis

Spatial analyses include:
- Shot distribution in the **final 30 meters**
- Adaptive pitch grids (penalty area vs outside the box)
- Identification of **high-volume and high-conversion zones**
- Joint analysis of **distance and opening angle**

---

### Shots on Target & Goal Frame Analysis

For shots on target:
- Shot-zone classification by pitch location
- Spatial visualization of shot origins
- Goal-mouth discretization (5 × 3 grid)
- Goal heatmaps by shot zone

This allows the study of **shot placement quality**, not only shot location.

---

## ⚙️ Feature Engineering & Selection

After exploratory analysis, features are finalized for modeling purposes.

Key steps include:

- Definition of the binary outcome (Goal vs No Goal)
- Removal of penalty kicks
- Engineering of core geometric variables:
  - **Distance to goal**: Euclidean distance between the shot location and the center of the goal
  - **Shot angle**: angle between the shot location and the goal center, capturing lateral shooting difficulty
  - **Opening angle**: visible angle of the goal mouth from the shooting location, accounting for goal width
- Grouping of:
  - Player positions into macro roles
  - Play patterns into tactical categories
- Encoding of categorical variables
- Correlation-based feature screening
- Removal of:
  - Outcome proxies
  - Post-shot information
  - Redundant categorical variables
- Multicollinearity assessment using **Variance Inflation Factor (VIF)**

The final feature set is constructed to ensure **causal xG estimation** and to avoid information leakage.

---

## 🤖 Expected Goals (xG) Modeling

Using the selected feature set, several probabilistic models are trained and compared:

- **Logistic Regression** (baseline model)
- **Random Forest Classifier**
- **XGBoost Classifier**

Each model estimates the probability that a shot results in a goal, interpreted as **Expected Goals (xG)**.

Models are trained using a stratified train/test split (70% / 30%), with all evaluations reported on the held-out test set.

---

## 📈 Model Evaluation

Models are evaluated from multiple complementary perspectives:

### Discrimination
- ROC Curve
- Area Under the Curve (AUC)

### Probabilistic Accuracy
- Log-loss

### Calibration
- Calibration tables (quantile-based bins)
- Calibration curves comparing predicted vs observed goal rates

### Football-Level Validation
- Mean xG for goals vs non-goals
- Real goals vs total expected goals
- Correlation with **StatsBomb xG**

---

### 📌 Summary of Results (Test Set)

Model performance on the held-out test set is summarized below:

- **Logistic Regression**  
  - AUC: **0.777**  
  - Log-loss: **0.276**  
  - Correlation with StatsBomb xG: **0.81**

- **Random Forest**  
  - AUC: **0.779**  
  - Log-loss: **0.275**  
  - Correlation with StatsBomb xG: **0.83**

- **XGBoost**  
  - AUC: **0.779**  
  - Log-loss: **0.274**  
  - Correlation with StatsBomb xG: **0.83**

Across all models, predicted xG values show:
- Clear separation between goals and non-goals  
- Stable probabilistic calibration  
- Aggregate expected goals closely aligned with observed goals  

Overall, the models exhibit **strong discriminative performance**, **robust calibration**, and **football-consistent behavior**.

This project is intended as a reproducible reference for shot-based football analytics and as a solid foundation for more advanced Expected Goals modeling.


## 🚀 How to Run the Project

### 1. Clone the repository

```` ``` ````
git clone https://github.com/matteovezzoli/football-shot-analysis.git

cd football-shot-analysis
```` ``` ````
### 2. Clone the repository

```` ``` ````
pip install -r requirements.txt
```` ``` ````
### 3. Run the project

```` ``` ````
python main.py
```` ``` ````
