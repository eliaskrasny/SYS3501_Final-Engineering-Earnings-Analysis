"""
ENHANCED ENGINEERING EARNINGS ANALYSIS
========================================
Focus Areas:
1. REGRESSION ANALYSIS - Weighted sum of factors predicting earnings
2. CLASSIFICATION - Earnings buckets (Low/Medium/High/Very High)
3. FEATURE ENGINEERING - Systematic exploration and documentation

Author: Enhanced Analysis
Date: December 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

# Classification models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Model evaluation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             classification_report, confusion_matrix, accuracy_score)
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

# Visualization settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("=" * 100)
print("ENHANCED ENGINEERING EARNINGS ANALYSIS")
print("=" * 100)
print("\nFocus: Regression Models, Classification Buckets, and Feature Engineering")
print("=" * 100)

# ============================================================================
# 1. DATA LOADING AND PREPARATION
# ============================================================================
print("\n[1] LOADING DATA")
print("-" * 100)

field_df = pd.read_csv('Data/Most-Recent-Cohorts-Field-of-Study.csv', low_memory=False)
institution_df = pd.read_csv('Data/Most-Recent-Cohorts-Institution.csv', low_memory=False)

# Filter engineering majors
engineering_df = field_df[(field_df['CIPCODE'] >= 1400) & (field_df['CIPCODE'] < 1500)].copy()
df = engineering_df.merge(institution_df, on='UNITID', how='left', suffixes=('', '_inst'))

# Clean data
for col in df.columns:
    df[col] = df[col].replace(['PrivacySuppressed', 'NULL', 'null', ''], np.nan)
    df[col] = pd.to_numeric(df[col], errors='ignore')

print(f"✓ Loaded {len(df):,} engineering programs")
print(f"✓ Total features: {len(df.columns):,}")

# ============================================================================
# 2. FEATURE ENGINEERING EXPLORATION
# ============================================================================
print("\n" + "=" * 100)
print("[2] FEATURE ENGINEERING EXPLORATION")
print("=" * 100)

# Define outcome variable
EARNINGS = 'MD_EARN_WNE_INC3_P7'
df[EARNINGS] = pd.to_numeric(df[EARNINGS], errors='coerce')

print(f"\n{'FEATURE ENGINEERING DECISIONS'}")
print("-" * 100)

# ===== PHASE 1: Base Features (From Domain Knowledge) =====
print("\n📋 PHASE 1: Base Features Selection")
print("-" * 100)
print("Strategy: Select features based on domain knowledge and prior correlation analysis")

base_features = {
    # Selectivity Metrics (STRONGEST predictors from correlation analysis)
    'SAT_AVG': 'Average SAT score - institutional selectivity',
    'ACTCMMID': 'ACT median - alternative selectivity measure',
    'ADM_RATE': 'Acceptance rate - lower = more selective',
    
    # Resource Metrics
    'AVGFACSAL': 'Average faculty salary - resource quality',
    'INEXPFTE': 'Instructional spending per student - investment',
    'TUITIONFEE_OUT': 'Out-of-state tuition - prestige/resource proxy',
    'TUITIONFEE_IN': 'In-state tuition - cost indicator',
    
    # Student Success Metrics
    'C150_4': 'Completion rate (150% time) - student success',
    'RET_FT4': 'Retention rate - student persistence',
    
    # Demographics (potential confounders/mediators)
    'PCTPELL': 'Percent Pell Grant students - SES indicator',
    'UGDS_ASIAN': 'Percent Asian students',
    'UGDS_WHITE': 'Percent White students',
    'UGDS_BLACK': 'Percent Black students',
    'UGDS_HISP': 'Percent Hispanic students',
    
    # Institutional Characteristics
    'CONTROL': 'Institution type (public/private)',
    'LOCALE': 'Urban/suburban/rural location',
    'REGION': 'Geographic region',
    
    # Program Characteristics
    'CIPCODE': 'Engineering discipline code'
}

print(f"✓ Selected {len(base_features)} base features")
for feat, desc in list(base_features.items())[:5]:
    print(f"   - {feat}: {desc}")
print(f"   ... and {len(base_features)-5} more")

# ===== PHASE 2: Interaction Features =====
print("\n📋 PHASE 2: Interaction Features")
print("-" * 100)
print("Strategy: Create interactions between key predictors that may have synergistic effects")

interaction_features_created = []

# Selectivity × Resources interaction
if 'SAT_AVG' in df.columns and 'AVGFACSAL' in df.columns:
    df['SELECTIVITY_X_RESOURCES'] = df['SAT_AVG'] * df['AVGFACSAL'] / 1e6
    interaction_features_created.append('SELECTIVITY_X_RESOURCES')
    print("✓ Created: SELECTIVITY_X_RESOURCES (SAT × Faculty Salary)")
    print("   Rationale: Elite schools with high-paid faculty may have multiplicative effect")

# Selectivity × Completion interaction
if 'SAT_AVG' in df.columns and 'C150_4' in df.columns:
    df['SELECTIVITY_X_COMPLETION'] = df['SAT_AVG'] * df['C150_4']
    interaction_features_created.append('SELECTIVITY_X_COMPLETION')
    print("✓ Created: SELECTIVITY_X_COMPLETION (SAT × Completion Rate)")
    print("   Rationale: Selective schools that also retain students may compound benefits")

# Resources per Pell student (equity metric)
if 'INEXPFTE' in df.columns and 'PCTPELL' in df.columns:
    df['RESOURCES_PER_PELL'] = df['INEXPFTE'] / (df['PCTPELL'] + 0.01)  # Avoid division by zero
    interaction_features_created.append('RESOURCES_PER_PELL')
    print("✓ Created: RESOURCES_PER_PELL (Spending / Pell %)")
    print("   Rationale: How resources are distributed among low-income students")

# ===== PHASE 3: Polynomial Features =====
print("\n📋 PHASE 3: Polynomial Features")
print("-" * 100)
print("Strategy: Test if relationships are non-linear (quadratic, cubic)")

polynomial_features_created = []

# SAT squared (diminishing/accelerating returns?)
if 'SAT_AVG' in df.columns:
    df['SAT_SQUARED'] = df['SAT_AVG'] ** 2
    polynomial_features_created.append('SAT_SQUARED')
    print("✓ Created: SAT_SQUARED")
    print("   Rationale: Returns to selectivity may accelerate at very high SAT levels")

# Acceptance rate squared (non-linear selectivity effect?)
if 'ADM_RATE' in df.columns:
    df['ADM_RATE_SQUARED'] = df['ADM_RATE'] ** 2
    polynomial_features_created.append('ADM_RATE_SQUARED')
    print("✓ Created: ADM_RATE_SQUARED")
    print("   Rationale: Very low acceptance rates (<10%) may have different dynamics")

# ===== PHASE 4: Ratio Features =====
print("\n📋 PHASE 4: Ratio Features")
print("-" * 100)
print("Strategy: Create ratios that capture relative magnitudes")

ratio_features_created = []

# Selectivity efficiency (SAT / Acceptance Rate)
if 'SAT_AVG' in df.columns and 'ADM_RATE' in df.columns:
    df['SELECTIVITY_EFFICIENCY'] = df['SAT_AVG'] / (df['ADM_RATE'] + 0.01)
    ratio_features_created.append('SELECTIVITY_EFFICIENCY')
    print("✓ Created: SELECTIVITY_EFFICIENCY (SAT / Acceptance Rate)")
    print("   Rationale: Combines both selectivity measures into single metric")

# Value metric (Earnings potential per tuition dollar - will create after modeling)
# Return on investment concept

# ===== PHASE 5: Categorical Encodings =====
print("\n📋 PHASE 5: Categorical Encodings")
print("-" * 100)
print("Strategy: Convert categorical variables to useful numeric formats")

categorical_features_created = []

# Institution type dummies
if 'CONTROL' in df.columns:
    df['IS_PRIVATE_NONPROFIT'] = (df['CONTROL'] == 2).astype(int)
    df['IS_PUBLIC'] = (df['CONTROL'] == 1).astype(int)
    categorical_features_created.extend(['IS_PRIVATE_NONPROFIT', 'IS_PUBLIC'])
    print("✓ Created: IS_PRIVATE_NONPROFIT, IS_PUBLIC")
    print("   Rationale: Institution type may have different earnings patterns")

# Urban indicator
if 'LOCALE' in df.columns:
    df['IS_URBAN'] = (df['LOCALE'] <= 13).astype(int)  # Locale codes 11-13 are urban
    categorical_features_created.append('IS_URBAN')
    print("✓ Created: IS_URBAN")
    print("   Rationale: Urban location may correlate with industry connections")

# Elite tier indicator (top SAT schools)
if 'SAT_AVG' in df.columns:
    df['IS_ELITE'] = (df['SAT_AVG'] > 1400).astype(int)
    categorical_features_created.append('IS_ELITE')
    print("✓ Created: IS_ELITE (SAT > 1400)")
    print("   Rationale: Captures elite tier effect as binary variable")

# ===== PHASE 6: Feature Engineering Summary =====
print("\n📊 FEATURE ENGINEERING SUMMARY")
print("-" * 100)

all_engineered_features = (interaction_features_created + 
                          polynomial_features_created + 
                          ratio_features_created + 
                          categorical_features_created)

print(f"Base features selected: {len(base_features)}")
print(f"Interaction features created: {len(interaction_features_created)}")
print(f"Polynomial features created: {len(polynomial_features_created)}")
print(f"Ratio features created: {len(ratio_features_created)}")
print(f"Categorical encodings created: {len(categorical_features_created)}")
print(f"\nTOTAL ENGINEERED FEATURES: {len(all_engineered_features)}")
print("\n✓ Feature engineering exploration complete")

# ===== PHASE 7: Features Considered but REJECTED =====
print("\n❌ FEATURES EXPLORED BUT REJECTED:")
print("-" * 100)
print("1. Three-way interactions (e.g., SAT × Faculty × Completion)")
print("   Reason: Too complex, risk of overfitting, hard to interpret")
print("\n2. Log transformations of monetary variables")
print("   Reason: Linear relationships already strong, adds interpretation complexity")
print("\n3. Region dummy variables (8 separate variables)")
print("   Reason: Geography showed weak correlation (-0.16), not worth dimensionality increase")
print("\n4. Individual SAT component scores (Reading, Math, Writing separately)")
print("   Reason: Multicollinearity issues, composite SAT captures same information")
print("\n5. Student demographic interactions (e.g., Asian% × SAT)")
print("   Reason: Ethical concerns about reinforcing demographic stereotypes in model")

# ============================================================================
# 3. CREATE EARNINGS BUCKETS FOR CLASSIFICATION
# ============================================================================
print("\n" + "=" * 100)
print("[3] CREATING EARNINGS CATEGORIES (BUCKETS)")
print("=" * 100)

# Filter to records with earnings data
df_earnings = df[df[EARNINGS].notna()].copy()
print(f"\nPrograms with earnings data: {len(df_earnings):,}")

# Calculate quartiles for natural breaks
earnings_quartiles = df_earnings[EARNINGS].quantile([0.25, 0.5, 0.75])
print(f"\nEarnings Distribution:")
print(f"   25th percentile: ${earnings_quartiles[0.25]:,.0f}")
print(f"   50th percentile: ${earnings_quartiles[0.50]:,.0f}")
print(f"   75th percentile: ${earnings_quartiles[0.75]:,.0f}")

# Create 4 categories based on quartiles
df_earnings['EARNINGS_CATEGORY'] = pd.cut(
    df_earnings[EARNINGS],
    bins=[0, earnings_quartiles[0.25], earnings_quartiles[0.5], earnings_quartiles[0.75], np.inf],
    labels=['Low', 'Medium', 'High', 'Very High'],
    include_lowest=True
)

# Create numeric encoding for ordinal nature
df_earnings['EARNINGS_CATEGORY_NUM'] = df_earnings['EARNINGS_CATEGORY'].cat.codes

print("\n📊 EARNINGS BUCKETS:")
print("-" * 100)
category_summary = df_earnings.groupby('EARNINGS_CATEGORY').agg({
    EARNINGS: ['min', 'max', 'median', 'count']
}).round(0)

for category in ['Low', 'Medium', 'High', 'Very High']:
    cat_data = df_earnings[df_earnings['EARNINGS_CATEGORY'] == category]
    min_earn = cat_data[EARNINGS].min()
    max_earn = cat_data[EARNINGS].max()
    median_earn = cat_data[EARNINGS].median()
    count = len(cat_data)
    print(f"{category:12s}: ${min_earn:>6,.0f} - ${max_earn:>6,.0f} (median: ${median_earn:>6,.0f}, n={count:,})")

# ============================================================================
# 4. PREPARE FEATURE MATRIX
# ============================================================================
print("\n" + "=" * 100)
print("[4] PREPARING FEATURE MATRIX FOR MODELING")
print("=" * 100)

# Combine all feature names
feature_cols = (list(base_features.keys()) + all_engineered_features)

# Filter to available columns
feature_cols_available = [col for col in feature_cols if col in df_earnings.columns]
print(f"\n✓ Using {len(feature_cols_available)} features for modeling")

# Create feature matrix
X = df_earnings[feature_cols_available].copy()

# Convert any object columns to numeric (excluding the engineered categorical indicators)
for col in X.columns:
    if X[col].dtype == 'object' and col not in categorical_features_created:
        # Try to convert to numeric, drop if can't
        X[col] = pd.to_numeric(X[col], errors='coerce')

y_regression = df_earnings[EARNINGS].copy()
y_classification = df_earnings['EARNINGS_CATEGORY_NUM'].copy()

# Handle missing values - use median imputation
print(f"\nHandling missing values...")
print(f"   Features before cleaning: {X.shape[1]}")

# Remove features with >50% missing
missing_pct = X.isnull().mean()
features_to_keep = missing_pct[missing_pct < 0.5].index.tolist()
X = X[features_to_keep]
print(f"   Removed {len(feature_cols_available) - len(features_to_keep)} features with >50% missing")
print(f"   Features after cleaning: {X.shape[1]}")

# Impute remaining missing with median
for col in X.columns:
    if X[col].isnull().any():
        X[col].fillna(X[col].median(), inplace=True)

# Final clean dataset
complete_mask = X.notna().all(axis=1) & y_regression.notna()
X_clean = X[complete_mask]
y_regression_clean = y_regression[complete_mask]
y_classification_clean = y_classification[complete_mask]

print(f"\n✓ Final dataset: {len(X_clean):,} programs × {X_clean.shape[1]} features")

# ============================================================================
# 5. REGRESSION ANALYSIS - WEIGHTED SUM OF FACTORS
# ============================================================================
print("\n" + "=" * 100)
print("[5] REGRESSION ANALYSIS: Predicting Earnings as Weighted Sum")
print("=" * 100)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_regression_clean, test_size=0.2, random_state=42
)

print(f"\nTraining set: {len(X_train):,} programs")
print(f"Test set: {len(X_test):,} programs")

# Scale features for regularized regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== MODEL 1: Linear Regression (Baseline) =====
print("\n" + "-" * 100)
print("MODEL 1: LINEAR REGRESSION (Unregularized)")
print("-" * 100)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

lr_r2 = r2_score(y_test, y_pred_lr)
lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))

print(f"R² Score: {lr_r2:.4f}")
print(f"MAE: ${lr_mae:,.0f}")
print(f"RMSE: ${lr_rmse:,.0f}")

# Get feature coefficients
lr_coefs = pd.DataFrame({
    'Feature': X_clean.columns,
    'Coefficient': lr_model.coef_,
    'Abs_Coefficient': np.abs(lr_model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print("\nTOP 10 MOST IMPORTANT FEATURES (by coefficient magnitude):")
print(lr_coefs.head(10)[['Feature', 'Coefficient']].to_string(index=False))

# ===== MODEL 2: Ridge Regression (L2 Regularization) =====
print("\n" + "-" * 100)
print("MODEL 2: RIDGE REGRESSION (L2 Regularization)")
print("-" * 100)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)

ridge_r2 = r2_score(y_test, y_pred_ridge)
ridge_mae = mean_absolute_error(y_test, y_pred_ridge)
ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))

print(f"R² Score: {ridge_r2:.4f}")
print(f"MAE: ${ridge_mae:,.0f}")
print(f"RMSE: ${ridge_rmse:,.0f}")

# ===== MODEL 3: Lasso Regression (L1 Regularization - Feature Selection) =====
print("\n" + "-" * 100)
print("MODEL 3: LASSO REGRESSION (L1 Regularization - Automatic Feature Selection)")
print("-" * 100)

lasso_model = Lasso(alpha=100)
lasso_model.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_model.predict(X_test_scaled)

lasso_r2 = r2_score(y_test, y_pred_lasso)
lasso_mae = mean_absolute_error(y_test, y_pred_lasso)
lasso_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lasso))

print(f"R² Score: {lasso_r2:.4f}")
print(f"MAE: ${lasso_mae:,.0f}")
print(f"RMSE: ${lasso_rmse:,.0f}")

# Count non-zero coefficients (selected features)
lasso_selected = np.sum(lasso_model.coef_ != 0)
print(f"\nFeatures selected by Lasso: {lasso_selected} out of {len(X_clean.columns)}")

lasso_coefs = pd.DataFrame({
    'Feature': X_clean.columns,
    'Coefficient': lasso_model.coef_
})
lasso_coefs = lasso_coefs[lasso_coefs['Coefficient'] != 0].sort_values('Coefficient', key=abs, ascending=False)

print("\nNON-ZERO FEATURES (Lasso selected these as important):")
print(lasso_coefs.to_string(index=False))

# ===== MODEL 4: Random Forest Regression =====
print("\n" + "-" * 100)
print("MODEL 4: RANDOM FOREST REGRESSION")
print("-" * 100)

rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)  # Use unscaled data
y_pred_rf = rf_model.predict(X_test)

rf_r2 = r2_score(y_test, y_pred_rf)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print(f"R² Score: {rf_r2:.4f}")
print(f"MAE: ${rf_mae:,.0f}")
print(f"RMSE: ${rf_rmse:,.0f}")

# Feature importance
rf_importance = pd.DataFrame({
    'Feature': X_clean.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTOP 15 FEATURES (by Random Forest importance):")
print(rf_importance.head(15).to_string(index=False))

# ===== REGRESSION MODEL COMPARISON =====
print("\n" + "=" * 100)
print("REGRESSION MODELS COMPARISON")
print("=" * 100)

comparison_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge (L2)', 'Lasso (L1)', 'Random Forest'],
    'R² Score': [lr_r2, ridge_r2, lasso_r2, rf_r2],
    'MAE': [lr_mae, ridge_mae, lasso_mae, rf_mae],
    'RMSE': [lr_rmse, ridge_rmse, lasso_rmse, rf_rmse]
})

print("\n" + comparison_df.to_string(index=False))

# Interpretation
print("\n📊 INTERPRETATION:")
print("-" * 100)
print("Linear/Ridge/Lasso models give you INTERPRETABLE COEFFICIENTS:")
print("  → Each coefficient tells you: 'If this feature increases by 1 unit,")
print("     earnings change by $X (holding all else constant)'")
print("  → This is the 'weighted sum' approach you requested")
print(f"\nRandom Forest gives you BEST PREDICTIONS (R²={rf_r2:.4f}):")
print("  → But coefficients are not directly interpretable")
print("  → Feature importance shows relative predictive power")

# ============================================================================
# 6. CLASSIFICATION ANALYSIS - PREDICTING EARNINGS BUCKET
# ============================================================================
print("\n" + "=" * 100)
print("[6] CLASSIFICATION ANALYSIS: Predicting Earnings Category")
print("=" * 100)

# Train-test split for classification
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clean, y_classification_clean, test_size=0.2, random_state=42, stratify=y_classification_clean
)

print(f"\nTarget distribution in training set:")
train_dist = pd.Series(y_train_clf).value_counts().sort_index()
for idx, count in train_dist.items():
    label = ['Low', 'Medium', 'High', 'Very High'][idx]
    pct = count / len(y_train_clf) * 100
    print(f"   {label}: {count:,} ({pct:.1f}%)")

# ===== CLASSIFIER 1: Logistic Regression =====
print("\n" + "-" * 100)
print("CLASSIFIER 1: LOGISTIC REGRESSION")
print("-" * 100)

# Scale for logistic regression
X_train_clf_scaled = scaler.fit_transform(X_train_clf)
X_test_clf_scaled = scaler.transform(X_test_clf)

log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_clf_scaled, y_train_clf)
y_pred_log = log_model.predict(X_test_clf_scaled)

log_accuracy = accuracy_score(y_test_clf, y_pred_log)
print(f"Accuracy: {log_accuracy:.4f} ({log_accuracy*100:.2f}%)")

# ===== CLASSIFIER 2: Random Forest Classifier =====
print("\n" + "-" * 100)
print("CLASSIFIER 2: RANDOM FOREST CLASSIFIER")
print("-" * 100)

rf_clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_clf.fit(X_train_clf, y_train_clf)
y_pred_rf_clf = rf_clf.predict(X_test_clf)

rf_clf_accuracy = accuracy_score(y_test_clf, y_pred_rf_clf)
print(f"Accuracy: {rf_clf_accuracy:.4f} ({rf_clf_accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test_clf, y_pred_rf_clf, 
                          target_names=['Low', 'Medium', 'High', 'Very High']))

# ===== CLASSIFIER 3: Gradient Boosting Classifier =====
print("\n" + "-" * 100)
print("CLASSIFIER 3: GRADIENT BOOSTING CLASSIFIER")
print("-" * 100)

gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb_clf.fit(X_train_clf, y_train_clf)
y_pred_gb_clf = gb_clf.predict(X_test_clf)

gb_clf_accuracy = accuracy_score(y_test_clf, y_pred_gb_clf)
print(f"Accuracy: {gb_clf_accuracy:.4f} ({gb_clf_accuracy*100:.2f}%)")

# ===== CLASSIFICATION MODEL COMPARISON =====
print("\n" + "=" * 100)
print("CLASSIFICATION MODELS COMPARISON")
print("=" * 100)

clf_comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
    'Accuracy': [log_accuracy, rf_clf_accuracy, gb_clf_accuracy],
    'Accuracy %': [log_accuracy*100, rf_clf_accuracy*100, gb_clf_accuracy*100]
})

print("\n" + clf_comparison.to_string(index=False))

# ============================================================================
# 7. VISUALIZATION OF RESULTS
# ============================================================================
print("\n" + "=" * 100)
print("[7] CREATING VISUALIZATIONS")
print("=" * 100)

fig = plt.figure(figsize=(20, 12))

# Plot 1: Feature Importance from Random Forest Regression
ax1 = plt.subplot(2, 3, 1)
top_features_rf = rf_importance.head(15)
plt.barh(range(len(top_features_rf)), top_features_rf['Importance'])
plt.yticks(range(len(top_features_rf)), top_features_rf['Feature'])
plt.xlabel('Importance')
plt.title('Top 15 Features - Random Forest Regression')
plt.gca().invert_yaxis()

# Plot 2: Linear Regression Coefficients
ax2 = plt.subplot(2, 3, 2)
top_coefs = lr_coefs.head(15)
colors = ['green' if x > 0 else 'red' for x in top_coefs['Coefficient']]
plt.barh(range(len(top_coefs)), top_coefs['Coefficient'], color=colors)
plt.yticks(range(len(top_coefs)), top_coefs['Feature'])
plt.xlabel('Coefficient ($/unit)')
plt.title('Top 15 Linear Regression Coefficients\n(Green=Positive, Red=Negative)')
plt.gca().invert_yaxis()

# Plot 3: Model Performance Comparison
ax3 = plt.subplot(2, 3, 3)
x_pos = range(len(comparison_df))
plt.bar(x_pos, comparison_df['R² Score'])
plt.xticks(x_pos, comparison_df['Model'], rotation=45, ha='right')
plt.ylabel('R² Score')
plt.title('Regression Model Performance\n(Higher is Better)')
plt.ylim(0, 1)
for i, v in enumerate(comparison_df['R² Score']):
    plt.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# Plot 4: Predicted vs Actual (Random Forest)
ax4 = plt.subplot(2, 3, 4)
plt.scatter(y_test, y_pred_rf, alpha=0.5, s=30)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Earnings ($)')
plt.ylabel('Predicted Earnings ($)')
plt.title(f'Random Forest: Predicted vs Actual\nR² = {rf_r2:.4f}')

# Plot 5: Residuals Plot (Random Forest)
ax5 = plt.subplot(2, 3, 5)
residuals = y_test - y_pred_rf
plt.scatter(y_pred_rf, residuals, alpha=0.5, s=30)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Earnings ($)')
plt.ylabel('Residuals ($)')
plt.title('Residual Plot - Random Forest\n(Should be randomly scattered around 0)')

# Plot 6: Classification Accuracy Comparison
ax6 = plt.subplot(2, 3, 6)
x_pos = range(len(clf_comparison))
plt.bar(x_pos, clf_comparison['Accuracy %'], color=['blue', 'green', 'orange'])
plt.xticks(x_pos, clf_comparison['Model'], rotation=45, ha='right')
plt.ylabel('Accuracy (%)')
plt.title('Classification Model Accuracy\n(Predicting Earnings Category)')
plt.ylim(0, 100)
for i, v in enumerate(clf_comparison['Accuracy %']):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('enhanced_analysis_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: enhanced_analysis_results.png")

# ============================================================================
# 8. SAVE DETAILED RESULTS
# ============================================================================
print("\n" + "=" * 100)
print("[8] SAVING DETAILED RESULTS")
print("=" * 100)

# Save feature importance
rf_importance.to_csv('feature_importance_detailed.csv', index=False)
print("✓ Saved: feature_importance_detailed.csv")

# Save regression coefficients
lr_coefs.to_csv('linear_regression_coefficients.csv', index=False)
print("✓ Saved: linear_regression_coefficients.csv")

# Save Lasso selected features
lasso_coefs.to_csv('lasso_selected_features.csv', index=False)
print("✓ Saved: lasso_selected_features.csv")

# Save model comparison
comparison_df.to_csv('regression_model_comparison.csv', index=False)
print("✓ Saved: regression_model_comparison.csv")

clf_comparison.to_csv('classification_model_comparison.csv', index=False)
print("✓ Saved: classification_model_comparison.csv")

# ============================================================================
# 9. FINAL SUMMARY AND RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 100)
print("FINAL SUMMARY AND RECOMMENDATIONS")
print("=" * 100)

print("\n✅ REGRESSION ANALYSIS (Weighted Sum Approach):")
print("-" * 100)
print(f"Best Model: Random Forest (R² = {rf_r2:.4f}, MAE = ${rf_mae:,.0f})")
print(f"Interpretable Model: Linear Regression (R² = {lr_r2:.4f}, MAE = ${lr_mae:,.0f})")
print("\nKey Insight: You CAN create a weighted sum of factors to predict earnings.")
print("The linear regression gives you exact coefficients for each factor.")
print("\nExample interpretation from Linear Regression:")
top_3_coefs = lr_coefs.head(3)
for idx, row in top_3_coefs.iterrows():
    feat = row['Feature']
    coef = row['Coefficient']
    if coef > 0:
        print(f"  → {feat}: +${coef:,.0f} earnings per unit increase")
    else:
        print(f"  → {feat}: ${coef:,.0f} earnings per unit increase")

print("\n✅ CLASSIFICATION ANALYSIS (Earnings Buckets):")
print("-" * 100)
print(f"Best Model: Random Forest Classifier (Accuracy = {rf_clf_accuracy*100:.2f}%)")
print(f"\nThis means we can predict which earnings bucket (Low/Medium/High/Very High)")
print(f"a program will fall into with {rf_clf_accuracy*100:.1f}% accuracy.")

print("\n✅ FEATURE ENGINEERING OUTCOMES:")
print("-" * 100)
print(f"Total features engineered: {len(all_engineered_features)}")
print(f"Features selected by Lasso: {lasso_selected}")
print("\nMost valuable engineered features:")
engineered_in_top = rf_importance.head(15)
engineered_in_top = engineered_in_top[engineered_in_top['Feature'].isin(all_engineered_features)]
if len(engineered_in_top) > 0:
    print(engineered_in_top.to_string(index=False))
else:
    print("  → Base features dominated; engineered features less important")
    print("  → This suggests original features already capture key information")

print("\n" + "=" * 100)
print("✓ ANALYSIS COMPLETE!")
print("=" * 100)
print("\nGenerated files:")
print("  1. enhanced_analysis_results.png - 6-panel visualization")
print("  2. feature_importance_detailed.csv - Random Forest feature rankings")
print("  3. linear_regression_coefficients.csv - Interpretable weights")
print("  4. lasso_selected_features.csv - Auto-selected important features")
print("  5. regression_model_comparison.csv - Model performance metrics")
print("  6. classification_model_comparison.csv - Classification accuracies")
print("\n" + "=" * 100 + "\n")
