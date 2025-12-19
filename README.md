# Predicting Engineering Graduate Earnings Using Institutional and Program-Level Data

**Authors:** Elias Krasny, Jiei Ota, William Wright, Elias Zell  
**Course:** SYS 3501  
**Date:** December 19, 2025  
**Institution:** University of Virginia

---

## 📋 Executive Summary

This project develops a comprehensive data science pipeline to predict post-graduation earnings for engineering programs in the United States. Using large-scale data from the U.S. Department of Education College Scorecard, we analyze 8,880 engineering programs to understand what drives earnings differences and whether outcomes can be predicted accurately.

**Key Results:**
- **Regression Models:** Random Forest achieves R² = 0.9835 (98.35% variance explained), MAE = $571
- **Classification Models:** Gradient Boosting achieves 97.9% accuracy in categorizing programs into earnings tiers
- **Top Predictor:** Out-of-state tuition accounts for ~60% of feature importance
- **Interpretable Models:** Linear regression achieves R² = 0.745 with clear coefficient interpretations

---

## 🎯 Research Questions

1. **Can we accurately predict median earnings** for engineering graduates three years after graduation using institutional and program characteristics?

2. **Which factors matter most** in explaining earnings differences across engineering programs, and how do they interact?

3. **Can programs be reliably classified** into actionable earnings tiers (Low/Medium/High/Very High)?

---

## 📊 Dataset

### Data Sources

⚠️ **IMPORTANT:** The raw data files (240 MB total) are **NOT included in this repository** due to their size. You must download them separately to reproduce the analysis.

**Download Links:**
- **Field of Study Dataset (142 MB):** [Most-Recent-Cohorts-Field-of-Study.csv](https://collegescorecard.ed.gov/data/)
- **Institution Dataset (98 MB):** [Most-Recent-Cohorts-Institution.csv](https://collegescorecard.ed.gov/data/)
- **Direct Download Page:** https://collegescorecard.ed.gov/data/

After downloading, place both CSV files in the `data/` folder of this project.

### Dataset Overview
- **Field of Study Dataset:** 229,188 program records (142 MB)
- **Institution Dataset:** 6,429 institutions with 3,306 variables (98 MB)
- **Source:** U.S. Department of Education College Scorecard

### Engineering Programs
- **Total Programs:** 8,880 engineering programs
- **With Earnings Data:** 8,230 programs
- **CIP Codes:** 1400-1499 (all engineering disciplines)

### Key Variables
- **Outcome:** Median earnings 3 years after graduation (`MD_EARN_WNE_INC3_P7`)
- **Predictors:** Tuition, faculty salary, SAT/ACT scores, admission rates, completion rates, demographics, institutional spending

### Earnings Distribution
| Statistic | Value |
|-----------|-------|
| Minimum | $24,920 |
| 25th Percentile | $47,044 |
| Median | $54,679 |
| 75th Percentile | $65,766 |
| Maximum | $140,193 |

---

## 🔧 Methodology

### 1. Data Integration & Cleaning
- Merged Field of Study and Institution datasets on `UNITID`
- Handled privacy suppression and missing values
- Filtered to engineering programs (CIP codes 1400-1499)
- Imputed missing values using median strategy

### 2. Feature Engineering
Created **10 engineered features** from **18 base features**:

**Interaction Terms (3):**
- `SELECTIVITY_X_RESOURCES`: SAT × Faculty Salary
- `SELECTIVITY_X_COMPLETION`: SAT × Completion Rate
- `RESOURCES_PER_PELL`: Spending / Pell Grant %

**Polynomial Terms (2):**
- `SAT_SQUARED`: Captures non-linear selectivity effects
- `ADM_RATE_SQUARED`: Models diminishing returns

**Ratio Features (1):**
- `SELECTIVITY_EFFICIENCY`: SAT / Admission Rate

**Categorical Encodings (4):**
- `IS_PRIVATE_NONPROFIT`, `IS_PUBLIC`
- `IS_URBAN`
- `IS_ELITE` (SAT > 1400)

**Total Features:** 27 (after removing 1 with >50% missing values)

### 3. Modeling Approaches

#### Regression Models (Predicting Exact Earnings)
| Model | R² Score | MAE | RMSE | Key Characteristics |
|-------|----------|-----|------|---------------------|
| Linear Regression | 0.7452 | $5,565 | $7,801 | Interpretable coefficients |
| Ridge (L2) | 0.7453 | $5,561 | $7,800 | Regularized, prevents overfitting |
| Lasso (L1) | 0.7409 | $5,607 | $7,867 | Feature selection (16/27 features) |
| **Random Forest** | **0.9835** | **$571** | **$1,985** | **Best predictive performance** |

#### Classification Models (Predicting Earnings Category)
| Model | Accuracy | Notes |
|-------|----------|-------|
| Logistic Regression | 69.4% | Baseline linear classifier |
| Random Forest | 97.7% | High accuracy, robust |
| **Gradient Boosting** | **97.9%** | **Best classification performance** |

**Quartile Categories:**
- Low: $24,920 - $47,038
- Medium: $47,062 - $54,679
- High: $54,725 - $65,766
- Very High: $65,993 - $140,193

---

## 🏆 Key Findings

### Feature Importance (from Random Forest)
1. **Out-of-State Tuition (59.2%)** - Dominant predictor, proxy for prestige/resources
2. **Asian Student Percentage (10.4%)** - Demographics matter
3. **Faculty Salary (5.2%)** - Institutional investment
4. **Completion Rate (3.3%)** - Student success indicator
5. **SELECTIVITY_X_COMPLETION (2.4%)** - Engineered interaction term

### Interpretable Insights (from Linear Regression)
- **Faculty Salary:** +$4,611 per unit increase
- **Out-of-State Tuition:** Positive association with earnings
- **Admission Rate:** -$3,412 per unit increase (lower = more selective = higher earnings)
- **SAT Score:** Positive effect with diminishing returns (negative SAT² coefficient)

### Classification Performance
- **97.9% accuracy** in predicting earnings tier
- Random guessing would achieve ~25% accuracy
- Practical for student/administrator decision-making

---

## 📁 Project Structure

```
Engineering_Analysis_Project/
├── README.md                                         # This file (project documentation)
│
├── data/                                             # ⚠️ NOT INCLUDED - DOWNLOAD SEPARATELY
│   ├── Most-Recent-Cohorts-Field-of-Study.csv      # 142 MB - Download from collegescorecard.ed.gov
│   └── Most-Recent-Cohorts-Institution.csv         # 98 MB  - Download from collegescorecard.ed.gov
│
├── notebooks/
│   └── Complete_Engineering_Earnings_Analysis.ipynb # Main analysis notebook (SUBMISSION FILE)
│
├── archive/                                          # Analysis outputs
│   ├── enhanced_regression_analysis.py              # Python script version
│   ├── enhanced_analysis_results.png                # 6-panel visualization
│   ├── feature_importance_detailed.csv              # 27 features ranked
│   ├── linear_regression_coefficients.csv           # Interpretable coefficients
│   ├── lasso_selected_features.csv                  # 16 selected features
│   ├── regression_model_comparison.csv              # Model performance metrics
│   └── classification_model_comparison.csv          # Classification accuracies
│
└── .venv/                                            # Python virtual environment (auto-generated)
```

**Note on Data Files:**
- The `data/` folder is included in the repository structure but the CSV files are **NOT tracked** due to size (240 MB total)
- You **MUST download these files manually** from https://collegescorecard.ed.gov/data/
- Without these files, the analysis cannot run

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+ (tested on 3.11.9)
- Jupyter Notebook or VS Code with Jupyter extension
- ~500 MB free disk space (for data files)
- ~2 GB RAM (for processing)

### Required Dependencies
```bash
pandas==2.3.3
numpy==2.3.5
scikit-learn==1.8.0
matplotlib==3.10.8
seaborn==0.13.2
scipy==1.16.3
jupyter
```

---

## 📥 Step-by-Step Reproduction Instructions

### Step 1: Clone or Download the Project
```bash
# Navigate to your desired location
cd path/to/your/projects

# If you have the project folder, navigate into it
cd Engineering_Analysis_Project
```

### Step 2: Download the Data Files

⚠️ **CRITICAL STEP:** The data files are NOT included in this repository.

1. Visit the **College Scorecard Data Download Page:**
   - **URL:** https://collegescorecard.ed.gov/data/

2. Download these two files:
   -Step 5: Run the Analysis

**Option 1: Jupyter Notebook (Recommended - Shows Full Process)**

```bash
# Launch Jupyter
jupyter notebook

# Navigate to: notebooks/Complete_Engineering_Earnings_Analysis.ipynb
# Then: Click "Run All" or run cells sequentially
```

**Option 2: Python Script (Automated Execution)**

```bash
# Run the complete analysis script
python archive/enhanced_regression_analysis.py
```

**Option 3: VS Code (If you prefer VS Code)**

1. Open the project folder in VS Code
2. Open `notebooks/Complete_Engineering_Earnings_Analysis.ipynb`
3. Select Python interpreter (`.venv` or your conda environment)
4. Click "Run All" at the top of the notebook

---

## ⏱️ Expected Runtime

| Task | Time | Notes |
|------|------|-------|
| Data loading | ~10-15 sec | Loading 240 MB CSV files |
| Data cleaning & merging | ~5 sec | Filtering to 8,880 engineering programs |
| Feature engineering | ~2 sec | Creating 10 engineered features |
| Linear/Ridge/Lasso | ~1 sec each | Fast for linear models |
| Random Forest | ~5 sec | 100 trees, parallel processing |
| Gradient Boosting | ~7-10 sec | Sequential tree building |
| Visualization | ~2 sec | 6-panel matplotlib figure |
| **Total Runtime** | **~30-60 sec** | Complete end-to-end execution |

---

## ✅ Verifying Successful Execution

After running, you should see these outputs:

### Console Output Indicators
```
✓ All libraries imported successfully
✓ Data cleaning complete
✓ Final dataset: 8,880 engineering programs
✓ Feature Engineering Summary: 27 total features
✓ Regression Models Complete
✓ Classification Models Complete
✓ Visualization saved
✓ Results saved successfully
```

### Expected Results Files (in `archive/`)
- ✅ `enhanced_analysis_results.png` (6-panel visualization)
- ✅ `feature_importance_detailed.csv` (27 rows)
- ✅ `linear_regression_coefficients.csv` (27 rows)
- ✅ `lasso_selected_features.csv` (16 rows)
- ✅ `regression_model_comparison.csv` (4 rows)
- ✅ `classification_model_comparison.csv` (3 rows)

### Expected Performance Metrics
- **Random Forest R²:** ~0.98 (98%+ variance explained)
- **Gradient Boosting Accuracy:** ~97.9%
- **Linear Regression R²:** ~0.75

If your results differ significantly, ensure you downloaded the correct data files.rnings_Analysis.ipynb
└── README.md
```

### Step 3: Set Up Python Environment

**Option A: Using Virtual Environment (Recommended)**
```bash
# Create virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac/Linux

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn scipy jupyter
```

**Option B: Using Conda**
```bash
conda create -n earnings_analysis python=3.11
conda activate earnings_analysis
pip install pandas numpy scikit-learn matplotlib seaborn scipy jupyter
```

### Step 4: Verify Data Files
```bash
# Check that data files exist
ls data/

# You should see:
# Most-Recent-Cohorts-Field-of-Study.csv
# Most-Recent-Cohorts-Institution.csv
```

If files are missing, **STOP** and return to Step 2.

### Running the Analysis

**Option 1: Jupyter Notebook (Recommended for exploration)**
```bash
jupyter notebook notebooks/Complete_Engineering_Earnings_Analysis.ipynb
```
Then run all cells sequentially.

**Option 2: Python Script (Automated execution)**
```bash
python archive/enhanced_regression_analysis.py
```

### Expected Runtime
- Full notebook execution: ~30-60 seconds
- Data loading: ~10 seconds
- Random Forest training: ~5 seconds
- Gradient Boosting: ~7 seconds

---

## 📈 Outputs

### Generated Files (saved to `archive/`)

1. **enhanced_analysis_results.png**
   - 6-panel comprehensive visualization
   - Feature importance charts
   - Predicted vs. actual plots
   - Model comparison bars

2. **feature_importance_detailed.csv**
   - All 27 features ranked by Random Forest importance
   🆘 Troubleshooting

### "FileNotFoundError: data/Most-Recent-Cohorts-Field-of-Study.csv"
**Solution:** You haven't downloaded the data files. See Step 2 above.

### "ModuleNotFoundError: No module named 'pandas'"
**Solution:** Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

### Data files are in the wrong location
**Solution:** Ensure your folder structure matches:
```
Engineering_Analysis_Project/
├── data/
│   ├── Most-Recent-Cohorts-Field-of-Study.csv  ← HERE
│   └── Most-Recent-Cohorts-Institution.csv     ← HERE
└── notebooks/
    └── Complete_Engineering_Earnings_Analysis.ipynb
```

### Results differ from expected values
**Solution:** Ensure you downloaded the **most recent cohort data** from College Scorecard. Older versions will produce different results.

### Out of memory errors
**Solution:** The analysis requires ~2 GB RAM. Close other applications or use a machine with more memory.

---

## 📚 References & Data Sources

### Primary Data Source
1. **U.S. Department of Education College Scorecard**  
   - **Main Page:** https://collegescorecard.ed.gov/
   - **Data Download:** https://collegescorecard.ed.gov/data/
   - **Documentation:** https://collegescorecard.ed.gov/data/documentation/
   - **Data Dictionary:** Available on download page

### Technical References
2. **Scikit-learn: Machine Learning in Python**  
   Pedregosa et al., JMLR 12, pp. 2825-2830, 2011  
   https://scikit-learn.org/

3. **Pandas: Powerful Python Data Analysis Toolkit**  
   McKinney, W. (2010)  
   https://pandas.pydata.org/

4. **Random Forests**  
   Breiman, L. (2001). Machine Learning, 45(1), 5-32

### How to Cite This Project
```
Krasny, E., Ota, J., Wright, W., & Zell, E. (2025). 
Predicting Engineering Graduate Earnings Using Institutional and Program-Level Data.
University of Virginia, SYS 3501 Final Project.
```
6. **classification_model_comparison.csv**
   - Accuracy scores for all 3 classifiers

---

## 💡 Key Insights for Stakeholders

### For Students
- **Institutional prestige matters:** Out-of-state tuition (proxy for prestige) is the strongest predictor
- **Selectivity helps, but plateaus:** Very high SAT scores show diminishing returns
- **Completion matters:** Schools that retain students tend to produce higher earners

### For Universities
- **Faculty investment pays off:** Higher faculty salaries associate with better outcomes
- **Resource allocation matters:** Spending per student, especially for low-income students
- **Completion rates are critical:** Focus on student success, not just admissions

### For Policymakers
- **Earnings vary widely:** Even within engineering, a high-paying field, $24k-$140k range exists
- **Inequality patterns:** Demographics and institutional type create disparities
- **Actionable categories:** 98% accurate classification enables targeted interventions

---

## ⚠️ Limitations & Caveats

1. **Correlation ≠ Causation:** High SAT schools correlate with high earnings, but attending one may not *cause* higher earnings (selection bias)

2. **Selection Bias:** Better students self-select into better schools

3. **Missing Context:** Earnings vary by region, industry, individual choices not captured here

4. **Snapshot Data:** One year of data; trends may change over time

5. **Engineering Only:** Patterns may differ for other fields

6. **No Cost-of-Living Adjustment:** $60k in NYC ≠ $60k in rural areas

---

## 🔮 Future Work

- **Temporal Analysis:** Track earnings trajectories over 5-10 years
- **Geographic Controls:** Add regional cost-of-living adjustments
- **Industry Breakdowns:** Analyze by engineering specialty and sector
- **Causal Inference:** Use quasi-experimental designs (e.g., regression discontinuity)
- **Interactive Dashboard:** Deploy Streamlit/Dash app for real-time predictions
- **Alumni Network Metrics:** Incorporate LinkedIn/professional network data

---

## 📚 References

1. **U.S. Department of Education College Scorecard**  
   https://collegescorecard.ed.gov/data/

2. **Scikit-learn: Machine Learning in Python**  
   Pedregosa et al., JMLR 12, pp. 2825-2830, 2011

3. **Pandas: Powerful Python Data Analysis Toolkit**  
   McKinney, W. (2010)

4. **Random Forests**  
   Breiman, L. (2001). Machine Learning, 45(1), 5-32

---

## 👥 Authors & Contributions

- **Elias Krasny:** 
- **Jiei Ota:**
- **William Wright:** 
- **Elias Zell:** 

---

## 📄 License

This project uses publicly available data from the U.S. Department of Education.  
Analysis code and documentation are available for educational purposes.

---

## 🙏 Acknowledgments

- **U.S. Department of Education** for making College Scorecard data publicly available
- **SYS 3501 Course Staff** for guidance and feedback
- **University of Virginia** for computational resources

---

## 📧 Contact

For questions or collaboration opportunities, please contact the authors through the University of Virginia SYS 3501 course.

---

**Last Updated:** December 19, 2025  
**Version:** 1.0
