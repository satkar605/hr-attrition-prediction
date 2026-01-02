# HR Employee Attrition Prediction

## Executive Summary

This project develops machine learning models to predict employee attrition using the IBM HR Analytics dataset. The analysis compares six classification algorithms across multiple data balancing strategies to identify employees at risk of leaving, enabling proactive retention interventions.

**Key Business Value:**
- Identifies 59-65% of employees likely to leave (recall metric)
- Enables targeted retention programs before employees depart
- Reduces recruitment and training costs associated with turnover
- Provides actionable insights through feature importance analysis

---

## Problem Statement

Employee turnover is costly for organizations, involving recruitment expenses, knowledge loss, and reduced productivity. Early identification of at-risk employees allows HR teams to implement targeted retention strategies. This project addresses the challenge of predicting attrition using employee demographic, job satisfaction, and work-life balance features.

**Dataset Characteristics:**
- **Total Employees:** 1,470
- **Attrition Cases:** 237 (16.1%)
- **Non-Attrition Cases:** 1,233 (83.9%)
- **Features:** 35 variables including demographics, job satisfaction, compensation, and work-life balance metrics

---

## Methodology

### Data Preprocessing

The analysis begins with comprehensive data exploration and feature engineering:

**Feature Engineering:**
- Created ratio features to capture career progression and compensation efficiency
- Bucketed high-cardinality categorical variables (EducationField, JobRole) into business-meaningful groups
- Removed redundant and uninformative features (EmployeeCount, StandardHours, rate variables)
- Standardized numeric features while preserving binary encodings

**Final Feature Set:** 29 features after engineering and encoding

### Data Balancing Strategies

Given the imbalanced nature of the dataset (16.1% attrition rate), three approaches were evaluated:

1. **Raw/Original Dataset** - Baseline imbalanced dataset
2. **SMOTE Balanced** - Synthetic Minority Oversampling Technique
3. **ADASYN Balanced** - Adaptive Synthetic Sampling

### Models Evaluated

Six machine learning algorithms were trained and compared:

1. **Logistic Regression** - Baseline interpretable linear model
2. **Decision Tree Classifier** - Interpretable tree-based model with hyperparameter tuning
3. **Random Forest Classifier** - Ensemble method with 300 estimators
4. **Gradient Boosting Classifier** - Sequential ensemble with learning rate optimization
5. **Support Vector Classifier (SVC)** - Margin-based classifier with RBF kernel
6. **Multi-layer Perceptron (MLP)** - Feedforward neural network

Each model was evaluated across all three dataset configurations, resulting in 18 model-dataset combinations.

### Evaluation Framework

**Primary Metric: Recall**
- Recall measures the proportion of employees who will leave that are correctly identified
- High recall is critical for HR applications: missing at-risk employees (false negatives) is more costly than false alarms
- Retention interventions are typically cheaper than losing employees

**Comprehensive Metrics:**
- Accuracy
- Precision (positive class)
- Recall (positive class) - **Primary focus**
- F1 Score (positive class)
- ROC-AUC
- Confusion Matrix analysis

---

## Key Findings

### Model Performance Summary

**Best Overall Model (Highest Recall):**
- **Decision Tree** on Raw/Original dataset
- **Recall: 64.79%** - Identifies nearly 65% of employees who will leave
- **Precision: 32.62%** - Trade-off for higher recall
- **ROC-AUC: 69.30%**

**Logistic Regression Results:**
- SMOTE Balanced model achieved **59.15% recall**
- 35.5% improvement over baseline raw model (43.66% recall)
- Demonstrates significant value of data balancing for imbalanced classification

**Tree-Based Models Performance:**
- Decision Tree: Best recall (64.79%) on raw data
- Random Forest: Best precision and overall balance
- Gradient Boosting: Strong performance across metrics

**Advanced Models (SVC & MLP):**
- Competitive performance with tree-based models
- SVC achieved strong precision on balanced datasets
- MLP showed robust performance across all metrics

### Impact of Data Balancing

**SMOTE and ADASYN Balancing:**
- Improved recall by 15-20 percentage points for most models
- Reduced false negatives (missed at-risk employees) significantly
- Acceptable trade-off: slight increase in false positives (false alarms)

**Business Interpretation:**
- Baseline models catch approximately 44% of at-risk employees
- Balanced models catch 59-65% of at-risk employees
- **Improvement: 15-21 more at-risk employees identified per 100**

### Feature Importance Insights

Key factors identified as predictive of attrition:

**High Importance Features:**
- Overtime status
- Business travel frequency
- Job satisfaction levels
- Work-life balance metrics
- Years at company and role stability
- Distance from home
- Department and job role

**Actionable Insights:**
- Employees working overtime show higher attrition risk
- Frequent business travelers are more likely to leave
- Lower job satisfaction correlates with higher turnover
- Work-life balance concerns are strong predictors

---

## Recommendations

### Model Selection for Production

**Primary Recommendation: Decision Tree on Raw Dataset**
- Highest recall (64.79%) - catches the most at-risk employees
- Interpretable model - allows HR teams to understand decision logic
- No data balancing required - simpler deployment pipeline
- Acceptable precision trade-off for HR retention use case

**Alternative: SMOTE-Balanced Logistic Regression**
- Strong recall (59.15%) with better precision
- Linear model provides coefficient interpretability
- Well-suited for explainable AI requirements

### Implementation Strategy

1. **Deployment Approach:**
   - Deploy model to score employees monthly or quarterly
   - Generate risk scores for all active employees
   - Flag employees above threshold for HR review

2. **Intervention Framework:**
   - High-risk employees: Immediate manager and HR engagement
   - Medium-risk employees: Regular check-ins and satisfaction surveys
   - Low-risk employees: Standard retention practices

3. **Model Monitoring:**
   - Track prediction accuracy over time
   - Monitor feature distributions for data drift
   - Retrain model annually or when significant organizational changes occur

4. **Cost-Benefit Analysis:**
   - Cost of false positive: Manager time for unnecessary intervention (~$50-100 per employee)
   - Cost of false negative: Employee departure (~$10,000-50,000+ per employee)
   - Model ROI: High recall minimizes costly false negatives

---

## Project Structure

```
hr-attrition/
├── notebooks/
│   ├── 01-data-exploration.ipynb          # Data cleaning and feature engineering
│   ├── 02-baseline-logistic-regression.ipynb  # Logistic regression models
│   ├── 03-tree-based-models.ipynb         # Decision Tree, Random Forest, Gradient Boosting
│   ├── 04-svc-nn-models.ipynb             # Support Vector Classifier and Neural Network
│   ├── 05-final-model-comparision.ipynb   # Comprehensive model comparison
│   └── hr-attrition-classification-models.ipynb  # Combined analysis notebook
├── data/
│   └── processed/                         # Preprocessed datasets and scalers
├── requirements.txt                       # Python dependencies
└── readme.md                             # This file
```

---

## Technical Requirements

**Python Version:** 3.9.6 or higher

**Key Dependencies:**
- pandas, numpy - Data manipulation
- scikit-learn - Machine learning models
- imbalanced-learn - SMOTE and ADASYN balancing
- matplotlib, seaborn - Visualization
- kagglehub - Dataset access

**Installation:**
```bash
pip install -r requirements.txt
```

---

## Usage

1. **Data Preparation:**
   - Run `01-data-exploration.ipynb` to download, clean, and preprocess the dataset
   - Processed data is saved to `data/processed/` for subsequent notebooks

2. **Model Training:**
   - Execute notebooks 02-04 to train individual model families
   - Each notebook evaluates models on raw, SMOTE, and ADASYN datasets

3. **Model Comparison:**
   - Run `05-final-model-comparision.ipynb` for comprehensive performance analysis
   - Review visualizations and metrics to select production model

4. **Production Deployment:**
   - Extract best model from notebook outputs
   - Implement scoring pipeline using saved model artifacts
   - Set up monitoring and retraining schedule

---

## Limitations and Future Work

**Current Limitations:**
- Model performance constrained by dataset size (1,470 employees)
- Limited external validation (single dataset)
- No temporal analysis (cross-sectional data only)

**Future Enhancements:**
- Incorporate time-series features for longitudinal analysis
- Expand dataset with additional organizations for generalizability
- Develop ensemble models combining top-performing algorithms
- Implement real-time scoring API for production deployment
- Add explainability dashboards for HR stakeholders

---

## Conclusion

This project successfully demonstrates the application of machine learning to predict employee attrition, achieving 59-65% recall in identifying at-risk employees. The analysis provides actionable insights for HR teams to implement targeted retention strategies, with significant potential cost savings through reduced turnover.

The Decision Tree model on raw data emerges as the recommended solution, balancing high recall with interpretability. However, the choice between models should align with organizational priorities: maximizing recall (Decision Tree) versus balancing precision and recall (Logistic Regression with SMOTE).

**Business Impact:**
- Enables proactive retention interventions
- Reduces costly employee turnover
- Provides data-driven insights for HR strategy
- Supports evidence-based decision making

---

## Contact and References

**Dataset Source:**
IBM HR Analytics Employee Attrition Dataset (Kaggle)
- Dataset ID: `pavansubhasht/ibm-hr-analytics-attrition-dataset`

**Methodology References:**
- SMOTE: Synthetic Minority Oversampling Technique
- ADASYN: Adaptive Synthetic Sampling Approach
- Standard machine learning evaluation practices for imbalanced classification

