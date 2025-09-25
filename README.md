Life Expectancy Analysis and Prediction
Overview
This project analyzes global life expectancy data using machine learning techniques to identify key factors that influence life expectancy across different countries and years. The analysis includes comprehensive data preprocessing, exploratory data analysis, feature engineering, and predictive modeling.


Dataset
Source: Life expectancy dataset with health, economic, and social indicators
Size: Multiple countries across different years
Features: 20+ variables including GDP, schooling, mortality rates, disease indicators, and more

Key Findings:

HIV/AIDS (Random Forest importance: ~0.60) - Strongest individual predictor
Adult Mortality (Correlation: -0.70) - Strong negative relationship
Schooling (Correlation: 0.72) - Education strongly predicts longevity
Income Composition (Correlation: 0.69) - Economic development indicator

Correlation Insights:

Positive Correlations: Schooling, GDP, income composition, immunization rates
Negative Correlations: Adult mortality, HIV/AIDS, infant deaths, malnutrition indicators
Moderate Relationships: BMI (0.56), alcohol consumption, healthcare expenditure

Visualizations Generated

Life Expectancy Distribution: Histogram with KDE curve
Correlation Heatmap: Complete feature correlation matrix
GDP Outlier Detection: Box plot showing economic disparities
GDP vs Life Expectancy: Scatter plot with regression line
Feature Relationships: Comprehensive pair plot matrix
Residual Analysis: Distribution of prediction errors
Feature Importance: Random Forest feature ranking
Model Comparison: Actual vs Predicted scatter plots
