## Life Expectancy Analysis and Prediction

### Overview

This project analyzes global life expectancy data using machine learning techniques to identify key factors that influence life expectancy across different countries and years. The analysis includes comprehensive data preprocessing, exploratory data analysis, feature engineering, and predictive modeling.


### Visualizations Generated

Life Expectancy Distribution: Histogram with KDE curve
distribution_life_expectancy.png
Correlation Heatmap: Complete feature correlation matrix
correlation_heatmap.png
GDP Outlier Detection: Box plot showing economic disparities
gdp_outliers.png
GDP vs Life Expectancy: Scatter plot with regression line
gdp_vs_life_expectancy.png
Feature Relationships: Comprehensive pair plot matrix
pairplot_graph.png
Residual Analysis: Distribution using Linear regression
linear_regression_distribution.png
Feature Importance: Random Forest feature ranking
Feature_importance_random_forest.png
Model Comparison: Actual vs Predicted scatter plots
actual_vs_predicted.png

### Dataset

Source: Life expectancy dataset with health, economic, and social indicators
Size: Multiple countries across different years
Features: 20+ variables including GDP, schooling, mortality rates, disease indicators, and more

### Key Findings:

HIV/AIDS (Random Forest importance: ~0.60) - Strongest individual predictor
Adult Mortality (Correlation: -0.70) - Strong negative relationship
Schooling (Correlation: 0.72) - Education strongly predicts longevity
Income Composition (Correlation: 0.69) - Economic development indicator

### Correlation Insights:

Positive Correlations: Schooling, GDP, income composition, immunization rates
Negative Correlations: Adult mortality, HIV/AIDS, infant deaths, malnutrition indicators
Moderate Relationships: BMI (0.56), alcohol consumption, healthcare expenditure
