import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



data = pd.read_csv('lifeexpectancydata.csv')

#Step 1 Data Cleaning

data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
data.fillna(data.select_dtypes(include=np.number).mean(), inplace=True)
if 'status' in data.columns:
    data['status'].fillna(data['status'].mode()[0], inplace=True)
if 'gdp' in data.columns:
    data['gdp'] = pd.to_numeric(data['gdp'], errors='coerce')
data.drop_duplicates(inplace=True)


#STEP 2 Feature Engineering

data['status'] = data['status'].map({'Developing':0,'Developed':1})

data['schooling_gdp'] = data['gdp'] * data['schooling']

numerical_cols = ['gdp','schooling','alcohol','bmi','schooling_gdp']

scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

#STEP 3 Exploratory Data Analysis

sns.histplot(data['life_expectancy'], kde=True)
plt.title('Distribution of Life expectancy')
plt.xlabel('Life expectancy')
plt.ylabel('Frequency')


corr = data.drop(columns=['country', 'year']).corr(numeric_only=True)

plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True,cmap='coolwarm',fmt='.2f')
plt.title('correlation heatmap')


sns.boxplot(x=data['gdp'])
plt.title('gdp outliers')


sns.regplot(x='gdp',y='life_expectancy',data=data)
plt.title('gdp vs life expectancy')


sns.pairplot(data[['gdp','bmi','life_expectancy','schooling','schooling_gdp','alcohol',]])


#STEP 4 TRAINING MODELS

X = data.drop(columns=['life_expectancy', 'country', 'year'])
y = data['life_expectancy']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred_lr = lr.predict(X_test)

ridge = Ridge()
ridge.fit(X_train,y_train)
y_pred_ridge = ridge.predict(X_test)

lasso = Lasso()
lasso.fit(X_train,y_train)
y_pred_lasso = lasso.predict(X_test)

rf = RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)


#Step 5 evaluate and compare Models
def evaluatemodel(name,y_true,y_pred):
    print(name)
    print('R2 Score',r2_score(y_true,y_pred))
    print('MAE',mean_absolute_error(y_true,y_pred))
    print('MSE',mean_squared_error(y_true,y_pred))
    print('-'*30)

evaluatemodel('Linear Regression',y_test,y_pred_lr)
evaluatemodel('Ridge Regression',y_test,y_pred_ridge)
evaluatemodel('Lasso Regression',y_test,y_pred_lasso)
evaluatemodel('Random Forest',y_test,y_pred_rf)

#Step 6 Visualisation

residuals = y_test - y_pred_lr
sns.histplot(residuals, kde=True)
plt.title('Residual Distribution using Linear Regression')
plt.xlabel('Residual')



importances = rf.feature_importances_
features = X.columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances,y=features)
plt.title('Feature Importance using Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Feature')



plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_rf, alpha=0.6, label='Random Forest')
plt.scatter(y_test, y_pred_lr, alpha=0.6, label='Linear Regression')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("Actual Life Expectancy")
plt.ylabel("Predicted Life Expectancy")
plt.title("Actual vs Predicted")
plt.legend()
plt.show()
