import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Optional: Warnings
import warnings
warnings.filterwarnings('ignore')
 
#Data Exploration
df = pd.read_csv(r'C:\Users\athar\OneDrive\Desktop\car\CarPrice.csv')

print(df.info())

print(df.head())


print(df.describe())


print(df.isnull().sum())

sns.pairplot(df)
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

#Data Preprocessing 
df.fillna(df.median(), inplace=True)

df['age'] = 2024 - df['year_of_manufacture']  


df.drop('year_of_manufacture', axis=1, inplace=True)


df = pd.get_dummies(df, columns=['brand'], drop_first=True)


plt.figure(figsize=(10, 6))
sns.boxplot(df['price'])
plt.title('Price Distribution with Outliers')
plt.show()


df = df[df['price'] < df['price'].quantile(0.99)]  
print(df.head())

#Model Development 
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)   

#Model Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared: {r2}')


coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

#Visualization
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Predicted vs Actual Prices')
plt.show()


residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.show()
