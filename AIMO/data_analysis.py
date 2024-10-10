
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load data
X_train_raw = pd.read_csv('/AIMO_challenge_data/X_train.csv')
y_train_raw = pd.read_csv('/AIMO_challenge_data/y_train.csv')
X_test_raw = pd.read_csv('/AIMO_challenge_data/X_test.csv')
y_test_raw = pd.read_csv('/AIMO_challenge_data/y_test.csv')

# Merge the target price with the input features for easier exploration
data = pd.merge(X_train_raw, y_train_raw, on='id_annonce')

# 1. Overview of the dataset
print("Basic Info of Dataset:")
data.info()
print("\nSummary Statistics:")
print(data.describe())


# 2. Checking for missing values
missing_values = data.isnull().sum()
print("\nMissing Values per Column:")
print(missing_values)

# 3. Distribution of price
plt.figure(figsize=(10,6))
sns.histplot(data['price'], bins=50, kde=True)
plt.title('Distribution of Property Prices')
plt.xlabel('Price (€)')
plt.ylabel('Frequency')
plt.show()

# 4. Categorical feature exploration (Property Type)
plt.figure(figsize=(10,6))
sns.countplot(data=data, x='property_type', order=data['property_type'].value_counts().index)
plt.title('Distribution of Property Types')
plt.xticks(rotation=45)
plt.show()

# 5. Relationship between size and price
plt.figure(figsize=(10,6))
sns.scatterplot(data=data, x='size', y='price', hue='property_type', alpha=0.6)
plt.title('Relationship between Size and Price')
plt.xlabel('Size (sq meters)')
plt.ylabel('Price (€)')
plt.show()

# 6. Correlation heatmap for numeric variables
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(12,8))
sns.heatmap(data[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 7. Checking for outliers in price
plt.figure(figsize=(10,6))
sns.boxplot(x=data['price'])
plt.title('Boxplot for Property Prices')
plt.show()

# 8. Exploring energy performance categories
plt.figure(figsize=(10,6))
sns.countplot(data=data, x='energy_performance_category')
plt.title('Distribution of Energy Performance Categories')
plt.show()