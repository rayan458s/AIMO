from AIMO import data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Specify the columns to keep
columns_to_keep = [
    'property_type', 'postal_code', 'size', 'floor', 'land_size',
    'energy_performance_category', 'exposition', 'nb_rooms',
    'nb_bedrooms', 'nb_bathrooms', 'nb_parking_places', 'nb_boxes',
    'has_a_balcony', 'nb_terraces', 'has_air_conditioning'
]

# Load data
x_train_raw = pd.read_csv('/Users/rayan/PycharmProjects/rakam-systems-service-template/AIMO/data/X_train.csv')
y_train_raw = pd.read_csv('/Users/rayan/PycharmProjects/rakam-systems-service-template/AIMO/data/y_train.csv')
x_test_raw = pd.read_csv('/Users/rayan/PycharmProjects/rakam-systems-service-template/AIMO/data/X_test.csv')
y_test_raw = pd.read_csv('/Users/rayan/PycharmProjects/rakam-systems-service-template/AIMO/data/y_test.csv')

# Keep only the specified  (Property_type, postal_code, 'size', 'floor','land_size','energy_performance_category','exposition', 'nb_rooms', 'nb_bedrooms', 'nb_bathrooms', 'nb_parking_places', 'nb_boxes', 'has_a_balcony', 'nb_terraces', 'has_air_conditioning'
x_train_raw = x_train_raw[columns_to_keep]
x_test_raw = x_test_raw[columns_to_keep]

# Separate the numerical and categorical columns
num_cols_train = x_train_raw.select_dtypes(include=['int64', 'float64']).columns
cat_cols_train = x_train_raw.select_dtypes(include=['object']).columns
num_cols_test = x_test_raw.select_dtypes(include=['int64', 'float64']).columns
cat_cols_test = x_test_raw.select_dtypes(include=['object']).columns

# Impute missing values in numerical columns with the mean
x_train_raw[num_cols_train] = x_train_raw[num_cols_train].fillna(x_train_raw[num_cols_train].mean())
x_test_raw[num_cols_test] = x_test_raw[num_cols_test].fillna(x_test_raw[num_cols_test].mean())

# Impute missing values in categorical columns with the most frequent value (mode)
x_train_raw[cat_cols_train] = x_train_raw[cat_cols_train].apply(lambda x: x.fillna(x.mode()[0]))
x_test_raw[cat_cols_test] = x_test_raw[cat_cols_test].apply(lambda x: x.fillna(x.mode()[0]))

# One-hot encode the categorical columns
x_train = pd.get_dummies(x_train_raw, drop_first=True)
x_test = pd.get_dummies(x_test_raw, drop_first=True)

# Align columns of train and test sets after one-hot encoding
x_train, x_test = x_train.align(x_test, join='left', axis=1, fill_value=0)

# Convert target variable to a Series
y_train = y_train_raw.squeeze()
y_test = y_test_raw.squeeze()

# # Debugging: Check for missing values in the final preprocessed data
# print("Missing values in x_train after preprocessing:", x_train.isnull().any().any())
# print("Missing values in x_test after preprocessing:", x_test.isnull().any().any())
# print("Missing values in y_train:", y_train.isnull().any())
# print("Missing values in y_test:", y_test.isnull().any())

# Initialize the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model on the training data
print("\nStarting training the model...")
rf_model.fit(x_train, y_train)

# Save the model to a file
joblib.dump(rf_model, 'random_forest_model.pk1')
print("\nModel saved successfully.")

# Make predictions on the test set
y_pred = rf_model.predict(x_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation results
print(f"\nMean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")




# # Visualize predictions vs actuals
# sns.scatterplot(x=y_test, y=y_pred)
# plt.xlabel("Actual Prices")
# plt.ylabel("Predicted Prices")
# plt.title("Actual vs Predicted Prices")
# plt.show()

# ##################### GRID SEARCH
#
# from sklearn.model_selection import GridSearchCV
#
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [10, 20, None],
#     'max_features': ['auto', 'sqrt', 'log2']
# }
#
# grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
# grid_search.fit(x_train, y_train)
#
# # Best parameters from the grid search
# print("Best parameters found: ", grid_search.best_params_)
#
#
#
# ##################### CROSS VALIDATION
#
# from sklearn.model_selection import cross_val_score
#
# # Perform 5-fold cross-validation on training data
# cv_scores = cross_val_score(rf_model, x_train, y_train, cv=5, scoring='r2')
#
# print(f"Cross-validated R2 score: {cv_scores.mean()} ± {cv_scores.std()}")
#
#
#
# ##################### FEATURE IMPORTANCE
#
#
# # Plot feature importances
# importances = rf_model.feature_importances_
# features = x_train.columns
# indices = np.argsort(importances)[::-1]
#
# plt.figure(figsize=(10,6))
# plt.title("Feature Importances")
# plt.bar(range(x_train.shape[1]), importances[indices], align="center")
# plt.xticks(range(x_train.shape[1]), [features[i] for i in indices], rotation=90)
# plt.show()# Plot feature importances
# importances = rf_model.feature_importances_
# features = x_train.columns
# indices = np.argsort(importances)[::-1]
#
# plt.figure(figsize=(10,6))
# plt.title("Feature Importances")
# plt.bar(range(x_train.shape[1]), importances[indices], align="center")
# plt.xticks(range(x_train.shape[1]), [features[i] for i in indices], rotation=90)
# plt.show()

