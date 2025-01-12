import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor # Import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error

data1 = pd.read_csv(r"D:\CodingAI\HW2\Lab1\test.csv")
target1 = pd.read_csv(r"D:\CodingAI\HW2\Lab1\sample_submission.csv")
data1["Price"] = target1["Price"]
data = pd.read_csv('D:\\CodingAI\\HW2\\Lab1\\train.csv')

print(data)

# Handle missing values
# Assuming 'Size' and 'Floor' can be imputed with median
# For 'Balcony', 'Walls', 'District', and 'Okrug', we will use a placeholder value or drop them
data['Size'] = data['Size'].fillna(data['Size'].median())
data['Floor'] = data['Floor'].fillna(data['Floor'].median())
# data = data.dropna(subset=['Balcony', 'Walls', 'District', 'Okrug'])

# Define categorical and numerical features
categorical_features = ['Balcony', 'Walls', 'District', 'Okrug']
numerical_features = ['Room', 'Size', 'Floor', 'FloorsTotal']

# Create a column transformer to handle categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Split the data into features and target
X = data.drop(['Id','Price'], axis=1)
y = data['Price']

# # Split the data into training and test sets
# X_train = X
# y_train = y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print(X_train.shape)
print(y_train.shape)
# Create a pipeline that preprocesses data and then trains a Random Forest model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor()) # Use RandomForestRegressor instead of LinearRegression
])

# Train the model
model.fit(X_train, y_train)

data1['Size'] = data1['Size'].fillna(data1['Size'].median())
data1['Floor'] = data1['Floor'].fillna(data1['Floor'].median())
# data = data.dropna(subset=['Balcony', 'Walls', 'District', 'Okrug'])

# Split the data into features and target
X = data1.drop(['Id','Price'], axis=1)
y = data1['Price']


# Make predictions on the test set
y_pred = model.predict(X)

prediction_df = pd.DataFrame({'Id': data1['Id'], 'Price': y_pred})

# Step 3: Save the DataFrame to a CSV file
prediction_df.to_csv(r'D:\CodingAI\HW2\Lab1\result\result_ML.csv', index=False)

rmsle = mean_squared_log_error(y_test, y_pred, squared=False)
print(f'RMSLE: {rmsle}')

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

r2 = r2_score(y_test, y_pred)
print(f'R^2 Score: {r2}')

# 1398803934656959.5
# 2319862054372993
# 539254083373581
# 834532722868224