import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data
data = pd.read_csv(r'D:\CodingAI\HW2\Lab3\train.csv')

# Split the data into features (X) and target (y)
X = data.drop(['id', 'y'], axis=1)
y = data['y']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

from imblearn.over_sampling import SMOTE
from sklearn.utils import compute_class_weight

# Assuming you've already split your data into training and testing sets
# and X_train, X_test, y_train, y_test are defined

# Oversampling using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

# Train the model with class weights
model = LogisticRegression(class_weight={0: class_weights[0], 1: class_weights[1]})
model.fit(X_train_resampled, y_train_resampled)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Make predictions on new data
data = pd.read_csv(r'D:\CodingAI\HW2\Lab3\test.csv')
test_data = data.drop('id', axis=1)
predictions = model.predict(test_data)

output = pd.DataFrame({'id':data['id'], 'y': predictions})
output.to_csv(r'D:\CodingAI\HW2\Lab3\result\result_ML.csv', index=False)