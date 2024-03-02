import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.ensemble import BalancedRandomForestClassifier

# Load the dataset
data = pd.read_csv(r'D:\CodingAI\HW2\Lab2\train.csv')

# Split the dataset into features and the target variable
X = data.drop(['ID_FIRM','BANKR'], axis=1)
y = data['BANKR']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Use BalancedRandomForestClassifier instead of RandomForestClassifier
clf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


test = pd.read_csv(r"D:\CodingAI\HW2\Lab2\test.csv")
X_test = test.drop(['ID_FIRM'], axis=1)
X_test_scaled = scaler.transform(X_test)
y_pred = clf.predict(X_test_scaled)
out_data = pd.DataFrame({'ID_FIRM': test['ID_FIRM'], 'BANKR': y_pred})
out_data.to_csv(r'D:\CodingAI\HW2\Lab2\result\result_ML.csv', index=False)
