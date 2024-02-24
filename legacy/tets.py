import pandas as pd
from sklearn import preprocessing
from collections import defaultdict
import joblib

# Step  1: Create a sample DataFrame
df = pd.DataFrame({
    'pets': ['cat', 'dog', 'cat', 'monkey', 'dog', 'dog'],   
    'owner': ['Champ', 'Ron', 'Brick', 'Champ', 'Veronica', 'Ron'],   
    'location': ['San_Diego', 'New_York', 'New_York', 'Washington', 'San_Diego', 'New_York']
})

# Step  2: Fit and transform data with LabelEncoder
# Dictionary to hold LabelEncoder instances
d = defaultdict(preprocessing.LabelEncoder)

# Fit and transform data
fit = df.apply(lambda x: d[x.name].fit_transform(x))


print(fit)
# Step  3: Save the dictionary of LabelEncoder instances to a file
joblib.dump(d, 'label_encoder_dict.joblib')

# For demonstration, let's simulate loading the encoders in a new script or session
# Step  4: Load the dictionary of LabelEncoder instances from the file
d = joblib.load('label_encoder_dict.joblib')

# Step  5: Encode new data with the loaded encoders
new_data = pd.DataFrame({
    'pets': ['cat', 'dog', 'monkey'],   
    'owner': ['Champ', 'Ron', 'Brick'],   
    'location': ['San_Diego', 'New_York', 'Washington']
})

new_data_encoded = new_data.apply(lambda x: d[x.name].transform(x))
print("Encoded New Data:")
print(new_data_encoded)

# Step  6: Decode predictions with the loaded encoders
# Assuming 'fit' is our encoded data, we can decode it
predictions_decoded = fit.apply(lambda x: d[x.name].inverse_transform(x))
print("\nDecoded Predictions:")
print(predictions_decoded)