import pandas as pd
from sklearn import preprocessing
from collections import defaultdict
import joblib

d = joblib.load('label_encoder_dict.joblib')

# Step  5: Encode new data with the loaded encoders
new_data = pd.DataFrame({
    'pets': ['dog', 'monkey', 'monkey', 'monkey'],   
    'owner': ['Champ', 'Ron', 'Brick', 'Ron'],   
    'location': ['San_Diego', 'New_York', 'Washington', 'New_York']
})

new_data_encoded = new_data.apply(lambda x: d[x.name].transform(x))
print("Encoded New Data:")
print(new_data_encoded)
