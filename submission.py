import torch
from torch.utils.data import dataloader, TensorDataset, DataLoader
import os
from model import Net
import pandas as pd

from Lab1.lab1_prep import Lab1Class
from Lab2.lab2_prep import Lab2Class
from Lab3.lab3_prep import Lab3Class

lab = Lab3Class()

data_input_size, data_output_size, *_ = lab.get_nn_param()
PATH = f"{lab.lab_dir()}model.pth"
model = Net(input_size=data_input_size, output_size=data_output_size)
model.load_state_dict(torch.load(PATH))
model.eval()

test_data, y_data, y_id = lab.preprocess_test()

test_data_tensor = torch.tensor(test_data.values, dtype=torch.float32)
test_target_tensor = torch.tensor(y_data.values, dtype=torch.float32)
test_dataset = TensorDataset(test_data_tensor, test_target_tensor)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Make predictions
predictions = []
with torch.no_grad():
    for data, target in test_loader:
        output = lab.output_mod(model(data)).argmax(1)
        # correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
        predictions.append(output.item())

x_data = pd.Series(predictions)

y_data_array = y_data.to_numpy()
y_data_flat = y_data_array.flatten()


# Now, y_data_flat is a  1-dimensional array
predictions_df = pd.DataFrame({
    'Id': y_id,
    'True': y_data_flat,  # Use the flattened y_data
    'Predicted': x_data.values  # Assuming x_data is already  1-dimensional
})

print(predictions_df)

file_path = f'{lab.lab_dir()}result_dataframe.csv'
print(file_path)
directory = os.path.dirname(file_path)
if not os.path.exists(directory):
    os.mkdir(directory)

predictions_df.to_csv(file_path, index=False)


file_path = f'{lab.lab_dir()}result_submission.csv'
print(file_path)
directory = os.path.dirname(file_path)
if not os.path.exists(directory):
    os.mkdir(directory)


predictions_df.drop('True', axis = 1).to_csv(file_path, index=False)
 