import torch
import sys
from  model import Net
from Lab1.lab1_prep import Lab1Class
from Lab2.lab2_prep import Lab2Class
from Lab3.lab3_prep import Lab3Class

PATH = "model.pth"
lab = Lab3Class()

data_input_size, data_output_size, *_ = lab.get_nn_param()
model = Net(input_size=data_input_size, output_size=data_output_size)
model.load_state_dict(torch.load(PATH))
model.eval() # turn the model to evaluation mode

test_data, y_data, y_id = lab.preprocess_test()

num = 1
single_sample = torch.tensor(test_data.iloc[num], dtype=torch.float32).unsqueeze(0)

print('#'*80)
print(single_sample)
output = lab.output_mod(model(single_sample))

print('#'*80)
print(f"{y_id[num]} | {output} | {y_data.iloc[num]} \n")