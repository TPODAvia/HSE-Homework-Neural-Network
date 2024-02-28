import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib
from collections import defaultdict
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

class Lab1Class:

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        self.number = 29

    def preprocess_fit(self):
        df = pd.read_csv('D:\\CodingAI\\HW2\\Lab1\\train.csv')

        print(df)
        df.rename(columns={'Price': 'output'}, inplace=True)
        X = df.drop(['Id', 'output', 'Room', 'Size', 'Floor','District', 'FloorsTotal'], axis=1)
        y = df['output']

        d = defaultdict(LabelEncoder)
        fit =  X.apply(lambda x: d[x.name].fit_transform(x))
        joblib.dump(d, 'label_encoder_dict.joblib')

        one_hot_encoder = OneHotEncoder(sparse=False)
        fit = pd.DataFrame(one_hot_encoder.fit_transform(fit))
        joblib.dump(one_hot_encoder, 'one_hot_encoder.joblib')

        X = pd.concat([fit, df[['Room', 'Size', 'Floor', 'FloorsTotal']]], axis=1)

        self.number = X.shape[1]

        return X, y

    def preprocess_test(self):
        df = pd.read_csv('D:\\CodingAI\\HW2\\Lab1\\test.csv')
        X = df.drop(['Id', 'Room', 'Size', 'Floor', 'District', 'FloorsTotal'], axis=1)
        y = pd.read_csv('D:\\CodingAI\\HW2\\Lab1\\sample_submission.csv')
        y_id = y['Id']
        y = y['Price'].astype(int)
        y = y.astype(float)
        # X = df.drop(['Id', 'Price', 'Room', 'Size', 'Floor', 'District', 'FloorsTotal'], axis=1)
        # y = pd.read_csv('D:\\CodingAI\\HW2\\Lab1\\sample_submission.csv')
        # y_id = df['Id']
        # y = df['Price'].astype(int)

        d = defaultdict(LabelEncoder)
        d = joblib.load('label_encoder_dict.joblib')
        fit =  X.apply(lambda x: d[x.name].fit_transform(x))

        one_hot_encoder = joblib.load('one_hot_encoder.joblib')
        fit = pd.DataFrame(one_hot_encoder.transform(fit))

        out_X = pd.concat([fit, df[['Room', 'Size', 'Floor', 'FloorsTotal']]], axis=1)

        return out_X, y, y_id

    def label_dic_test(self):
        d = joblib.load('label_encoder_dict.joblib')
        df = pd.read_csv('D:\\CodingAI\\HW2\\Lab1\\train_100.csv')
        df.rename(columns={'Price': 'output'}, inplace=True)
        X = df.drop(['Id', 'output', 'Room', 'Size', 'Floor', 'FloorsTotal'], axis=1)

        new_data_encoded = X.apply(lambda x: d[x.name].transform(x))
        print("Encoded New Data:")
        print(new_data_encoded)

    def get_nn_param(self):
        input_size = self.number
        output_size = 1
        learning_rate = 0.01
        loss_method = torch.nn.MSELoss()
        return input_size, output_size, learning_rate, loss_method

    def output_mod(self, output, target):
        # Add an extra dimension to match the output shape
        # loss = criterion(output, target)
        return output, target.unsqueeze(1)

    def output_mod_test(self, output, target):
        # Add an extra dimension to match the output shape
        # loss = criterion(output, target)
        return output, target.unsqueeze(1)

    def lab_dir(self):
        return 'D:\\CodingAI\\HW2\\Lab1\\result\\'

    def check_nan_in_dataset(self, data):
        return data.isnull().sum()

if __name__ == '__main__':

    lab = Lab1Class()
    data, target, *_ = lab.preprocess_fit()
    
    print(data)

    print('#'*80)
    print(lab.check_nan_in_dataset(data))
    print('#'*80)
    print(lab.check_nan_in_dataset(target))

    # Load the test data
    data.to_csv("data_out.csv")
    target.to_csv("target_out.csv")
    
    # print('#'*80)
    # lab.label_dic_test()