import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import torch
from imblearn.over_sampling import SMOTE, ADASYN
from statsmodels.stats.outliers_influence import variance_inflation_factor
import sys

class Lab2Class:

    def __init__(self):
        self.number = 0

    def preprocess_fit(self):
        # Step  1: Load the dataset
        data = pd.read_csv('D:\\CodingAI\\HW2\\Lab2\\train.csv')

        # Step  2: Rename the last column as 'output'
        data.rename(columns={'BANKR': 'output'}, inplace=True)

        # # Step  3: Handle missing values if any (optional)
        # data.fillna(data.mean(), inplace=True)

        # # Step  4: Split the data into majority and minority classes
        # majority_class = data[data['output'] ==  0]
        # minority_class = data[data['output'] ==  1]

        # The dataset is large so we can delete the majority_class for balancing
        # majority_upsampled = resample(majority_class, replace=True, n_samples=len(minority_class), random_state=42)
        # data = pd.concat([majority_upsampled, minority_class])

        # Step  7: Split into features and target
        # self.id_firm = data['ID_FIRM']
        # X = data.drop(['output'], axis=1)
        # y = data['output']

        X = data.drop(['ID_FIRM', 'output'], axis = 1).reset_index(drop = True)
        y = data['output'].reset_index(drop = True)
        X_resample, y = SMOTE().fit_resample(X,y)

        vif_data = pd.DataFrame()
        vif_data['Features'] = X_resample.columns
        vif_data['VIF'] = [variance_inflation_factor(X_resample.values, i) for i in range(len(X_resample.columns))]

        while any(vif_data['VIF']>=2.8):
            i  = vif_data['VIF'].idxmax()
            X_resample.drop(X_resample.columns[i], axis = 1, inplace = True)
            X_resample.reset_index(drop =True)
            vif_data = pd.DataFrame()
            vif_data['Features'] = X_resample.columns
            vif_data['VIF'] = [variance_inflation_factor(X_resample.values, i) for i in range(len(X_resample.columns))]

        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X_resample), columns = X_resample.columns)
        self.number = X.shape[1]
        
        return X, y

    def preprocess_fit100(self):
        # Step  1: Load the dataset
        data = pd.read_csv('D:\\CodingAI\\HW2\\Lab2\\train_100.csv')

        # Step  2: Rename the last column as 'output'
        data.rename(columns={'BANKR': 'output'}, inplace=True)

        # Step  3: Handle missing values if any (optional)
        data.fillna(data.mean(), inplace=True)

        # Step  4: Now we need to obtain the imbalance dataset
        majority_class = data[data['output'] ==  0]
        minority_class = data[data['output'] ==  1]
        minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
        data = pd.concat([majority_class, minority_upsampled])

        # Step  7: Split into features and target
        X = data.drop(['ID_FIRM', 'output'], axis=1)
        y = data['output']

        print(X.shape)
        return X, y

    def preprocess_test(self):

        df = pd.read_csv('D:\\CodingAI\\HW2\\Lab2\\test.csv')
        df_result = pd.read_csv('D:\\CodingAI\\HW2\\Lab2\\sample_submission.csv')
        X = df.drop('ID_FIRM', axis= 1)
        y = df_result['BANKR']
        id_firm = df_result['ID_FIRM']

        return X, y, id_firm

    def get_nn_param(self):
        input_size = self.number
        output_size = 2
        learning_rate = 0.01
        loss_method = torch.nn.CrossEntropyLoss()
        return input_size, output_size, learning_rate, loss_method
    
    def output_mod(self, output, target):
        # output = torch.sigmoid(output)
        return output, target.long()

    def output_mod_test(self, output, target):
        # output = torch.sigmoid(output)
        return output.argmax(1), target.long()

    def lab_dir(self):
        return 'D:\\CodingAI\\HW2\\Lab2\\result\\'

    def check_nan_in_dataset(self, data):
        return data.isnull().sum()

if __name__ == '__main__':

    lab = Lab2Class()
    data, target = lab.preprocess_fit()
    
    print('#'*80)
    print(lab.check_nan_in_dataset(data))
    print('#'*80)
    print(lab.check_nan_in_dataset(target))

    data.to_csv("data_out.csv")
    target.to_csv("target_out.csv")
    