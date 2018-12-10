from data_processing import data_proc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import boxcox

def split_data(data_frame):
    x_df,y_df =data_frame.loc[:,data_frame.columns != 'Survived'] ,data_frame['Survived']
    return x_df, y_df

get_data = data_proc("./raw_data/train.csv", "./raw_data/test.csv")
train_df, predict_df = get_data.total_data_proc(4)
x_train_df, y_train_df = split_data(train_df)
x_train, x_test, y_train, y_test = train_test_split(x_train_df, y_train_df, test_size=0.2, random_state=100)

# print(train_df.isnull().sum().sort_values(ascending=False))
for item in pd.isnull(predict_df['Age']):
    print(item)
print(predict_df['Age'])



# x_train_cp, x_test_cp = x_train.copy(), x_test.copy()
# scaler = MinMaxScaler()
# x_train_mms, x_test_mms = scaler.fit_transform(x_train_cp), scaler.transform(x_test_cp)

# lr = LogisticRegression(C=1)
# lr.fit(x_train, y_train)
# test_result = lr.predict(x_test)
# tr = np.array(test_result).reshape(-1)
# yt = np.array(y_test).reshape(-1)
# result_dic = {'real value':yt, 'predict value':tr}
# result_df = pd.DataFrame(data=result_dic)
# result_df['result'] = result_df['real value']^result_df['predict value']
# tmp_sum = sum(result_df['result'])
# print(tmp_sum)
