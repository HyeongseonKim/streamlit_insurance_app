from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle


def insurance_cancellation_predict(client_row):
    loaded_cancel_model=pickle.load(open(r'./data/보험해지모델.sav','rb'))
    result = loaded_cancel_model.predict(client_row)

    return result

def insurance_cancellation_proba(client_row):
    loaded_cancel_model=pickle.load(open(r'./data/보험해지모델.sav','rb'))
    result = loaded_cancel_model.predict_proba(client_row)

    return result

def loan_overdue_predict(client_row):
    loaded_overdue_model=pickle.load(open(r'./data/대출연체모델.sav','rb'))
    result = loaded_overdue_model.predict(client_row)

    return result

def loan_overdue_proba(client_row):
    loaded_overdue_model=pickle.load(open(r'./data/대출연체모델.sav','rb'))
    result = loaded_overdue_model.predict_proba(client_row)

    return result

def insurance_model_train():
    x_resampled=pd.read_csv(r'./data/보험 해지 여부 x.csv',encoding='euc-kr')
    y_resampled=pd.read_csv(r'./data/보험 해지 여부 y.csv',encoding='euc-kr')

    loaded_loan_model=pickle.load(open(r'./data/대출연체모델.csv'))

    train_x, test_x, train_y, test_y = train_test_split(x_resampled, y_resampled, random_state=1234, test_size=0.3)
    rf = RandomForestClassifier(random_state=1234,max_depth=6)
    rf.fit(train_x, train_y)
    return rf