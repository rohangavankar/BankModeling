import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
jobs_dict = {"admin.":0, "unknown":1, "unemployed":2, "management":3 ,"housemaid":4,"entrepreneur":5, "student":6 ,
             "blue-collar":7,"self-employed":8,"retired":9,"technician":10,"services":11}
marital_dict = {"married":0,"divorced":1,"single":2}
education_dict = {"unknown":0,"secondary":1,"primary":2,"tertiary":3}
default_dict = {"yes":1,"no":0}
housing_dict = {"yes":1,"no":0}
loan_dict = {"yes":1,"no":0}
contact_dict = {"unknown":0,"telephone":1,"cellular":2}
month_dict={ "jan":0, "feb":1, "mar":2, "apr":3,"may":4, "jun":5, "jul":6, "aug":7, "sep":8, "oct":9, "nov":10, "dec":11}
poutcome_dict={"unknown":0,"other":1,"failure":2,"success":3}
y_dict={'no':0, 'yes':1}
def load_data(filename):
    df = pd.read_csv(filename, delimiter=';')
    return df
def convert_to_numeric(bank):
    bank['job_no']=bank.job.map(lambda x:jobs_dict[x])
    bank['marital_no']=bank.marital.map(lambda x:marital_dict[x])
    bank['education_no']=bank.education.map(lambda x:education_dict[x])
    bank['default_no']=bank.default.map(lambda x:default_dict[x])
    bank['housing_no']=bank.housing.map(lambda x:housing_dict[x])
    bank['loan_no']=bank.loan.map(lambda x:loan_dict[x])
    bank['contact_no']=bank.contact.map(lambda x:contact_dict[x])
    bank['month_no']=bank.month.map(lambda x:month_dict[x])
    bank['poutcome_no']=bank.poutcome.map(lambda x:poutcome_dict[x])
    bank['y_no']=bank.y.map(lambda x:y_dict[x])
    return bank

def do_etl(filename):
    df = load_data(filename)
    convert_to_numeric(df)
    df.drop_duplicates()
    return df





