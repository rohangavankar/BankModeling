import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import etl

def box_plot(df):
    data=df['balance']
    data_age=df['age']
    fig1, ax=plt.subplots(1,2)
    ax[0].set_title('Balance')
    ax[0].boxplot(data)
    ax[1].set_title('Age')
    ax[1].boxplot(data_age)
    plt.show()

def bar_chart(df):
    df_s = df[['job_no','job']]
    dfg = df_s.groupby('job_no').count().reset_index()
    job_counts = list(dfg.job.values)
    data = list(job_counts)
    x = np.arange(len(etl.jobs_dict.keys()))
    x = np.arange(len(data))
    fig, ax = plt.subplots()
    plt.bar(x, data)
    plt.xticks(x, etl.jobs_dict.keys())
    plt.show()

def main():
    df = do_etl("/Users/rohangavankar/PycharmProjects/test_app/bank/bank.csv")
    print(df.balance.describe())
    print(df.age.describe())
    df_temp = df[['balance', 'age', 'job_no']]
    print(df_temp.describe())
    box_plot(df)


main()