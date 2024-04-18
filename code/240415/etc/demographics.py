import pandas as pd

def get_mean_std(df, column):
    return df[column].mean(), df[column].std()


if __name__=='__main__':
    pwd = '/users/kimeloo/documents/coding/capstone/data/240415'
    train = pd.read_csv(pwd+'/train_replaced.csv')

    train_htn = train[train['htnderv_s1']==1]
    train_ctl = train[train['htnderv_s1']==0]

    all = dict()
    ctl = dict()
    htn = dict()
    for col in ['age_s1', 'bmi_s1', 'gender']:
        all[col] = get_mean_std(train, col)
        ctl[col] = get_mean_std(train_ctl, col)
        htn[col] = get_mean_std(train_htn, col)

    print(*all.items(), sep='\n')
    print(*htn.items(), sep='\n')
    print(*ctl.items(), sep='\n')

    for df in [train, train_ctl, train_htn]:
        print(df['gender'].value_counts())
        print(df['gender'].value_counts(normalize=True))