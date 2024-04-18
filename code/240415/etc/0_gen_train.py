import preprocess
import pandas as pd

if __name__=='__main__':
    pwd = '/users/kimeloo/Documents/Coding/capstone/data/240415'
    train = pd.read_csv(pwd+'/train.csv', low_memory=False)
    val = pd.read_csv(pwd+'/val.csv', low_memory=False)
    test = pd.read_csv(pwd+'/test.csv', low_memory=False)
    strategy = pd.read_csv(pwd+'/strategy.csv')
    
    pp = preprocess.preprocess(train=train, val=val, test=test)
    pp.column_replace(strategy)
    train, val, test = pp.return_data()
    train.to_csv(pwd+'/train_replaced.csv', index=False)