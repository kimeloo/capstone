import pandas as pd

def generator(pwd, file_original, file_train, file_val_nsrrid, file_test_nsrrid):
    # CSV 파일 불러오기
    original = pd.read_csv(f'{pwd}/{file_original}', low_memory=False)
    train = pd.read_csv(f'{pwd}/{file_train}')
    val_nsrrid = pd.read_csv(f'{pwd}/{file_val_nsrrid}', low_memory=False)
    test_nsrrid = pd.read_csv(f'{pwd}/{file_test_nsrrid}', low_memory=False)

    # val, test에서 nsrrid가 일치하는 행만 original에서 추출, train까지 추출
    train = original[original['nsrrid'].isin(train['nsrrid'])]
    val = original[original['nsrrid'].isin(val_nsrrid['nsrrid'])]
    test = original[original['nsrrid'].isin(test_nsrrid['nsrrid'])]

    return train, val, test

if __name__ == '__main__':
    # 파일 경로, 파일명 입력
    pwd = '~/Documents/Coding/capstone/data/240415'
    file_original = 'shhs1-dataset-0.20.0.csv'
    file_train = '04_htnderv_s1_train.csv'
    file_val_nsrrid = 'val_nsrrid_htnderv_s1.csv'
    file_test_nsrrid = 'test_nsrrid_htnderv_s1.csv'
    
    train, val, test = generator(pwd, file_original, file_train, file_val_nsrrid, file_test_nsrrid)
    print(train.shape, val.shape, test.shape)
    train.to_csv(f'{pwd}/train.csv', index=False)
    val.to_csv(f'{pwd}/val.csv', index=False)
    test.to_csv(f'{pwd}/test.csv', index=False)