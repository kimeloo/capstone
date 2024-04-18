import pandas as pd

if __name__=='__main__':
    pwd = '/users/kimeloo/Documents/Coding/capstone/data/240415'
    train = pd.read_csv(pwd+'/train_replaced.csv')
    dictionary = pd.read_csv(pwd+'/shhs-data-dictionary-0.20.0-variables.csv')
    dictionary = dictionary[['id', 'folder', 'Korean', 'display_name']]

    # train의 column명과 일치하는 id를 찾아 dictionary 재생성
    columns = train.columns
    dictionary = dictionary[dictionary['id'].isin(columns)]

    # dictionary에 없는 column은 추가
    for column in columns:
        if column not in dictionary['id'].values:
            new = pd.DataFrame({'id': column, 'folder': '', 'Korean': '', 'display_name': ''}, index=[0])
            dictionary = pd.concat([dictionary, new], ignore_index=True)

    # 한글 인코딩
    dictionary.to_csv(pwd+'/dictionary.csv', index=False, encoding='EUC-KR')