import pandas as pd
import sqlite3

pwd = '/Users/kimeloo/Documents/Coding/capstone/data/240201'
db_name = 'shhs.db'

# db 연결하기
db_path = f'{pwd}/{db_name}'
conn = sqlite3.connect(db_path)

# 실험, 대조군 pptid 추출
## X_1 : 실험군(experimental group)
## X_0 : 대조군(control group)
sql_query = '''
            SELECT DISTINCT shhs1.*
            FROM (
                SELECT X_1.pptid as X_e, X_0.pptid as X_c
                FROM shhs1 as X_1
                INNER JOIN shhs1 as X_0
                ON  X_1.gender=X_0.gender
                AND X_1.height-X_0.height BETWEEN -3 AND 3
                AND X_1.weight-X_0.weight BETWEEN -1 AND 1
                AND X_1.age_s1-X_0.age_s1 BETWEEN -3 AND 3
                AND X_0.parrptdiab==0
                WHERE X_1.parrptdiab==1
                group by X_1.pptid) as X_pptid
            JOIN shhs1
            ON X_pptid.X_e = shhs1.pptid
            OR X_pptid.X_c = shhs1.pptid
            '''

train = pd.read_sql(sql_query, conn)
conn.close()

# print(train)
# print(train.info())
# train data 중, parrptdiab 값의 개수 확인
print(train['parrptdiab'].value_counts())

# csv로 저장
train.to_csv(f'{pwd}/diab_train_data.csv', index=False)