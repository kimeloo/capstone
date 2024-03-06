import sqlite3
import pandas as pd
from pyparsing import dbl_quoted_string

# 파일 경로, 파일명 입력
pwd = '~/Documents/Coding/capstone/data/240201'
filename = 'shhs1-dataset-0.20.0.csv'

# CSV 파일을 데이터프레임으로 읽기
df = pd.read_csv(f'{pwd}/{filename}')

# outliercheck1, outliercheck2 컬럼의 값이 1이면 해당 행 제거
df = df[df['outliercheck1'] != 1]
df = df[df['outliercheck2'] != 1]

# SQLite 데이터베이스 연결 : 경로 지정하면 저장 안되는 버그로 인해 pwd 사용 불가
# db_path = f'{pwd}/shhs.db'
db_path = 'shhs.db'
conn = sqlite3.connect(db_path)

# 데이터프레임을 SQLite 데이터베이스에 쓰기
df.to_sql('shhs1', conn, if_exists='replace', index=False)

# 연결 종료
conn.close()
