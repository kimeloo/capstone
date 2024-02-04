import sqlite3
import pandas as pd
from pyparsing import dbl_quoted_string

# 파일 경로, 파일명 입력
pwd = '~/Documents/Coding/capstone/data/240201'
filename = 'shhs1-dataset-0.20.0.csv'

# CSV 파일을 데이터프레임으로 읽기
df = pd.read_csv(f'{pwd}/{filename}')

# SQLite 데이터베이스 연결
# db_path = f'{pwd}/shhs.db'
db_path = 'shhs.db'
conn = sqlite3.connect(db_path)

# 데이터프레임을 SQLite 데이터베이스에 쓰기
df.to_sql('shhs1', conn, if_exists='replace', index=False)

# 연결 종료
conn.close()
