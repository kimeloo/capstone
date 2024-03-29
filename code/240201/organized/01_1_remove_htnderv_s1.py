import pandas as pd

# 파일 경로, 파일명 입력
pwd = '~/Documents/Coding/capstone/data/240201'
filename = 'shhs1-dataset-0.20.0.csv'
# CSV 파일 불러오기
df = pd.read_csv(f'{pwd}/{filename}', low_memory=False)
print(df.shape)

# 1. 전체 데이터에서 필요없는 컬럼/행 제거
## outliercheck1, outliercheck2 컬럼의 값이 1이면 해당 행 제거
df = df[df['outliercheck1'] != 1]
df = df[df['outliercheck2'] != 1]
## 결과 컬럼이 null이면 해당 행 제거
df = df.dropna(subset=['htnderv_s1'])     # 고혈압
## 필요 없는 컬럼 제거 (nsrrid 제외)
df = df.drop(columns=['pptid', 'scorer_id', 'monitor_id', 'headbox_id'])
df = df.drop(columns=['pptidr'])
df = df[df.columns.drop(list(df.filter(regex='^raw')))]
df = df[df.columns.drop(list(df.filter(regex='^shhs1')))]
df = df.drop(columns=['ohga1', 'insuln1'])
df = df.drop(columns=['syst120', 'syst220', 'syst320', 'dias120', 'dias220', 'dias320'])
df = df.drop(columns=['age_category_s1'])
df = df.drop(columns=['gh_s1', 'genhth25', 'exclnt25'])
df = df.drop(columns=['ecgdate', 'stdydtqa', 'staging1', 'staging2', 'staging3', 'staging4', 'staging5', 'staging7', 'staging8', 'restan1', 'restan2', 'restan3', 'restan4', 'restan5', 'overall_shhs1', 'hrov150', 'hrund30', 'oxyund70', 'ahiov50', 'educat', 'date02', 'date10', 'date25', 'visitnumber', 'hrdur', 'airdur', 'chestdur', 'abdodur', 'eeg1dur', 'eeg2dur', 'eogrdur', 'eogldur', 'chindur', 'oximdur', 'posdur', 'hrqual', 'airqual', 'chstqual', 'abdoqual', 'eeg1qual', 'eeg2qual', 'eogrqual', 'eoglqual', 'chinqual', 'oximqual', 'posqual', 'lightoff', 'psg_month'])
df = df.drop(columns=['srhype', 'htnmed1', 'systbp', 'armbp', 'ankbp', 'diasbp'])   # 고혈압 관련 컬럼
## 시간 데이터 정수로 변환
column_name = 'rcrdtime'
change_name = column_name + "_numeric"
df[column_name] = pd.to_datetime(df[column_name], format='%H:%M:%S', errors='coerce').dt.time
df[change_name] = [(t.hour*3600 + t.minute*60 + t.second) if not pd.isna(t) else pd.NaT for t in df[column_name]]
df[column_name] = df[change_name]
df = df.drop(columns=change_name)
print(df.shape)

# 전체 df를 파일로 저장
df.to_csv(f'{pwd}/01_htnderv_s1.csv', index=False)