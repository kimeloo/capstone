import numpy as np
import pandas as pd
from pyedflib import highlevel
from glob import glob

# 0. 설정값 지정
target = 'ECG'          # 추출할 데이터 종류
interval_sec = 30       # 추출할 데이터 간격(초)
delay_min = 15          # 시작점과 끝점에서 제외할 데이터(분)

# 1. 파일 경로 지정
# path_dat = '/Volumes/HDD1/Capstone/data/240314_SHHS1'           # 데이터 저장된 폴더
path_dat = '/Volumes/Samsung_T5/Capstone'                       # 데이터 저장된 폴더
path_edf = 'SHHS dataset/shhs/polysomnography/edfs/shhs1'       # 위 폴더 내, edf 파일 저장된 폴더
path_save = '~/Documents/Coding/capstone/data/240314/polysomnography/edfs/shhs1'     # 추출한 csv파일 저장할 폴더

#####################################################
# 2. edf 파일 리스트 불러오기
files = glob(f'{path_dat}/{path_edf}/*.edf')
print(f'files : {len(files)}')
result = np.array([]).reshape(1, 0)

for file in files:
    # 3. edf 불러오기
    signals, signal_headers, header = highlevel.read_edf(file)

    # 4. 사용할 데이터 선택
    for idx, label in enumerate([info['label'] for info in signal_headers]):
        if label == target:
            target_idx = idx
            break

    # 5. sample_rate 확인
    sample_rate = int(signal_headers[target_idx]['sample_rate'])

    # 6. 데이터 추출
    data = signals[target_idx]

    # 7. 데이터 정제
    interval_cal = sample_rate * interval_sec
    delay_cal = int(delay_min * (60 / interval_sec))
    data_refined = data[::interval_cal][delay_cal:-delay_cal]
    # 7.1 nsrrid 첫 열에 추가
    nsrrid = file.split("/")[-1].split("-")[1].split(".")[0]
    data_refined = np.insert(data_refined, 0, nsrrid, axis=0)     # 1번 행에 nsrrid를 끼워넣음

    # 8. 콘솔 확인 및 result ndarray에 저장
    print(f'loading {file.split("/")[-1]} : {data_refined.shape[0]} samples')
    max_cols = result.shape[1] if result.shape[1] > data_refined.shape[0] else data_refined.shape[0]
    result_temp = np.full((result.shape[0]+1, max_cols), np.nan)
    result_temp[:result.shape[0], :result.shape[1]] = result
    result_temp[-1, :data_refined.shape[0]] = data_refined
    result = result_temp.copy()
    print(f'read : {result.shape[0]-1} / {len(files)}')

# 9. csv 저장
df = pd.DataFrame(result)
df.drop(0, axis=0, inplace=True)        # 1번째 행 삭제(초기 result 변수에 저장된 빈 행)
df.to_csv(f'{path_save}/{target}.csv', index=False, header=False)
print(f'saved : {path_save}/{target}.csv')