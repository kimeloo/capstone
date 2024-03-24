from glob import glob
import pandas as pd

# 파일 경로, 파일명 입력
csv_pwd = '~/Documents/Coding/capstone/data/240314'
csv_file = '01_htnderv_s1.csv'
edf_pwd = '/Volumes/HDD1/Capstone/data/240314_SHHS1/SHHS dataset/shhs/polysomnography/edfs'
edf_file = '*.edf'

# edf 파일 리스트 (하위폴더 포함)
# edf_list = glob(edf_pwd + '/**/' + edf_file, recursive=True)
edf_list = glob(edf_pwd + '/shhs1/' + edf_file)
edf_list.extend(glob(edf_pwd + '/mod_shh1/' + edf_file))

# CSV 파일 불러오기
df = pd.read_csv(f'{csv_pwd}/{csv_file}', low_memory=False)
print(df.shape)

# edf 파일명에서 nsrrid 추출
edf_id_list = [int(edf.split('/')[-1].split('.')[-2].split('-')[-1]) for edf in edf_list]

# nsrrid가 edf 파일에 있는지 확인
df = df[df['nsrrid'].isin(edf_id_list)]
print(df.shape)

# 전체 df를 파일로 저장
df.to_csv(f'{csv_pwd}/01_2_htnderv_s1.csv', index=False)