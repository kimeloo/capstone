from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 주성분의 개수 입력
num = int(input("주성분의 개수를 입력하세요. "))

# 파일 경로, 파일명 입력
pwd = '~/Documents/Coding/capstone/data/240122'
filename = 'preprocessed_data.csv'

# DataFrame 생성
df = pd.read_csv(f'{pwd}/{filename}')

# 데이터 표준화 (중요: PCA를 적용하기 전에 특성들을 표준화해주는 것이 좋습니다)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# PCA 모델 생성 및 학습
pca = PCA(n_components=num)  # 주성분의 개수를 지정
pca_result = pca.fit_transform(scaled_data)

print(pca_result.shape)

# 주성분으로 이루어진 데이터프레임 생성 (컬럼명 자동 생성)
pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])

# 주성분을 이용한 분석 결과 출력
print(pca_df)

pca_df.to_csv(f'{pwd}/PCA_{num}_data.csv', index=False, encoding='utf-8')