# 약어 to 영문 해석 search
import requests
from bs4 import BeautifulSoup
def get_en_trans(word):
    url = f'https://sleepdata.org/datasets/mros/variables?search={word}'
    # 응답시간 초과 예외 처리
    retry_count = 0
    max_ret_count = 5
    while retry_count < max_ret_count:
        try:
            response = requests.get(url, timeout=10)
            break
        except requests.Timeout:
            print("** Request Timeout, Retrying... **")
            retry_count += 1
    else:
        print("** Request Timeout, Failed **")
        return ""
    ## 정상 응답시,
    if response.status_code == 200:
        html = BeautifulSoup(response.text, 'html.parser')
        # 검색 결과 없음 예외 처리 (.get_text()의 NoneType 처리 불가 오류)
        try:
            ## 'd-none d-sm-table-cell' class 찾아 text 반환 (검색 결과가 포함된 영역)
            en_trans = html.find('td', {'class': 'd-none d-sm-table-cell'}).get_text()
        except:
            print(f'{word} : ** Error **')
            return ""
        ## 검색 결과 text 자르기
        en_trans = en_trans.split('\n')[1]
        ## 약어, 검색 결과 콘솔에 출력
        print(f'{word} : {en_trans}')
        return en_trans
    
# 영문 해석 to 한글 translate
from googletrans import Translator
def get_ko_trans(english):
    translator = Translator()
    ## 영문 해석이 없는 경우 예외 처리 (''의 언어 감지 불가 오류)
    if english == '':
        return ''
    
    ## 언어 감지
    lang_detection = translator.detect(english)
    ### 감지된 언어가 한글인 경우, 번역 중단
    if lang_detection.lang == 'ko':
        return english
    ### 감지된 언어가 한글이 아닌 경우, 한글로 번역
    #### 응답시간 초과 예외 처리
    retry_count = 0
    max_ret_count = 5
    while retry_count < max_ret_count:
        try:
            translation = translator.translate(english, src=lang_detection.lang, dest='ko')
            korean = translation.text
            break
        except:
            retry_count += 1
    else:
        return ''
    print(f'{english} : {korean}')
    return korean

# 코드 실행 (CSV에 적용)
import pandas as pd
if __name__ == "__main__":
    pwd = '~/Documents/Coding/capstone/data/240122'
    df = pd.read_csv(f'{pwd}/word.csv')
    df['English'] = df['Word'].apply(get_en_trans)
    df['Korean'] = df['English'].apply(get_ko_trans)
    df.to_csv(f'{pwd}/word_translate.csv', index=False, encoding='utf-8')