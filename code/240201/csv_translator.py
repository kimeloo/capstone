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
    pwd = '~/Documents/Coding/capstone/data/240201'
    df1 = pd.read_csv(f'{pwd}/columns.csv')
    df = pd.read_csv(f'{pwd}/shhs-data-dictionary-0.20.0-variables.csv')

    # df1의 id를 기준으로 df 재정렬
    df = df.set_index('id').loc[df1['id']].reset_index()

    df['Korean'] = df['display_name'].apply(get_ko_trans)
    df.to_csv(f'{pwd}/word_translate.csv', index=False, encoding='utf-8')