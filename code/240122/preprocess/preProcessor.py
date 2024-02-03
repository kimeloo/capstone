def replace_column_char(df, column_name, replace_value):
    '''
    Replace character data in a column with numeric values.
    Raise KeyError when column_name is not in df

    df
       DataFrame (pandas.DataFrame)
    column_name
        Name of the column to replace character data
    replace_value
        Value to replace character data
        'mean' : Replace with the mean value of the column
        'median' : Replace with the median value of the column
        'mode' : Replace with the mode value of the column
        number : Replace with the entered number
    '''

    # 컬럼의 존재 여부 체크
    if column_name not in df.columns:
        raise KeyError(f'{column_name}은 DataFrame에 존재하지 않습니다.')

    # 컬럼을 number로 변환, character는 na로 대체
    column_to_num = pd.to_numeric(df[column_name], errors='coerce')

    # 채워넣을 값 구하기
    if replace_value == 'mean':
        replace_value = column_to_num.mean()
    elif replace_value == 'median':
        replace_value = column_to_num.median()
    elif replace_value == 'mode':
        replace_value = column_to_num.mode()
    
    # na값을 채워넣음
    df[column_name] = column_to_num.fillna(replace_value)
    return df

if __name__ == '__main__':
    import pandas as pd
    # 파일 경로, 파일명 입력
    pwd = '~/Documents/Coding/capstone/data/240122'
    filename = 'dataset_230122.csv'
    
    # DataFrame 생성
    df = pd.read_csv(f'{pwd}/{filename}')
    
    # 입력 데이터와 결과 분리
    X = df.drop('pd', axis=1)
    y = df['pd']

    # 기타사항0
    ## 첫번째 컬럼인 index 제거하기
    X = X.drop(X.columns[0], axis=1)

    # 기타사항1
    ## 재측정 값 -> 기존 측정값 덮어씌우기
    ## hwgt4=1이면, hwmeas3과 4는 hwmeas1과 2에 덮어씌움
    ## 이후 hwgt4, hwmeas3, hwmeas4 컬럼 삭제
    ### hwmeas1,2는 int, hwmeas3,4는 string이므로 형식 변환
    X['hwmeas3'] = pd.to_numeric(X['hwmeas3'], errors='coerce').fillna(0)
    X['hwmeas4'] = pd.to_numeric(X['hwmeas4'], errors='coerce').fillna(0)
    X.loc[(df['hwgt4'] == 1) & (X['hwmeas3'].notnull()), 'hwmeas1'] = X['hwmeas3']
    X.loc[(df['hwgt4'] == 1) & (X['hwmeas4'].notnull()), 'hwmeas2'] = X['hwmeas4']

    # 기타사항2
    ## 문자열로 저장된 시간 데이터(HH:MM:DD)를 str -> datetime -> int로 변환
    column_time1 = ['pqptmwak', 'poxuritm', 'poxwktm', 'postendp', 'poligoff']
    for column_name in column_time1:
        change_name = column_name + "_numeric"
        X[column_name] = pd.to_datetime(X[column_name].apply(lambda x: f'{x}:00' if len(x) == 5 else x), format='%H:%M:%S', errors='coerce').dt.time
        X[change_name] = [(t.hour*3600 + t.minute*60 + t.second) if not pd.isna(t) else pd.NaT for t in X[column_name]]
        X[column_name] = X[change_name]
        X = X.drop(columns=change_name)

    # 기타사항3
    ## 문자열로 저장된 시간 데이터를 str -> datetime -> int로 변환
    ## 24시 이전인 경우는 그대로, 24시 이후(0~12시)인 경우는 24시간을 더함 -> 취침시간의 왜곡 방지
    column_time2 = ['pqptmbed', 'poxbedtm', 'postlotp', 'postontp', 'poststtp']
    for column_name in column_time2:
        change_name = column_name + "_numeric"
        X[column_name] = pd.to_datetime(X[column_name].apply(lambda x: f'{x}:00' if len(x) == 5 else x), format='%H:%M:%S', errors='coerce').dt.time
        X[change_name] = [(t.hour*3600 + t.minute*60 + t.second + (24*3600 if t.hour<12 else 0)) if not pd.isna(t) else pd.NaT for t in X[column_name]]
        X[column_name] = X[change_name]
        X = X.drop(columns=change_name)

    # 기타사항4
    ## 문자열로 저장된 날짜 데이터를 str -> datetime -> int로 변환
    column_date = ['postdydt']
    for column_name in column_date:
        change_name = column_name + "_numeric"
        X[column_name] = X[column_name].apply(lambda t: pd.to_datetime(t, errors='coerce'))
        X[change_name] = X[column_name].apply(lambda t: (t - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s') if not pd.isna(t) else None)

        X[column_name] = X[change_name]
        X = X.drop(columns=change_name)


    # 필요없는 컬럼 제거
    ## nsrrid : ID이므로 제거
    ## hwgt4, hwmeas3, hwmeas4 : hwmeas1,2의 재측정 값이므로 위에서 처리 후 컬럼 제거
    ## hwdsht, hwdswt, hwdswtpc, hwdsbmi : 결측치 50% 이상으로 제거, 데이터 추가되는 경우 재고려 필요
    ## mhpark, mhparkt : pd 컬럼과 정확히 일치하므로 제거, 데이터 추가되는 경우 재고려 필요
    column_delete = ['nsrrid', 'hwgt4', 'hwmeas3', 'hwmeas4', 'hwdsht', 'hwdswt', 'hwdswtpc', 'hwdsbmi', 'mhpark', 'mhparkt']
    X = X.drop(columns=column_delete)

    # 문자/숫자 혼재 데이터
    ## 평균값으로 치환할 컬럼
    column_to_mean = ['pqpsnor', 'pqpoth', 'pqbptype', 'pqbploud','pqbppaus', 'pqbplegs', 'pqbpconf', 'pqbpoth', 'hwneck1', 'hwneck2', 'hwneck3', 'hwneck', 'fosocial', 'poxuritm', 'polsao2r', 'polsao2n', 'poxsao2r', 'poxsao2n', 'poligoff', 'potmst1p', 'potmst1', 'potmst2p', 'potmst2', 'potms34p', 'potmst34', 'potmremp', 'potmrem', 'poremlat','poai_all', 'poai_rem', 'poai_nre', 'pordi0pa', 'pordi2pa', 'pordi3pa', 'pordi4pa', 'pordi5pa', 'pooai4pa', 'pocai4pa', 'posao2re', 'posao2nr', 'popcstaa', 'popcstad', 'poavplma']
    for column_name in column_to_mean:
        replace_column_char(X, column_name, 'mean')
        
    ## 0으로 치환할 컬럼
    ### 'mhparkt' 컬럼 제거하였으므로 치환 생략
    column_to_zero = ['slsnore', 'sloftsno', 'slstopbr', 'slsbtims', 'slsa', 'slscap', 'slssurg', 'slslpdis', 'slinsomn', 'slrestlg', 'slperleg', 'slnarc', 'slsdoth', 'mhfalltm', 'mhfract', 'mhhead', 'mhsprain', 'mhbruise', 'mhother', 'mhnoinjr', 'mhdiabt', 'mhhthyt', 'mhrheut', 'mhlthyt', 'mhosteot', 'mhoat', 'mhprostt', 'mhlivert', 'mhrenalt', 'mhcobpdt', 'mhbronct', 'mhasthmt', 'mhallert', 'mhglaut', 'mhmit', 'mhangint', 'mhchft', 'cvblkat', 'cvtiat', 'mhstrkt', 'cvrhdt', 'mhbpt', 'cvcabg', 'cvapcora', 'cvaorane', 'cvbplegs', 'cvaplow', 'cvsurgbv', 'cvpace', 'cvvalve', 'cvchpain', 'cvcphill', 'cvcpwalk', 'cvcpdo', 'cvcprel', 'cvcprelt', 'cvlocsum', 'cvlocsl', 'cvloclc', 'cvlocla', 'cvlocot', 'cvlocdk', 'cvcp30m', 'cvcpdoc', 'cvcpdsay', 'cvlpstil', 'cvlphill', 'cvlpwalk', 'cvlpstst', 'cvlpcalf', 'cvlphosp', 'tusmkcgn', 'tucpiamt', 'slnapdy', 'slnaphr', 'poxinter', 'poxsnort', 'poxlegk', 'mhafib', 'mhafibs', 'mhhr', 'mhhrs', 'cvrose', 'cvrosegr', 'cvpvd', 'd1diabjh', 'd1cdiabjh', 'posllatp', 'poremli', 'polongap', 'poplmasl', 'poplmanr', 'poplmare', 'poplmcas', 'poplmcan', 'poplmcad', 'poplmcar']
    for column_name in column_to_zero:
        replace_column_char(X, column_name, 0)
    
    # 저장
    X.to_csv(f'{pwd}/preprocessed_data.csv', index=False, encoding='utf-8')