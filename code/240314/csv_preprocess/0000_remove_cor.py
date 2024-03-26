remove_list = ['asa1', 'diuret1', 'ccb1']

# 아래는 rp_s1와 모두 높은 연관을 보이는 컬럼들
remove_list.extend(['phctdn25', 'phacls25', 'limit25', 'exefrt25', 'probsa25', 'painin25', 'pep25', 'energ25', 'tired25', 'hlthy25', 'pf_s1', 'vt_s1', 'sf_s1', 'mh_s1', 'pcs_s1'])

columns = ['ccb1','ace1','beta1','diuret1','age_s1','dig1','waso','vaso1','aai','hctz1','chol','sipeffp','rcrdtime','mcs_s1','mi2slp02','stloutp','ntg1','timebedp','stonsetp','trig','timest2','twuweh02','avsao2nh','hremt2p','avdnop4','ahremop','slplatp','timest1p']
for c in remove_list:
    if c in columns:
        print(c)