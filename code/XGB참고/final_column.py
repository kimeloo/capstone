# 그나마 잘나오는 것들

# Sleep Monitoring
columns = ['timest2', 'times34p', 'rdirem0p', 'stonsetp', 'remlaiip', 'timebedp', 'stloutp', 'avsao2nh', 'timest2p', 'hstg342p'] #sleep_monitoring, 542 columns (그나마 잘 나옴)

# sleep_architecture
columns = ['times34p', 'timest2', 'timebedp', 'rdirem0p', 'stonsetp', 'remlaiip', 'stloutp', 'timest2p', 'hstg342p', 'avsao2nh'] #sleep_architecture, 56 columns (그나마 잘 나옴)

# heart_rate
# columns = ['dmnbnoh, davbroh, amnbnoh, aavbroh, havbnoh, aavbnoh, hmxbnoh, havbroh, davbnbh, amnbroh'] #heart_rate, 46 columns (잘 안나옴)

# 예전 코드
columns = ['oai0p', 'oai4p', 'oai4pa', 'cai0p', 'cai4p', 'cai4pa']
for i in ['0', '2', '3', '4', '5']:
    for j in ['p', 'pa', 'ps', 'pns']:
        columns.append(f'rdi{i}{j}')
    columns.append(f'rdirem{i}p')
    columns.append(f'rdinr{i}p')
