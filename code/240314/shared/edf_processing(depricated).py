import numpy as np
from pyedflib import highlevel

#최종 dataset(subjects)열람 후 바로 datset 형성, 아직 sampling method 적용 필요
def dataset(subjects, database_name = 'D:\Desktop\data\shhs1\polysomnography\edfs\shhs1\shhs1-'): # 미완성 주파수 sampling 거치지 않음, 최종본
    signals = []
    for idx, subject in enumerate(subjects):
        database = f'{database_name}{subject}.edf'
        signal,_ = select_dataset(database).dataset()
        signals.append(signal)
        print(f'reading {idx}', end=' , ')
    dataset = np.vstack(signal for signal in signals) #error 배열 크기 안맞음
    return dataset

class select_dataset():
    '''
    date : 2024.03.15
    author : somin oh
    
    초기 설정값 
    target = ['ECG']
    delay = 15분
    
    method인 dataset으로 해당 target의 signal을 30초 간격으로 만들어주며,  반환값으로 signal 제공한다.
    '''

    target = ['ECG']
    delay = 15*2

    def __init__(self, file_path):
        self.signals, self.signal_headers, self.header = highlevel.read_edf(file_path)
        self.sample_rate = [signal_header['sample_rate'] for signal_header in self.signal_headers]

    def get_index(self):
        index = []
        for idx, signal in enumerate([signals['label'] for signals in self.signal_headers]):
            if signal in select_dataset.target:
                index.append(idx)
        return index

    def dataset(self):
        signals = np.array(self.signals)
        sample_rate = np.array(self.sample_rate)
        index = self.get_index()

        signals = signals[index][0]
        sample_rate = sample_rate[index][0]

        signal = signals[::sample_rate][::30][select_dataset.delay : -select_dataset.delay]


        return signal, sample_rate

