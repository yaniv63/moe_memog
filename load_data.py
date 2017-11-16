from scipy.io import loadmat
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

def load_medical_data(data_view='Dense', set_type ='Both'):
    raw_data = loadmat('data/%s.mat' % data_view)
    dataCC = raw_data.get('featuresCC')
    dataMLO = raw_data.get('featuresMLO')
    target = raw_data.get('Labels').transpose()[0]
    dataCC, dataMLO, target = shuffle(dataCC,dataMLO,target,random_state=0)
    if set_type=='CC':
        data = {'CC': dataCC}
    elif set_type=='MLO':
        data = {'MLO' : dataMLO}
    else:
        data = {'CC': dataCC,'MLO': dataMLO }
    return data,target

def split_data(data,target,is_multi_expert,kfold):
    if is_multi_expert:
        data_split  = list(kfold.split(data['MLO'],target))
    else:
        data_split = list(kfold.split(data,target))

    return data_split

def get_fold_samples(data,target,split_indexes,is_multi_expert,fold,is_train):

    set = 0 if is_train else 1
    if is_multi_expert:
        CC_samples = data['CC'][split_indexes[fold][set]]
        MLO_samples = data['MLO'][split_indexes[fold][set]]
        data_samples = [CC_samples,MLO_samples]
    else:
        data_samples = data[split_indexes[fold][set]]
    target_samples =target[split_indexes[fold][set]]
    return data_samples,target_samples

def scale_data_normlize(data):
    scaler = StandardScaler(with_std=False)
    for key,d in data.items():
        d = scaler.fit_transform(d)
        data[key] =d
    return data

if __name__ == '__main__':
    d,l = load_medical_data('Fatty')