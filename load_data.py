from scipy.io import loadmat
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler



def _load_view(view):
    raw_data = loadmat('data/%s.mat' % view)
    dataCC = raw_data.get('featuresCC')
    dataMLO = raw_data.get('featuresMLO')
    target = raw_data.get('Labels').transpose()[0]
    return dataCC,dataMLO,target

def _load_all_views():
    D_cc,D_MLO,D_t =_load_view('Dense')
    F_cc,F_MLO,F_t = _load_view('Fatty')
    cc = np.concatenate((D_cc,F_cc))
    mlo = np.concatenate((D_MLO,F_MLO))
    target = np.concatenate((D_t,F_t))
    return cc,mlo,target


def load_medical_data(data_view='Dense', set_type ='Both'):
    dataCC, dataMLO, target = _load_all_views() if data_view=='all' else _load_view(data_view)
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