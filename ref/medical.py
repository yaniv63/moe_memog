import scipy
import numpy as np

def load_medical_data(data_type='Fatty',test_ratio = 0.15,set_type = 'Both'):
    raw_data = scipy.io.loadmat('../data/%s.mat'%data_type)
    dataCC = raw_data.get('featuresCC')
    dataMLO = raw_data.get('featuresMLO')
    if set_type=='CC':
        data = dataCC
    elif set_type=='MLO':
        data = dataMLO
    else:
        data = np.concatenate((dataCC, dataMLO), axis=1)
    target = raw_data.get('Labels').transpose()[0]

    train_set_end_example = np.int((1-test_ratio) * data.shape[0])
    train_set_x = data[:train_set_end_example, :]
    train_set_y = target[:train_set_end_example]
    test_set_x = data[train_set_end_example:, :]
    test_set_y = target[train_set_end_example:]

    rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    return rval
