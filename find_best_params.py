import os

def find_max_w_by_metric(path,metric):
    '''
    assume file name is a_avalue-x-b_bvalue-x-c_cvalue....extesion
    -x- sign used for seperation
    '''
    files = {}
    max_v = 0
    for filename in os.listdir(path):
        files[filename] = {}
        basename, extension = os.path.splitext(filename)
        if extension =='.log':
            continue
        stats = basename.split('-x-')
        for stat in stats:
            name, value = stat.split('_')
            files[filename][name] = float(value)
    for filename, d in files.items():
        if filename =='run.log':
            continue
        if d[metric] < max_v:
            min_file = filename
            max_v = d[metric]
    return min_file