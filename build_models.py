import numpy as np
np.random.seed(42)
import os
from collections import defaultdict
from keras.optimizers import SGD
from load_data import load_medical_data,scale_data_normlize
# from exception_handler import *
from multi_predictors_combined import expert_model,n_parameters_combined_model,n_experts_combined_model,simple_model,n_experts_combined_model_gate_parameters
from train_model import train_phase
from utils.funcs import merge_dicts
from utils.logging_tools import get_logger
from utils.paths import *
from sklearn.model_selection import KFold
from load_data import split_data,get_fold_samples
from evaluate_model import eval_model
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
import itertools

run_dir = get_run_dir()

## parameters
from utils.params import callbacks_params,params

keys = list(params)
for params_index,values in enumerate(itertools.product(*map(params.get, keys))):
    current_params =  dict(zip(keys, values))
    np.random.seed(42)
    params_dir = run_dir + str(params_index) +'/'
    makedirs(params_dir)
    logger = get_logger(params_dir)
    logger.info("{}".format(current_params))
    logger.info("index {}".format(params_index))

    scale_data = False
    fit_params_dict = {'batch_size':16,'epochs':current_params['epoch_num'],'validation_split':0.0,'verbose':1}
    kfold_params_dict = {'n_splits':10, 'shuffle':False, 'random_state':42}

    logger.info("scale data {}".format(scale_data))
    logger.info("{}".format(fit_params_dict))
    logger.info("{}".format(callbacks_params))

    ## data
    data_view = ['CC','MLO']
    data_type = ['Dense','Fatty']
    type_v = 1
    data,target = load_medical_data(data_type[type_v])
    if scale_data:
        data = scale_data_normlize(data)
    feature_number = data[data_view[0]].shape[1]
    logger.info("data type {}".format(data_type[type_v]))

    kf = KFold(**kfold_params_dict)
    from utils.metrices import fmeasure
    ## train
    experts = {}
    experts_split_indexes =split_data(data['CC'],is_multi_expert=False,kfold=kf)
    print "train experts"
    for i, view in enumerate(data_view):
        from utils.metrices import fmeasure
        expert = expert_model(input_shape=(feature_number,), index=view,params=current_params)
        #optimizer = SGD(lr=0.01, nesterov=True,momentum=0.9)
        expert.compile(optimizer=current_params['optimizer'], loss='binary_crossentropy', metrics=['accuracy', fmeasure])
        train_phase(expert, data[view], target, view, fit_params_dict, experts_split_indexes,False,params_dir)
        experts[view] = expert

    s_model = simple_model(params=current_params)
    s_data = np.concatenate((data['CC'], data['MLO']), axis=1)
    #optimizer = SGD(lr=0.01, nesterov=True,momentum=0.9)
    s_model.compile(optimizer=current_params['optimizer'], loss='binary_crossentropy', metrics=['accuracy', fmeasure])
    train_phase(s_model, s_data, target, 'simple_model', fit_params_dict, experts_split_indexes, False,params_dir)

    print "train combined models"
    combined_split_indexes =split_data(data,is_multi_expert=True,kfold=kf)

    moe_vectors = n_experts_combined_model(input_shape=(feature_number,), n=2,params =current_params)
    # moe_parameters = n_experts_combined_model_gate_parameters(input_shape=(feature_number,), n=2)
    # parameters_model = n_parameters_combined_model(input_shape=(feature_number,),n=2)
    combined_models = {'MOE': moe_vectors}#, 'parameters_model': parameters_model,'moe_parameters':moe_parameters}
    for model_name,model in combined_models.items():
        optimizer = SGD(lr=0.1, decay=1e-5, nesterov=True)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', fmeasure])
        train_phase(model,data,target,model_name,fit_params_dict,combined_split_indexes,True,params_dir)

    #predictors = merge_dicts(experts, combined_models)

    ## evaluate
    print "evaluate models"
    metrics = [accuracy_score,f1_score,confusion_matrix]
    scores = defaultdict(dict)
    def check_model(model,model_name,data,target,metrics,indexes,is_multi_expert):
        results_acc = []
        results_f1 = []
        for i in range(kfold_params_dict['n_splits']):
            x_test, y_test = get_fold_samples(data, target, indexes, is_multi_expert=is_multi_expert,
                                              fold=i, is_train=False)
            model.load_weights(params_dir + "model_{}_fold_{}.h5".format(model_name, i))
            res = eval_model(model, x_test, y_test, metrics)
            results_acc.append(res[0])
            results_f1.append(res[1])
        acc_mean = np.mean(results_acc)
        acc_std = np.std(results_acc)
        f1_mean = np.mean(results_f1)
        f1_std = np.std(results_f1)
        logger.info("{} acc    {:.5f} (+/- {:.5f})".format(model_name,acc_mean, acc_std))
        logger.info("{} f1    {:.5f} (+/- {:.5f})".format(model_name,f1_mean, f1_std))
        return (acc_mean,f1_mean)

    for model_name,model in experts.items():
        a,f1 = check_model(model,model_name,data[model_name],target,metrics,experts_split_indexes,False)
        scores[model_name] = {'acc':a,'f1':f1}

    a, f1 = check_model(s_model,'simple_model',s_data,target,metrics,experts_split_indexes,False)
    scores['SIMPLE'] = {'acc': a, 'f1': f1}

    for model_name,model in combined_models.items():
        a, f1 = check_model(model,model_name,data,target,metrics,combined_split_indexes,True)
        scores[model_name] = {'acc':a,'f1':f1}

    folder_name=str(params_index)
    for k, v in scores.items():
        res = "_{}_acc_{:.4f}_f1_{:.4f}-x-".format(k,v['acc'],v['f1'])
        folder_name += res
    os.rename(params_dir,run_dir +folder_name)

