import numpy as np
np.random.seed(42)
import os
from collections import defaultdict
from keras.optimizers import SGD
from load_data import load_medical_data,scale_data_normlize
# from exception_handler import *
from multi_predictors_combined import expert_model,simple_model,n_experts_combined_model_b,multilabel_model
from train_model import train_phase
from utils.funcs import merge_dicts
from utils.logging_tools import get_logger
from utils.paths import *
from sklearn.model_selection import KFold,StratifiedKFold
from load_data import split_data,get_fold_samples
from evaluate_model import eval_model
from sklearn.metrics import accuracy_score,f1_score
import itertools
from pympler.tracker import SummaryTracker
import gc
#gc.set_debug(gc.DEBUG_LEAK)
tracker = SummaryTracker()

run_dir = get_run_dir()

# state = np.random.get_state()
import pickle

# with open("state.npy", "wb") as f:
#     state = pickle.dump(state,f)
with open("state.npy", "rb") as f:
    state = pickle.load(f)

## parameters
from utils.params import callbacks_params,params
keys = list(params)
for params_index,values in enumerate(itertools.product(*map(params.get, keys))):
    #tracker.print_diff()
    current_params =  dict(zip(keys, values))
    # current_params = {'optimizer': 'adam', 'nn_gate': 3, 'w_init': 'glorot_normal', 'nn_layer2': 15, 'nn_layer1': 10,
    #               'epoch_num': 500, 'dropout2': 0, 'dropout1': 0.2}
    # current_params = {  'nn_layer1' : 12, 'nn_layer2' : 12,
    #               'dropout1' : 0.5,'dropout2': 0.5,'w_init': 'glorot_uniform','epoch_num':10,'optimizer' : 'sgd','nn_gate': 3 }

    check_seed = 0
    np.random.seed(check_seed)
    params_dir = run_dir + str(params_index) +'/'
    makedirs(params_dir)
    logger = get_logger(params_dir)
    logger.info("{}".format(current_params))
    logger.info("index {}".format(params_index))

    scale_data = False
    fit_params_dict = {'batch_size':16,'epochs':current_params['epoch_num'],'validation_split':0.0,'verbose':0}
    kfold_params_dict = {'n_splits':10, 'shuffle':False, 'random_state':42}

    logger.info("scale data {}".format(scale_data))
    logger.info("{}".format(fit_params_dict))
    logger.info("{}".format(callbacks_params))

    ## data
    data_view = ['CC','MLO']
    data_type = ['Dense','Fatty']
    type_v = 0
    data,target = load_medical_data(data_type[type_v])
    if scale_data:
        data = scale_data_normlize(data)
    feature_number = data[data_view[0]].shape[1]
    logger.info("data type {}".format(data_type[type_v]))

    kf = KFold(**kfold_params_dict)

    ## train
    experts = {}
    experts_split_indexes =split_data(data['CC'],target,is_multi_expert=False,kfold=kf)
    print "train experts"
    for i, view in enumerate(data_view):
        np.random.seed(check_seed)
        expert = expert_model(input_shape=(feature_number,), index=view,params=current_params)
        expert.compile(optimizer=current_params['optimizer'], loss='binary_crossentropy', metrics=['accuracy'])#, fmeasure])
        train_phase(expert, data[view], target, view, fit_params_dict, experts_split_indexes,False,params_dir)
        experts[view] = expert

    np.random.seed(check_seed)
    s_model = simple_model(params=current_params)
    s_data = np.concatenate((data['CC'], data['MLO']), axis=1)
    s_model.compile(optimizer=current_params['optimizer'], loss='binary_crossentropy', metrics=['accuracy'])#, fmeasure])
    train_phase(s_model, s_data, target, 'simple_model', fit_params_dict, experts_split_indexes, False,params_dir)


    print "train combined models"
    combined_split_indexes =split_data(data,target,is_multi_expert=True,kfold=kf)
    np.random.seed(check_seed)
    moe_multilabel = multilabel_model(input_shape=(feature_number,), n=2,params =current_params)
    moe_multilabel.compile(optimizer=current_params['optimizer'],loss={'main_output': 'binary_crossentropy',
                        'exp0_output': 'binary_crossentropy',
                        'exp1_output': 'binary_crossentropy',
                        },
                  loss_weights=[1., 0.2, 0.2],
                  metrics=['accuracy'])

    moe_vectors = n_experts_combined_model_b(input_shape=(feature_number,), n=2,params =current_params)
    moe_vectors.compile(optimizer=current_params['optimizer'], loss='binary_crossentropy',
                  metrics=['accuracy'])  # , fmeasure])

    combined_models = {'MOE': moe_vectors,'MULTILABEL':moe_multilabel}
    for model_name,model in combined_models.items():
        train_phase(model,data,target,model_name,fit_params_dict,combined_split_indexes,True,params_dir)


    ## evaluate
    print "evaluate models"
    metrics = [accuracy_score,f1_score]
    scores = defaultdict(dict)
    def check_model(model,model_name,data,target,metrics,indexes,is_multi_expert):
        results_acc = []
        results_f1 = []
        predictions = []
        for i in range(kfold_params_dict['n_splits']):
            x_test, y_test = get_fold_samples(data, target, indexes, is_multi_expert=is_multi_expert,
                                              fold=i, is_train=False)
            model.load_weights(params_dir + "model_{}_fold_{}.h5".format(model_name, i))
            res,pred = eval_model(model, x_test, y_test, metrics)
            results_acc.append(res[0])
            results_f1.append(res[1])
            predictions.append(pred)
        acc_mean = np.mean(results_acc)
        acc_std = np.std(results_acc)
        f1_mean = np.mean(results_f1)
        f1_std = np.std(results_f1)
        logger.info("{} acc    {:.5f} (+/- {:.5f})".format(model_name,acc_mean, acc_std))
        logger.info("{} f1    {:.5f} (+/- {:.5f})".format(model_name,f1_mean, f1_std))
        return (acc_mean,f1_mean,predictions)

    for model_name,model in experts.items():
        a,f1,pred = check_model(model,model_name,data[model_name],target,metrics,experts_split_indexes,False)
        scores[model_name] = {'acc':a,'f1':f1,'pred':pred}

    #res = avg_predictions([scores['CC']['pred'],scores['MLO']['pred']],target,metrics)

    for model_name,model in combined_models.items():
        a, f1,pred = check_model(model,model_name,data,target,metrics,combined_split_indexes,True)
        scores[model_name] = {'acc':a,'f1':f1,'pred':pred}

    a, f1,pred = check_model(s_model, 'simple_model', s_data, target, metrics, experts_split_indexes, False)
    scores['SIMPLE'] = {'acc': a, 'f1': f1,'pred':pred}


    folder_name=str(params_index)
    for k, v in scores.items():
        if k == 'pred': continue
        res = "_{}_acc_{:.4f}_f1_{:.4f}-x-".format(k,v['acc'],v['f1'])
        folder_name += res
    os.rename(params_dir,run_dir +folder_name)
    del experts
    del moe_vectors
    gc.collect()




