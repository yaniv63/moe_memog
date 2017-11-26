import  numpy as np
from models.memo_models import BaseLineModel,ExpertModel,MOE,MultilabelMOE
from load_data import  load_medical_data
from sklearn.model_selection import KFold #TODO check if startified
from utils.params import callbacks_params
from utils.paths import get_run_dir
import copy
from eval_model import EvalModel
from collections import defaultdict
from sklearn.metrics import accuracy_score,f1_score
from utils.plotting_tools import PlotHandler
from utils.logging_tools import get_logger
run_dir = get_run_dir()
logger = get_logger(run_dir)
## data
data_view = ['CC','MLO']

data_type = ['Dense','Fatty']
type_v = 1
data,target = load_medical_data(data_type[type_v])
# target = target[:20]
# data = {k:v[:20] for k,v in data.items()}
concat_data = np.concatenate((data['CC'], data['MLO']), axis=1)

## params
nb_splits = 10
kfold_params_dict = {'n_splits': nb_splits, 'shuffle': False, 'random_state': 42}
fit_params_dict = {'batch_size': 16, 'epochs': 500, 'verbose': 0,
                   'optimizer' : 'adam', 'loss' : 'binary_crossentropy', 'metrics' : ['accuracy'],'shuffle': False,'validation_split':0.1}
expert_params = {  'nn_layer1' : 24, 'nn_layer2' : 12,
                  'dropout1' : 0.5,'dropout2': 0.5,'w_init': 'glorot_uniform' }
baseline_params = {'nn_layer1' : 24, 'nn_layer2' : 24,'nn_layer3' : 3,'nn_layer4' : 3,
                  'dropout1' : 0.5,'dropout2': 0.5,'dropout3': 0.5,'w_init': 'glorot_uniform'}
moe_params = { 'nn_layer1' : 22, 'nn_layer2' : 12,
                  'dropout1' : 0.5,'dropout2': 0.5,'w_init': 'glorot_uniform','nn_gate1' : 20,'nn_gate2':20}
multi_moe_params =copy.copy(fit_params_dict)
multi_moe_params['loss_weights'] = [1,1,1]

logger.info("{}".format(fit_params_dict))
logger.info("{}".format(callbacks_params))
logger.info("expert params {}".format(expert_params))
logger.info("baseline_params {}".format(baseline_params))
logger.info("moe params {}".format(moe_params))
logger.info("multilabel params {}".format(multi_moe_params))


expert_num =2
## helping objects
metrics = {'acc':accuracy_score,'f1':f1_score}
eval_m = EvalModel(metrics)
kf = KFold(**kfold_params_dict)


## funcs

def prepare_callbacks(callbacks_params,name):
    callbacks_p = copy.deepcopy(callbacks_params)
    callbacks_p['model_type'] = name
    return callbacks_p

def create_model_by_data(X, Y, model, model_name,model_params, callbacks_params, fit_params,eval_m, save_path,kargs={}):
    models = {}
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        name = model_name + "_fold_{}".format(i)
        c_params = prepare_callbacks(callbacks_params,name)
        models[name] = model(x_train, y_train, x_test, y_test, name,model_params, c_params, fit_params,eval_m, save_path,**kargs)
    return models


def avg_result(results,metrics):
    avg = {}
    for metric in metrics.keys():
        res = [d[metric] for d in results.values()]
        avg[metric] ={'mean':np.mean(res),'std':np.std(res)}
    return avg

def avg_experts(models, model_names, nb_splits):
    res = {}
    for i in range(nb_splits):
        preds = []
        for name in model_names:
            preds.append(models[name][name+'_exp_fold_'+str(i)].prediction)
        avg_pred = np.mean(preds,axis=0)
        hard_pred = np.round(avg_pred,0)
        res[i] = eval_m.eval(models[name][name+'_exp_fold_'+str(i)].check_labels,hard_pred)
    return res

def avg_total(results,metrics):
    res = {}
    for metric in metrics.keys():
        m = [run[metric]['mean'] for run in results]
        s = [run[metric]['std'] for run in results]
        res[metric] = {'mean':np.mean(m),'std':np.mean(s)}
    return res

## create models

experts = {}
for view in data_view:
    view_exp = create_model_by_data(data[view],target,ExpertModel,"{}_exp".format(view),expert_params,callbacks_params,fit_params_dict,eval_m, run_dir)
    experts[view] = view_exp

#basemodels = create_model_by_data(concat_data, target, BaseLineModel, "baseline", baseline_params, callbacks_params, fit_params_dict, eval_m, run_dir)
moe_model = create_model_by_data(concat_data,target,MOE,"moe",moe_params,callbacks_params,fit_params_dict,eval_m,run_dir,{'expert_num':expert_num,'experts':None})
# for i,model in enumerate(moe_model.values()):
#     exp = [experts[view]["{}_exp_fold_{}".format(view,i)] for view in data_view]
#     model.add_experts(exp)
multilabel_moe = create_model_by_data(concat_data,target,MultilabelMOE,"multilabel_moe",moe_params,callbacks_params,multi_moe_params,eval_m,run_dir,{'expert_num':expert_num,'experts':None})
models = {}
#models.update(experts)
#models["baseline"]=basemodels
models["moe"] = moe_model
#models['multilabel'] = multilabel_moe


## plot handlers
plot_handlers = {}
plot_metrics  = ['loss']#,'acc']
for model_name in ['moe']:#['CC','MLO','baseline','moe']:
    plot_handlers[model_name] = PlotHandler(models[model_name].values(), run_dir, plot_metrics)
#plot_handlers['multilabel'] = PlotHandler(models['multilabel'].values(), run_dir, ['loss','main_output_acc','exp0_output_acc','exp1_output_acc'])

## train & eval

seed_num = 7
init_seed = 12
total_results = defaultdict(list)
for seed in range(init_seed,init_seed+seed_num):
    logger.info( "\n seed {} \n".format(seed))
    seed_results = defaultdict(dict)
    for model_type,m in models.items():
        for model in m.values():
            model._reset()
            model.prepare_model(seed)
            model.pretrain_model()
            model.fit_model()
            model.predict_model()
            model.eval_model()
            seed_results[model_type][model.name] = model.eval_results

    # seed_results['avg'] = avg_experts(experts, data_view, nb_splits)
    for model in models.keys():#+['avg']:
        avg = avg_result(seed_results[model], metrics)
        total_results[model].append(avg)
    for k,v in plot_handlers.items():
        plot_handlers[k].plot_metrics()

    for model_name, model in moe_model.items():
        logger.info("\nmodel {} \n ".format(model_name))
        model._create_stat_model()
        model.predict_stats()
        model.print_stats()

logger.info( "\n total avg performance: \n")
for model in models.keys():#+['avg']:
    avg = avg_total(total_results[model], metrics)
    for k, v in avg.items():
        logger.info( "{} {}    {:.5f} (+/- {:.5f})".format(model, k, v['mean'], v['std']))


