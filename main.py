import  numpy as np
import pickle
from models.memo_models import BaseLineModel,ExpertModel,MOE,MultilabelMOE
from load_data import  load_medical_data
from sklearn.model_selection import KFold
from utils.params import callbacks_params,mean_fpr
from utils.paths import get_run_dir
import copy
from eval_model import EvalModel
from collections import defaultdict
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,precision_score,recall_score,roc_curve
from utils.plotting_tools import PlotHandler,plot_roc
from utils.logging_tools import get_logger
run_dir = get_run_dir()
logger = get_logger(run_dir)

## data
data_view = ['CC','MLO']

data_type = ['Dense','Fatty','all']
type_v = 1
data,target = load_medical_data(data_type[type_v])
concat_data = np.concatenate((data['CC'], data['MLO']), axis=1)

## params
nb_splits = 10
kfold_params_dict = {'n_splits': nb_splits, 'shuffle': False, 'random_state': 42}
fit_params_dict = {'batch_size': 16, 'epochs':500, 'validation_split': 0.0, 'verbose': 0,
                   'optimizer' : 'adam', 'loss' : 'binary_crossentropy', 'metrics' : ['accuracy'],'shuffle': False}
expert_params = {  'nn_layer1' : 24, 'nn_layer2' : 12,
                  'dropout1' : 0.5,'dropout2': 0.5,'w_init': 'glorot_uniform' }
baseline_params = {'nn_layer1' : 24, 'nn_layer2' : 24,'nn_layer3' : 20,'nn_layer4' : 3,
                  'dropout1' : 0.5,'dropout2': 0.5,'dropout3': 0.5,'w_init': 'glorot_uniform'}
moe_params = { 'nn_layer1' : 24, 'nn_layer2' : 24,
                  'dropout1' : 0.5,'dropout2': 0.5,'w_init': 'glorot_uniform','nn_gate1' : 3,'nn_gate2':3}
multi_moe_params =copy.copy(fit_params_dict)
multi_moe_params['loss_weights'] = [1,1,1]

logger.info("data type {}".format(data_type[type_v]))
logger.info("{}".format(fit_params_dict))
logger.info("{}".format(callbacks_params))
logger.info("expert params {}".format(expert_params))
logger.info("baseline_params {}".format(baseline_params))
logger.info("moe params {}".format(moe_params))
logger.info("multilabel params {}".format(multi_moe_params))

expert_num =2
## helping objects
metrics = {'acc':accuracy_score,'f1':f1_score,'auc':roc_auc_score,'precision':precision_score,'recall':recall_score}#'roc':roc_curve}
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

def aggregate_result(agg,results,metrics):
    for metric in metrics.keys():
        res = [d[metric] for d in results.values()]
        agg[metric] =[a+b for a,b in zip(res,agg[metric])]
    return agg

def avg_experts(models, model_names, nb_splits):
    res = {}
    stats = {}
    for i in range(nb_splits):
        preds = []

        for name in model_names:
            preds.append(models[name][name+'_exp_fold_'+str(i)].prediction)
        avg_pred = np.mean(preds,axis=0)
        hard_pred = np.round(avg_pred,0)
        labels = models[name][name+'_exp_fold_'+str(i)].check_labels
        res[i] = eval_m.eval(labels,hard_pred)
        stats[i] = [avg_pred,labels]
    return res,stats

def avg_total(results,metrics):
    res = {}
    for metric in metrics.keys():
        m = [run[metric]['mean'] for run in results]
        s = [run[metric]['std'] for run in results]
        res[metric] = {'mean':np.mean(m),'std':np.mean(s)}
    return res

def avg_roc(models):
    rocs = {}
    for model_type,m in models.items():
        cross_roc = []
        for model in m.values():
            cross_roc.append(model.interp_tpr)
        rocs[model_type] =np.mean(np.vstack(cross_roc),0)
    return rocs

# def fold_roc(models):
#     rocs = {}
#     for model_type,m in models.items():
#         cross_roc = []
#         for model in m.values():
#             cross_roc.append(model.interp_tpr)
#         rocs[model_type] =np.mean(np.vstack(cross_roc),0)
#     return rocs

def aggragate_init(model_names,metrics,init_v,nb_folds):
    agg = defaultdict(dict)
    for model in model_names:
        for metric in metrics:
            agg[model][metric] = [init_v]*nb_folds
    return agg

def avg_seed_roc(types,rocs):
    r = {}
    for type in types:
        roc_type = [roc[type] for roc in rocs]
        r[type] = np.mean(np.vstack(roc_type),0)
    return r

def avg_exp_roc(models, model_names, nb_splits):
    res = []
    for i in range(nb_splits):
        preds = []
        for name in model_names:
            preds.append(models[name][name+'_exp_fold_'+str(i)].prediction)
        avg_pred = np.mean(preds,axis=0)
        fpr, tpr, threshold = roc_curve(models[name][name+'_exp_fold_'+str(i)].y_test, avg_pred)
        res.append( np.interp(mean_fpr, fpr, tpr))
    return np.mean(np.vstack(res),0)

def stat_test(models,types,metrics):
    from scipy.stats import ttest_rel
    metric = 'acc'
    model = 'baseline'
    a = models['multilabel'][metric]
    b = models[model][metric]
    print ttest_rel(a,b)

def save_predictions(seed,models,avg_stats):
    with open('./model_stats_dense_{}.h'.format(seed),'wb') as f:
        preds = {}
        for model_type, m in models.items():
            model_preds = {}
            for model_name,model in m.items():
                model_preds[model_name] = [model.prediction,model.y_test]
            preds[model_type] = model_preds
        preds['avg'] = avg_stats
        pickle.dump(preds,f)

## create models

experts = {}
for view in data_view:
    view_exp = create_model_by_data(data[view],target,ExpertModel,"{}_exp".format(view),expert_params,callbacks_params,fit_params_dict,eval_m, run_dir)
    experts[view] = view_exp

basemodels = create_model_by_data(concat_data, target, BaseLineModel, "baseline", baseline_params, callbacks_params, fit_params_dict, eval_m, run_dir)
moe_model = create_model_by_data(concat_data,target,MOE,"moe",moe_params,callbacks_params,fit_params_dict,eval_m,run_dir,{'expert_num':expert_num,'experts':None})
multilabel_moe = create_model_by_data(concat_data,target,MultilabelMOE,"multilabel_moe",moe_params,callbacks_params,multi_moe_params,eval_m,run_dir,{'expert_num':expert_num,'experts':None})

models = {}
models.update(experts)
models["baseline"]=basemodels
#models["moe"] = moe_model
models['multilabel'] = multilabel_moe


## plot handlers
plot_handlers = {}
plot_metrics  = ['loss','acc']
for model_name in ['CC','MLO','baseline']:#,'moe']:
    plot_handlers[model_name] = PlotHandler(models[model_name].values(), run_dir, plot_metrics)
plot_handlers['multilabel'] = PlotHandler(models['multilabel'].values(), run_dir, ['loss','main_output_acc','exp0_output_acc','exp1_output_acc'])

## train & eval

stat_seed = aggragate_init(models.keys()+['avg'],metrics.keys(),0,10)
seed_num = 7
init_seed = 12
total_results = defaultdict(list)
total_rocs = []
for seed in range(init_seed,init_seed+seed_num):
    logger.info( "\n seed {} \n".format(seed))
    seed_results = defaultdict(dict)
    for model_type,m in models.items():
        for model in m.values():
            model._reset()
            model.prepare_model(seed)
 #           model.pretrain_model()
            model.fit_model()
            model.predict_model(predict_set='test')
            model.calc_roc()
            model.eval_model()
            seed_results[model_type][model.name] = model.eval_results

    seed_results['avg'],avg_stats = avg_experts(experts, data_view, nb_splits)
    avg_model_roc = avg_exp_roc(experts, data_view, nb_splits)
    models_roc = avg_roc(models)
    models_roc['avg'] = avg_model_roc

    total_rocs.append( models_roc)
    for model in models.keys()+['avg']:
        avg = avg_result(seed_results[model], metrics)
        total_results[model].append(avg)
        stat_seed[model] = aggregate_result(stat_seed[model],seed_results[model], metrics)

    # stat_test(stat_seed,models.keys()+['avg'],metrics)

    for k,v in plot_handlers.items():
        plot_handlers[k].plot_metrics()

    for model_name, model in multilabel_moe.items():
        logger.info("\nmodel {} \n ".format(model_name))
        model._create_stat_model()
        model.predict_model(use_stat_model=True,predict_set='test')
        model.print_stats()
    save_predictions(seed,models,avg_stats)


logger.info( "\n total avg performance: \n")
for model in models.keys()+['avg']:
    avg = avg_total(total_results[model], metrics)
    for k, v in avg.items():
        logger.info( "{} {}    {:.5f} (+/- {:.5f})".format(model, k, v['mean'], v['std']))


rocs = avg_seed_roc(models.keys()+['avg'],total_rocs)
with open("roc_data.h","wb") as f:
    pickle.dump(rocs,f)
# plot_roc(rocs,models.keys()+['avg'])
# stat_test(stat_seed,models.keys()+['avg'],metrics.keys())
#

