from keras.models import Model
import logging
#from utils.plotting_tools import plot_training
from train_tools import create_callbacks
from load_data import split_data,get_fold_samples
from utils.params import callbacks_params
import pickle


logger = logging.getLogger('root')


def load_weights_for_fold(model,init_w,is_multi_expert,fold,path):
    model.set_weights(init_w)
    # if is_multi_expert:
    #     model.load_weights(path +"model_CC_fold_{}.h5".format(fold),by_name=True)
    #     model.load_weights(path +"model_MLO_fold_{}.h5".format(fold),by_name=True)



def train_phase(model, data, target, name, fit_params_dict, data_split_indexes,is_multi_expert,save_path):

    assert isinstance(model, Model) , "expecting keras model" #TODO use interface
    init_w = model.get_weights()
    logs = []
    n_splits = len(data_split_indexes)
    for i in range (n_splits):
        logger.info("train fold {}".format(i))
        X_train, y_train = get_fold_samples(data,target,data_split_indexes,is_multi_expert,fold=i,is_train=True)
        logger.info("training expert {}".format(name))
        load_weights_for_fold(model,init_w,is_multi_expert,fold=i,path = save_path)
        callbacks = create_callbacks(name=name,fold=i,callbacks_params=callbacks_params,save_path=save_path)
        history = model.fit(X_train,y_train,callbacks=callbacks,**fit_params_dict)
        #model.save_weights(save_path + "model_{}_fold_{}.h5".format(name,i))
        logs.append(history.history)
        logger.info("finished fold {}".format(i))

    #plot_training(logs,name,save_path)
    # with open(save_path + 'cross_valid_stats_{}.lst'.format(name), 'wb') as fp:
    #     pickle.dump(logs, fp)
