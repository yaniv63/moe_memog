import numpy as np
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from utils.params import mean_fpr

class MyModel(object):
    def load_weights(self, w,byname=False):
        self.model.load_weights(w,by_name=byname)

    def __init__(self, x_train, y_train, x_test, y_test, name,model_params, callback_params, fit_params,eval_tool, save_path):
        """
        :param x_train: numpy vectors for model
        :param target: data labels
        :param name: model name
        :param callback_params: params to use in building callbacks
        :param fit_params: params to use in fit phase
        :param save_path: path to save model weights
        """
        self.x_train,self.x_val,self.y_train,self.y_val=\
            train_test_split(x_train,y_train,shuffle=False,random_state=0,test_size=fit_params['validation_split'])
        self.y_test = y_test
        self.x_test = x_test
        self.save_path = save_path
        self.model_params = model_params
        self.fit_params = fit_params
        self.callback_params = callback_params
        self.eval_m = eval_tool
        self.name = name
        self.type = self.name.split('_')[0]
        self._reset()

    def fit_model(self):
        if self.model:
            params = self.fit_params
            self._compile_phase(params)
            self._fit_phase(params)
        else:
            raise Exception("you have to create the model first")

    def _compile_phase(self,params):
        self.model.compile(optimizer=params['optimizer'], loss=params['loss'],
                           metrics=params['metrics'])
    def _fit_phase(self,params):
        self.history = self.model.fit(self.x_train, self.y_train, callbacks=self.callbacks,
                       batch_size=params['batch_size'], epochs=params['epochs'],
                       verbose=params['verbose'])

    def pred_data(self,set='val'):

        if set =='train':
            self.check_sampels = self.x_train
            self.check_labels = self.y_train
        elif set == 'val':
            self.check_sampels = self.x_val
            self.check_labels = self.y_val
        elif set == 'test':
            self.check_sampels = self.x_test
            self.check_labels = self.y_test



    def predict_model(self, predict_set='val'):
        self.pred_data(predict_set)
        self.prediction = self.model.predict(self.check_sampels)
        self.hard_pred = np.round(self.prediction,0)

    def _reset(self):
        self.model = None
        self.callbacks = []
        self.prediction = None
        self.stat_model = None
        self.seed = None

    def prepare_model(self,seed,callback_params=None,model_params=None):
        if model_params is None:
            model_params = self.model_params
        if callback_params is None:
            callback_params = self.callback_params
        self.seed = seed
        np.random.seed(seed)
        self.model = self._create_model(model_params)
        self.callbacks = self._create_callbacks(callback_params)

    def eval_model(self):
        self.eval_results  = self.eval_m.eval(self.check_labels,self.hard_pred)

    def _create_model(self,params):
        raise NotImplementedError

    def _create_stat_model(self,params):
        raise NotImplementedError

    def _create_callbacks(self,params):
        self.w_path = self.save_path + 'model_{}.h5'.format(self.name +'_'+ str(self.seed))
        save_weights = ModelCheckpoint(filepath=self.w_path,
                                       **params['ModelCheckpoint'])#params['name']

        early_stop = EarlyStopping(**params['earlystop'])

        reduce_lr = ReduceLROnPlateau(**params['ReduceLROnPlateau'])
        mycallbacks = [save_weights, early_stop, reduce_lr]

        return mycallbacks

    def get_history(self,metric):
        return self.history.history[metric]

    def pretrain_model(self):
        pass

    def calc_roc(self):
        fpr,tpr,threshold = roc_curve(self.y_test,self.prediction)
        self.interp_tpr = np.interp(mean_fpr,fpr,tpr)
