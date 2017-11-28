import numpy as np
from my_model import MyModel
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,concatenate,dot
import logging
logger = logging.getLogger('root')

def expert_NN(name,params):
    model = Sequential()
    model.add(Dense(params['nn_layer1'], activation="relu", input_dim=14, name='exp{}_dense1'.format(name),
                    kernel_initializer=params['w_init']))
    model.add(Dropout(rate=params['dropout1']))
    model.add(Dense(params['nn_layer2'], activation="relu", name='exp{}_dense2'.format(name),
                    kernel_initializer=params['w_init']))
    model.add(Dropout(rate=params['dropout2']))
    model.add(
        Dense(1, activation="sigmoid", name='exp{}_output'.format(name), kernel_initializer=params['w_init']))
    return model

def gate_model(params,expert_num,data):
    gate_input = concatenate(data, axis=1)
    gate_dense = Dense(params['nn_gate1'], activation='relu', name='dense1_gate',
                       kernel_initializer=params['w_init'])(gate_input)
    gate_dense2 = Dense(params['nn_gate2'], activation='relu', name='dense2_gate',
                        kernel_initializer=params['w_init'])(gate_dense)
    d2 = Dropout(rate=params['dropout2'])(gate_dense2)
    coefficients = Dense(expert_num, activation='softmax', name='out_gate',
                         kernel_initializer=params['w_init'])(d2)
    return coefficients

class BaseLineModel(MyModel):

    def __init__(self,x_train, y_train, x_test, y_test, name,model_params, callback_params, fit_params,eval_m, save_path):
        super(BaseLineModel,self).__init__(x_train, y_train, x_test, y_test, name,model_params, callback_params, fit_params,eval_m, save_path)

    def _create_model(self,params):
        model = Sequential()
        model.add(Dense(params['nn_layer1'], kernel_initializer=params['w_init'], activation="relu", input_dim=28))
        model.add(Dropout(rate=params['dropout1']))
        model.add(Dense(params['nn_layer2'], kernel_initializer=params['w_init'], activation="relu"))
        model.add(Dropout(rate=params['dropout2']))
        #model.add(Dense(params['nn_layer3'], kernel_initializer=params['w_init'], activation="relu"))
        model.add(Dense(1, kernel_initializer=params['w_init'], activation="sigmoid"))
        return model

class ExpertModel(MyModel):

    def __init__(self,x_train, y_train, x_test, y_test, name,model_params, callback_params, fit_params,eval_m, save_path):
        super(ExpertModel,self).__init__(x_train, y_train, x_test, y_test, name,model_params, callback_params, fit_params,eval_m, save_path)

    def _create_model(self, params):
        model = expert_NN(self.name,params)
        # name = self.name
        # model = Sequential()
        # model.add(Dense(params['nn_layer1'], activation="relu", input_dim=14, name='exp{}_dense1'.format(name),
        #                 kernel_initializer=params['w_init']))
        # model.add(Dropout(rate=params['dropout1']))
        # model.add(Dense(params['nn_layer2'], activation="relu", name='exp{}_dense2'.format(name),
        #                 kernel_initializer=params['w_init']))
        # model.add(Dropout(rate=params['dropout2']))
        # model.add(
        #     Dense(1, activation="sigmoid", name='exp{}_output'.format(name), kernel_initializer=params['w_init']))
        return model



class MOE(MyModel):
    def __init__(self, x_train, y_train, x_test, y_test, name, model_params, callback_params, fit_params, eval_m,
                 save_path,expert_num,experts):
        super(MOE, self).__init__(x_train, y_train, x_test, y_test, name, model_params, callback_params,
                                          fit_params, eval_m, save_path)
        self.expert_num = expert_num
        # split data for experts
        self.x_train = np.split(np.array(self.x_train),self.expert_num,1)
        self.x_val = np.split(np.array(self.x_val),self.expert_num,1)
        self.x_test = np.split(np.array(self.x_test),self.expert_num,1)
        self.experts = experts


    def _create_model(self,params):
        decisions = []
        experts = []
        data = []

        for i in range(self.expert_num):
            experts.append(expert_NN( i, params))
            data.append(experts[i].input)
            decisions.append(experts[i].output)

        merged_decisions = concatenate(decisions, axis=1)

        coefficients = gate_model(params,self.expert_num,data)
        weighted_prediction = dot([coefficients, merged_decisions], axes=1, name='main_output')
        model = Model(inputs=data, outputs=weighted_prediction)
        return model

    def pretrain_model(self):
        for expert in self.experts:
            self.model.load_weights(expert.get_weights())
        predicts = zip([expert.prediction for expert in self.experts])
        gate_labels = [np.argmin(np.abs(np.array(predict)-label))  for predict,label in zip(predicts,self.check_labels)]
        gate_model(self.model_params,self.expert_num,)


class MultilabelMOE(MOE):

    def __init__(self, x_train, y_train, x_test, y_test, name, model_params, callback_params, fit_params, eval_m,
                 save_path,expert_num,experts = None):
        super(MultilabelMOE, self).__init__(x_train, y_train, x_test, y_test, name, model_params, callback_params,
                                          fit_params, eval_m, save_path,expert_num,experts)
        self.y_train = [self.y_train]*(self.expert_num+1)

    def _create_model(self,params):
        model = super(MultilabelMOE, self)._create_model(params)
        outputs = [model.output]
        for i in range(self.expert_num):
            outputs.append(model.get_layer('exp{}_output'.format(i)).output)
        multilabel = Model(
            inputs=model.inputs,
            outputs=outputs
        )
        return multilabel

    def _create_stat_model(self,create_model=False):
        model = self._create_model(self.model_params) if create_model else self.model
        self.stat_model = Model(
                            inputs=model.inputs,
                            outputs= model.outputs + [model.get_layer('out_gate').output]
        )

    def _compile_phase(self,params):
        self.model.compile(optimizer=params['optimizer'], loss=params['loss'],
                           metrics=params['metrics'],loss_weights =params['loss_weights'])

    def predict_model(self,use_stat_model=False,predict_set='val'):
        self.pred_data(predict_set)
        model = self.stat_model if use_stat_model else self.model
        self.prediction = model.predict(self.check_sampels)
        self.hard_pred = {}
        for i,name in enumerate(['main'] + range(self.expert_num)):
            self.hard_pred[name] = np.round(self.prediction[i],0)
        self.prediction = zip(*self.prediction)

    def eval_model(self):
        self.eval_results  = self.eval_m.eval(self.check_labels,self.hard_pred['main'])
        self.exp_results = {}
        for i in range(self.expert_num):
            self.exp_results[i] = self.eval_m.eval(self.check_labels,self.hard_pred[i])


    def print_stats(self):
        if self.prediction is not None:
            res = self.prediction
            for j in range(len(self.check_labels)):
                logger.info(" sample {} target {} output {:.2f} exp1 {:.2f} exp2 {:.2f} gate {} ".format(j, self.check_labels[j],
                                                                                                  float(res[j][0]),
                                                                                                  float(res[j][1]),
                                                                                                  float(res[j][2]),
                                                                                                  res[j][3]))
        else:
            raise Exception("need to predict before presenting")
