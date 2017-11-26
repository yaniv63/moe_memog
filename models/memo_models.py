import numpy as np
from my_model import MyModel
from keras.models import Sequential,Model
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras.layers import Dense,Dropout,concatenate,dot
from keras.utils import to_categorical
from utils.plotting_tools import plot_training
import logging
logger = logging.getLogger('root')

def expert_NN(name,params):
    model = Sequential(name = "exp{}".format(name))
    model.add(Dense(params['nn_layer1'], activation="relu", input_dim=14, name='exp{}_dense1'.format(name),
                    kernel_initializer=params['w_init']))
    model.add(Dropout(rate=params['dropout1']))
    model.add(Dense(params['nn_layer2'], activation="relu", name='exp{}_dense2'.format(name),
                    kernel_initializer=params['w_init']))
    model.add(Dropout(rate=params['dropout2']))
    model.add(
        Dense(1, activation="sigmoid", name='exp{}_output'.format(name), kernel_initializer=params['w_init']))
    return model

def gate_model(params,expert_num,dims):
    gate =Sequential(name="gate_model")
    gate.add(Dense(params['nn_gate1'], activation='relu', name='dense1_gate',
                       kernel_initializer=params['w_init'],input_dim=dims))
    gate.add(Dense(params['nn_gate2'], activation='relu', name='dense2_gate',
                        kernel_initializer=params['w_init']))
    gate.add(Dropout(rate=params['dropout2']))

    gate.add(Dense(expert_num, activation='softmax', name='out_gate',
                         kernel_initializer=params['w_init']))
    return gate

class BaseLineModel(MyModel):

    def __init__(self,x_train, y_train, x_test, y_test, name,model_params, callback_params, fit_params,eval_m, save_path):
        super(BaseLineModel,self).__init__(x_train, y_train, x_test, y_test, name,model_params, callback_params, fit_params,eval_m, save_path)

    def _create_model(self,params):
        model = Sequential()
        model.add(Dense(params['nn_layer1'], kernel_initializer=params['w_init'], activation="relu", input_dim=28))
        model.add(Dropout(rate=params['dropout1']))
        model.add(Dense(params['nn_layer2'], kernel_initializer=params['w_init'], activation="relu"))
        model.add(Dropout(rate=params['dropout2']))
        model.add(Dense(params['nn_layer3'], kernel_initializer=params['w_init'], activation="relu"))
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
            experts.append(expert_NN( str(i)+'_expert', params))
            data.append(experts[i].input)
            decisions.append(experts[i].output)

        merged_decisions = concatenate(decisions, axis=1)

        gate_input = concatenate(data, axis=1)
        gate_dense = Dense(params['nn_gate1'], activation='relu', name='dense1_gate',
                           kernel_initializer=params['w_init'])(gate_input)
        gate_dense2 = Dense(params['nn_gate2'], activation='relu', name='dense2_gate',
                            kernel_initializer=params['w_init'])(gate_dense)
        d2 = Dropout(rate=params['dropout2'])(gate_dense2)
        coefficients = Dense(self.expert_num, activation='softmax', name='out_gate',
                             kernel_initializer=params['w_init'])(d2)

        #coefficients = gate_model(params,self.expert_num,dims=28)(gate_input)
        weighted_prediction = dot([coefficients, merged_decisions], axes=1, name='main_output')
        model = Model(inputs=data, outputs=weighted_prediction)
        return model

    def pretrain_model(self):

        def expert_predict(name,data,labels,val,val_l):
            # expert = expert_NN(name, self.model_params)
            # expert.compile(loss=self.fit_params['loss'], optimizer=self.fit_params['optimizer'])
            # expert.fit(data, labels, epochs=500, verbose=0,)
            #            expert.predict_model()
            #expert.save_weights(filename)
            #filename = 'exp{}_pretrain'.format(name)
            #return expert.predict(data),expert.predict(val)

            name =str(name) + '_expert'
            expert = ExpertModel(data,labels,val,val_l,name,self.model_params,self.callback_params,self.fit_params,self.eval_m,self.save_path)
            expert.prepare_model(self.seed)
            expert.fit_model()
            layers_n = [n.name for n in expert.model.layers]
            self.model.load_weights(expert.w_path, by_name=True)
            for l in layers_n:
                try:
                    self.model.get_layer(l).trainable = False
                except ValueError:
                    pass
            self.model.compile(loss=self.fit_params['loss'], optimizer=self.fit_params['optimizer'])
            return 0

        predicts = [expert_predict(i,self.x_train[i],self.y_train,self.x_val[i],self.y_val) for i in range(self.expert_num)]
        # predict_train = [x[0] for x in predicts ]
        # predict_val =  [x[1] for x in predicts]
        # gate_labels = [np.argmin(np.abs(np.array(predict)-label))  for predict,label in zip(zip(*predict_train),self.y_train)]
        # gate_labels_val = [np.argmin(np.abs(np.array(predict)-label))  for predict,label in zip(zip(*predict_val),self.y_val)]
        # labels= to_categorical(gate_labels)
        # labels_val = to_categorical(gate_labels_val)
        # gate =gate_model(self.model_params,self.expert_num,dims=28)
        # gate.compile(loss='categorical_crossentropy',optimizer='sgd')
        # his = gate.fit(np.concatenate(self.x_train,axis=1),labels,validation_data=(np.concatenate(self.x_val,axis=1),labels_val),epochs=500,verbose=2)
        # plot_training(his.history,self.name,"./temp/{}".format(self.name))
        # r = gate.predict(np.concatenate(self.x_train,axis=1))
        # print "train: \n",r
        # p = gate.predict(np.concatenate(self.x_val,axis=1))
        # print "val:\n",p
        # filename = 'gate_pretrain_{}'.format(self.name)
        # gate.save_weights(filename)
        # self.model.load_weights(filename,by_name=True)

    def add_experts(self,experts):
        self.experts = experts

    def _create_stat_model(self,model = None):
        model = model if model is not None else self.model
        multipredict = self._add_experts_output(model)
        self.stat_model = self._add_gate_output(multipredict)

    def _add_gate_output(self,model):
        return Model(
            inputs=model.inputs,
            outputs=model.outputs + [model.get_layer('out_gate').output]
        )


    def _add_experts_output(self, model):
        outputs = [model.output]
        for i in range(self.expert_num):
            outputs.append(model.get_layer('exp{}_output'.format(str(i)+'_expert')).output)
        multipredict_model = Model(
            inputs=model.inputs,
            outputs=outputs
        )
        return multipredict_model

    def predict_stats(self,model=None,predict_set='val'):
        self.pred_data(predict_set)
        model = model if model is not None else self.stat_model
        self.prediction = model.predict(self.check_sampels)
        self.hard_pred = {}
        for i,name in enumerate(['main'] + range(self.expert_num)):
            self.hard_pred[name] = np.round(self.prediction[i],0)
        self.prediction = zip(*self.prediction)

    def print_stats(self):
        if self.prediction is not None:
            res = self.prediction
            for j in range(len(self.check_labels)):
                logger.info( " sample {} target {} output {:.2f} exp0 {:.2f} exp1 {:.2f} gate {} ".format(j, self.check_labels[j],
                                                                                                  float(res[j][0]),
                                                                                                  float(res[j][1]),
                                                                                                  float(res[j][2]),
                                                                                                  res[j][3]))
        else:
            raise Exception("need to predict before presenting")

class MultilabelMOE(MOE):

    def __init__(self, x_train, y_train, x_test, y_test, name, model_params, callback_params, fit_params, eval_m,
                 save_path,expert_num,experts = None):
        super(MultilabelMOE, self).__init__(x_train, y_train, x_test, y_test, name, model_params, callback_params,
                                          fit_params, eval_m, save_path,expert_num,experts)
        self.y_train = [self.y_train]*(self.expert_num+1)

    def _create_model(self,params):
        model = super(MultilabelMOE, self)._create_model(params)
        multipredict = super(MultilabelMOE, self)._add_experts_output(model)
        return multipredict

    def _create_stat_model(self,model = None):
        m = self.model if model is None else model
        self.stat_model = self._add_gate_output(m)


    def _compile_phase(self,params):
        self.model.compile(optimizer=params['optimizer'], loss=params['loss'],
                           metrics=params['metrics'],loss_weights =params['loss_weights'])

    def predict_model(self,predict_set='val'):
        self.load_weights(self.w_path,byname=False)
        super(MultilabelMOE, self).predict_stats(self.model,predict_set)

    def eval_model(self):
        self.eval_results  = self.eval_m.eval(self.check_labels,self.hard_pred['main'])
        self.exp_results = {}
        for i in range(self.expert_num):
            self.exp_results[i] = self.eval_m.eval(self.check_labels,self.hard_pred[i])

    def pretrain_model(self):

        def expert_predict(name,data,labels):
            expert = expert_NN(name, self.model_params)
            expert.compile(loss=self.fit_params['loss'], optimizer=self.fit_params['optimizer'])
            expert.fit(data, labels, epochs=500,verbose=0)
            filename = 'exp{}_pretrain'.format(name)
            expert.save_weights(filename)
            self.model.load_weights(filename, by_name=True)
            return expert.predict(data)

        predicts = [expert_predict(i,self.x_train[i],self.y_train[0]) for i in range(self.expert_num)]
        predicts = zip(*predicts)
        gate_labels = [np.argmin(np.abs(np.array(predict)-label))  for predict,label in zip(predicts,self.y_train[0])]
        labels= to_categorical(gate_labels)
        gate =gate_model(self.model_params,self.expert_num,dims=28)
        gate.compile(loss='categorical_crossentropy',optimizer='sgd')
        his =gate.fit(np.concatenate(self.x_train,axis=1),labels,epochs=500,verbose=0)
        plot_training(his.history,self.name,"./temp/{}".format(self.name))
        filename = 'gate_pretrain_{}'.format(self.name)
        gate.save_weights(filename)
        self.model.load_weights(filename,by_name=True)