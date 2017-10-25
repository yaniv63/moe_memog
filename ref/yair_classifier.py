from sklearn.model_selection import StratifiedKFold
from medical import load_medical_data
from keras.models import Sequential
from keras.models import Model,load_model,save_model
from keras.layers import Dense, Input, merge
from keras import backend as K
from keras import callbacks
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD,RMSprop
import scipy.io
import scipy
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

def my_init(shape, dtype=None, name=None):
    return 0.5*K.ones(shape, dtype=dtype)


class My_Callback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        model.save_weights('init_weights.h5')
        print 'save init weights'


def load_data(set_type):
    (X_train,Y_train),(X_test,y_test) = load_medical_data(data_type='Fatty',test_ratio=0,set_type=set_type)
    return (X_train,Y_train)

def create_basic_net():
    model = Sequential()
    model.add(Dense(24, input_dim=14, activation='relu', init='glorot_normal'))
    model.add(Dense(24,  activation='relu', init='glorot_normal'))
    model.add(Dense(1, activation='sigmoid', init='glorot_normal'))
    return model

def generate_model(model_idx,X_CC,X_MLO,y):
    model_CC = []
    model_MLO = []
    if model_idx == 1:
        model = Sequential()
        model.add(Dense(24, input_dim=28, activation='relu', init='glorot_normal'))
        model.add(Dense(24, activation='relu', init='glorot_normal'))
        model.add(Dense(2, activation='softmax', init='glorot_normal'))

    if model_idx == 2 or model_idx == 3:
        model_CC = create_basic_net()
        #sgd = SGD(lr=0.01, decay=1e-5, momentum=0.5, nesterov=True)
        #model_CC.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        #model_CC.fit(X_CC, y, nb_epoch=100, batch_size=100, verbose=0)
        #CC_init_w = model_CC.get_weights()
        model_MLO = create_basic_net()
        #model_MLO.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        #model_MLO.fit(X_MLO, y, nb_epoch=100, batch_size=100, verbose=0)
        #MLO_init_w = model_MLO.get_weights()

        CCin = Input(shape=(14,))
        MLOin = Input(shape=(14,))

        y_CC = model_CC(CCin)
        y_MLO = model_MLO(MLOin)
        merged = merge([y_CC, y_MLO],'concat')
        y = Dense(1, activation='sigmoid', init='one')(merged)
        model = Model(input=[CCin, MLOin], output=y)
        model_CC =  Model(input=CCin, output=y_CC)
        model_MLO = Model(input=MLOin, output=y_MLO)
        final_w = model.get_weights()
    return model,model_CC,model_MLO
# fix random seed for reproducibility
#seed = 19
#np.random.seed(seed)
# split into input (X) and output (Y) variables
[X_CC,y] = load_data('CC')
[X_MLO,y] = load_data('MLO')
[X,yy] = load_data('both')

#reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.7,
#                               min_lr=0.0001 , verbose=0, epsilon=0.01)
reduce_lr = []
print_w = My_Callback()
# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True)
cvscores = []
cvscores_CC = []
cvscores_MLO = []
tot_score = []
num_of_seeds = 20
for idx in range(num_of_seeds):
    for train, test in kfold.split(X_CC, yy):
        y = to_categorical(yy, 2)
        model_idx = 1
        model,model_CC,model_MLO=generate_model(model_idx,X_CC,X_MLO,y)
        # Compile model
        sgd = SGD(lr=0.1, decay=1e-5, momentum=0.5, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Fit the model
        if model_idx == 1:
            model.fit(X[train], y[train], nb_epoch=500, batch_size=200, verbose=0)
            scores = model.evaluate(X[test], y[test], verbose=0)
        if model_idx == 2:
            model.fit([X_CC[train], X_MLO[train]], y[train], nb_epoch=1000, batch_size=512, verbose=1, callbacks=[])
            scores = model.evaluate([X_CC[test], X_MLO[test]], y[test], verbose=1)
        if model_idx == 3:
            model_CC.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
            model_CC.fit(X_CC[train], y[train], nb_epoch=1000, batch_size=512, verbose=0, callbacks=[])
            model_MLO.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
            model_MLO.fit(X_MLO[train], y[train], nb_epoch=1000, batch_size=512, verbose=0, callbacks=[])
            scores_CC = model_CC.evaluate(X_CC[test], y[test], verbose=0)
            scores_MLO = model_MLO.evaluate(X_MLO[test], y[test], verbose=0)

        # printim:
        if model_idx != 3:
            #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1] * 100)
        else:
            #print("CC: %s: %.2f%%" % (model_CC.metrics_names[1], scores_CC[1] * 100))
            #print("MLO: %s: %.2f%%" % (model_MLO.metrics_names[1], scores_MLO[1] * 100))
            cvscores_CC.append(scores_CC[1] * 100)
            cvscores_MLO.append(scores_MLO[1] * 100)
    print('iteration %s / % s:' % (idx , num_of_seeds))
    if model_idx != 3:
        #print (cvscores)
        print("    %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    else:
        print("    CC: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores_CC), np.std(cvscores_CC)))
        print("    MLO: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores_MLO), np.std(cvscores_MLO)))

print ' ------ THE END -------'
