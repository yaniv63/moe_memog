import logging
logger = logging.getLogger('root')

callbacks_params = {'earlystop':{'patience':50,'monitor':'loss','mode':'min'},
                    'ReduceLROnPlateau': {'monitor':'loss', 'factor':0.2,
                                  'patience':15, 'min_lr':0.001,'verbose':2},
                    'ModelCheckpoint':{'monitor':'loss','save_best_only':True,'save_weights_only':True}}

params = {
    'optimizer' : ['sgd','adam'],
    'nn_layer1' : [6,10,15,24],
    'nn_layer2' : [6,10,15,24],
    'dropout1' : [0,0.2,0.5,0.8],
    'dropout2': [0, 0.2, 0.5, 0.8],
    'epoch_num' : [100,500],
    'w_init' : ['glorot_normal','glorot_uniform'],
    'nn_gate' : [3,6,10]
}

