from keras import backend as K
from keras.callbacks import ModelCheckpoint, LambdaCallback, EarlyStopping,ReduceLROnPlateau


import logging
logger = logging.getLogger('root')

# print_logs = LambdaCallback(on_epoch_end=lambda epoch, logs:
#     logger.debug("epoch {} loss {:.5f} acc {:.5f} fmeasure {:.5f} val_loss {:.5f} val_acc {:.5f} val_fmeasure{:.5f} ".
#                  format(epoch, logs['loss'], logs['acc'], logs['fmeasure'], logs['val_loss'], logs['val_acc'],
#                         logs['val_fmeasure'])))

def create_callbacks(name,fold,callbacks_params,save_path):
    save_weights = ModelCheckpoint(filepath=save_path + 'model_{}_fold_{}.h5'.format(name, fold), **callbacks_params['ModelCheckpoint'])
    print_logs = LambdaCallback(on_epoch_end=lambda epoch, logs:
    logger.debug("epoch {} loss {:.5f} acc {:.5f} fmeasure  ".
                 format(epoch, logs['loss'], logs['acc'])))#, logs['fmeasure'])))
    #reducelr = ReduceLR(name,fold,0.8,patience=15,save_path=save_path)
    early_stop = EarlyStopping(**callbacks_params['earlystop'])
    reduce_lr = ReduceLROnPlateau(**callbacks_params['ReduceLROnPlateau'])
    mycallbacks = [print_logs,save_weights,early_stop,reduce_lr] #reducelr
    return mycallbacks


class ReduceLR(EarlyStopping):

    def __init__(self,name,fold,factor,save_path,*args,**kwargs):
        super(ReduceLR,self).__init__(*args,**kwargs)
        self.name = name
        self.fold = fold
        self.factor = factor
        self.save_path

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            logger.warn('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logger.info('loading previous weights')
                self.model.load_weights(self.save_path + 'model_{}_fold_{}.h5'.format(self.name, self.fold))
                self.wait = 0
                old_lr = float(K.get_value(self.model.optimizer.lr))
                new_lr = old_lr * self.factor
                K.set_value(self.model.optimizer.lr, new_lr)
                logger.info('\nEpoch {}: reducing learning rate from  {} to {}'.format(epoch,old_lr, new_lr))