import logging
logger = logging.getLogger('root')
from keras.models import Model
import numpy as np


def eval_model(model,data,target,metrics):
    assert isinstance(model, Model) , "expecting keras model" #TODO use interface
    prediction = model.predict(data)
    hard_pred = np.round(prediction,0)
    return eval(target,hard_pred,metrics),prediction

# def avg_predictions(pred_arr,target,metrics,n_splits):
#     for i in range(n_splits):
#
#     pred = np.mean(pred_arr,axis=0)
#     hard_pred = np.round(pred,0)
#     return eval(target,hard_pred,metrics)


def eval(target,pred,metrics):
    results = []
    for metric in metrics:
        try:
            result = metric(target,pred)
            results.append(result)
        except Exception as e:
            raise e
    return results




