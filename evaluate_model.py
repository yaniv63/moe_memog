import logging
logger = logging.getLogger('root')
from keras.models import Model
import numpy as np


def eval_model(model,data,target,metrics):
    assert isinstance(model, Model) , "expecting keras model" #TODO use interface
    results = []
    prediction = model.predict(data)
    hard_pred = np.round(prediction,0)
    for metric in metrics:
        try:
            result = metric(target,hard_pred)
            results.append(result)
        except Exception as e:
            raise e
    return results



