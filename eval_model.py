class EvalModel(object):
    def __init__(self,metrics):
        self.metrics = metrics

    def eval(self,target,predict):
        results = {}
        for metric_name,metric in self.metrics.items():
            try:
                result = metric(target, predict)
                results[metric_name] = result
            except Exception as e:
                raise e
        return results