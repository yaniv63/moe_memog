import numpy as np
import itertools
import matplotlib

#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10

def generic_plot(kwargs):

    if kwargs.has_key("figure_name"):
        f1 = plt.figure(kwargs["figure_name"])
    if kwargs.has_key("title"):
        plt.title(kwargs["title"])
    if kwargs.has_key("ylabel"):
        plt.ylabel(kwargs["ylabel"])
    if kwargs.has_key("xlabel"):
        plt.xlabel(kwargs["xlabel"])
    if kwargs.has_key("line_att"):
        line_attribute = kwargs["line_att"]
    else:
        line_attribute = ''
    if kwargs.has_key("image_att"):
        image_attribute = kwargs["image_att"]
    else:
        image_attribute = {}
    if kwargs.has_key("x"):
        plt.plot(kwargs["x"],kwargs["y"],**line_attribute)
    elif  kwargs.has_key("y"):
        plt.plot(kwargs["y"],**line_attribute)
    elif kwargs.has_key("image"):
        plt.imshow(kwargs["image"],**image_attribute)
    if kwargs.has_key("legend"):
        plt.legend(kwargs["legend"], loc=0)
    if kwargs.has_key("save_file"):
        plt.savefig(kwargs["save_file"],dpi=100)

def plot_training(logs,name,save_path):
    metrics = ['acc', 'val_acc', 'loss', 'val_loss']#, 'fmeasure', 'val_fmeasure']
    linestyles = ['-', '--']
    colors = ['b','y','r','g','c','m','k',[.80, .19, .46],[.61, .51, .74],[.31, .87, .56]]
    for j,history in enumerate(logs):
        for i in [0,2,4]:
            params = {'figure_name': metrics[i]+name, 'y':history[metrics[i]],'title':'model_{} '.format(name) + metrics[i],
                      'ylabel':metrics[i],'xlabel':'epoch',"line_att":dict(linestyle=linestyles[0],color=colors[j])}
            generic_plot(params)
            #params = {'figure_name': metrics[i]+name, 'y':history[metrics[i+1]],"line_att":dict(linestyle=linestyles[1],color=colors[j])}
            #generic_plot(params)
    for i in [0, 2, 4]:
        params = {'figure_name': metrics[i]+name,
                  'save_file': save_path + 'model_{}_'.format(name) + metrics[i] + '.png'}
        generic_plot(params)
    plt.close('all')


class PlotHandler(object):
    def __init__(self,models,save_path,metrics):
        self.models = models
        self.colors = ['b','y','r','g','c','m','k',[.80, .19, .46],[.61, .51, .74],[.31, .87, .56]]
        self.save_path = save_path
        self.metrics = metrics

    def plot_metric(self):
        for metric in self.metrics:
            for i,model in enumerate(self.models):
                history = model.get_history(metric)
                self._plot_model(history,metric,model.name,i)
        plt.close('all')

    def _plot_model(self,history,metric,name,color_num):
        linestyles = ['-', '--']
        params = {'figure_name': metric+'_'+name, 'y':history,'title':'model_{} '.format(name) + metric,
                  'ylabel':metric,'xlabel':'epoch',"line_att":dict(linestyle=linestyles[0],color=self.colors[color_num]),
                  'save_file': self.save_path + 'model_{}_'.format(name) + metric + '.png'}
        generic_plot(params)

    #TODO : add creation in main;add in model return metric and name "cc"
