from multi_predictors_combined import  check_params_model,n_experts_combined_model_b
from load_data import get_fold_samples,load_medical_data,split_data
from sklearn.model_selection import KFold
##parameters
current_params = {'optimizer': 'adam', 'nn_gate': 3, 'w_init': 'glorot_normal', 'nn_layer2': 15, 'nn_layer1': 10,
                  'epoch_num': 500, 'dropout2': 0, 'dropout1': 0.2}
kfold_params_dict = {'n_splits': 10, 'shuffle': False, 'random_state': 42}
w_dir ='/home/yaniv/PycharmProjects/memograph data/runs/29_10_2017_16_51/82_CC_acc_0.7038_f1_0.6469-x-_SIMPLE_acc_0.7192_f1_0.6437-x-_MLO_acc_0.6808_f1_0.6103-x-_MOE_acc_0.7462_f1_0.7003-x-/'
n = 2
n_splits = 10
is_multi_expert = True
model_name = 'MOE'
## data
data_view = ['CC', 'MLO']
data_type = ['Dense', 'Fatty']
type_v = 1
data, target = load_medical_data(data_type[type_v])
feature_number = data[data_view[0]].shape[1]
kf = KFold(**kfold_params_dict)
combined_split_indexes = split_data(data,target=target ,is_multi_expert=True, kfold=kf)

model  = check_params_model((feature_number,),current_params,n)
for i in range(n_splits):
    x_test, y_test = get_fold_samples(data, target, combined_split_indexes, is_multi_expert=is_multi_expert,
                                      fold=i, is_train=False)
    model.load_weights(w_dir + "model_{}_fold_{}.h5".format(model_name, i))
    pred = model.predict(x_test)
    res = zip(*pred)
    print "fold {}".format(i)
    for j in range(len(y_test)):
        print " sample {} target {} output {:.2f} gate {} exp1 {:.2f} exp2 {:.2f}".format(j,y_test[j],float(res[j][0]),res[j][1],float(res[j][2]),float(res[j][3]))


