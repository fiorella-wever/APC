import math
import numpy as np
from numpy import ma
import pandas as pd
import seaborn as sns
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import metrics
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.utils import shuffle
from collections import defaultdict

def make_classes_balanced(data):
    # let's make sure the classes are balanced
    train_class_count = Counter(data["y_classes"])
    min_class = min(train_class_count, key=train_class_count.get)
    n_min_values = train_class_count[min_class]
    n_classes = len(train_class_count)

    final_idx = []
    for c in range(n_classes):
        idx_to_keep = list(np.random.choice(np.where(data["y_classes"] == c)[0], n_min_values, replace=False))
        final_idx += idx_to_keep

    final_idx = shuffle(final_idx)
    X_train = data["X_train"][final_idx]
    X_aux_train = data["X_aux"][final_idx]
    y_train = data["y"][final_idx]
    
    return X_train, X_aux_train, y_train

def get_class_weights(class_balance_dict):    
    # Scaling by total/n_classes helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    n_classes = len(class_balance_dict)
    class_weights = dict()
    for i in range(n_classes):
        frac = class_balance_dict[i]
        class_weights[i] = (1 / frac)/n_classes

    return class_weights

def get_OS_class_weights(min_class_percentage, n_classes):    
    # when oversampling the minority class to a certain percentage
    maj_class_percentage = 100 - (n_classes - 1)*min_class_percentage
    class_weights = dict()
    for i in range(n_classes):
        if i == 0:
            class_weights[i] = (1 / maj_class_percentage * 100)/n_classes
        else:
            class_weights[i] = (1 / min_class_percentage * 100)/n_classes

    return class_weights

def show_learning_plots(history, model_type, dataset):
    sns.set_style("white")    
    # plot loss curve
    plt.figure(figsize=(10,8))
    plt.plot(history.history['loss'], label="Train")
    plt.plot(history.history['val_loss'], label="Validation")
    plt.title('{} Model Loss'.format(model_type))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.savefig('plots/loss_curves_{}_{}.png'.format(model_type, dataset), bbox_inches='tight')

    plt.show()
    plt.close()
    
def plot_confusion_matrix(cnf_matrix, model_type, class_names, dataset):
    # Physionet: class_names=["0", "1"] 
    # Clue: class_names=["ON", "OFF", "OTHER-H", "OTHER-NH"]
    fig, ax = plt.subplots(figsize=(10,8))
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    
    ax.xaxis.set_ticklabels(class_names); ax.yaxis.set_ticklabels(class_names);
        
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.ylabel('Actual label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.savefig('plots/confusion_matrix_{}_{}.png'.format(model_type, dataset), bbox_inches='tight')
    plt.show()
    plt.close()
    
def auprc(y_true, y_pred):
    ytrue = tf.math.argmax(y_true, 1)
    return tf.py_function(average_precision_score, (ytrue, y_pred[:,1]), tf.double)

def masked_mse(y_true, y_pred):
    y_true_val = y_true[:,:,::2]
    y_true_mask = 1 - y_true[:,:,1::2]
    y_pred_val = y_pred[:,:,::2]
    abs_loss = K.square(y_pred_val - y_true_val) * y_true_mask
    final_loss = K.sum(abs_loss) / K.sum(y_true_mask) #(batch_size,)   
    
    return final_loss

def get_results_df(y_test, y_pred, y_test_p, y_pred_p, model, n_classes):
    results_dict = defaultdict()
    f1_scores_per_class = metrics.f1_score(y_test, y_pred, average=None).tolist()
    results_dict["weighted_f1_score"] = f1_score(y_test, y_pred, average='weighted')
    results_dict["weighted_precision"], results_dict["weighted_recall"], _, _ = metrics.precision_recall_fscore_support(y_test, y_pred, average='weighted')
    for c in range(n_classes):
        col_name = "f1_" + str(c)
        results_dict[col_name] = f1_scores_per_class[c]
        if n_classes == 2:
            results_dict["roc_auc"] = roc_auc_score(y_test, y_pred_p[:,1])
            results_dict["prc_auc"] = average_precision_score(y_test, y_pred_p[:,1])
    
    results_df = pd.DataFrame(results_dict, index=[0])
    results_df["model"] = model
    
    return results_df

def convert_to_tensor(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return arg

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0):

    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)

def oversample_batch_generator_GRU(data, class_balance, epochs, batch_size):
    n_samples = data["X_train"].shape[0]
    majority_class = max(range(len(class_balance)), key=lambda i: class_balance[i])
    minority_class = min(range(len(class_balance)), key=lambda i: class_balance[i])
    classes = [0, 1]
    
    n_majority = class_balance[majority_class] * n_samples
    n_minority = class_balance[minority_class] * n_samples
    
    pos_idx = np.where(data["y_classes"] == 1)[0]
    neg_idx = np.where(data["y_classes"] == 0)[0]
    
    # One epoch should at least contain all negative examples or max each instance of the minority class 3 times
    steps_per_epoch = min(math.ceil(2 * n_majority / batch_size), math.ceil(3 * 2 * n_minority / batch_size))
    indexes = [0 for _ in range(2)]
    #half the batch
    n_batch_samples = int(np.round(batch_size * 0.5))
    
    while True:
        for batch in range(steps_per_epoch):
            #shuffle the idx
            pos_idx = shuffle(pos_idx)
            batch_pos_idx = pos_idx[:n_batch_samples]
            neg_idx = shuffle(neg_idx)
            batch_neg_idx = neg_idx[:n_batch_samples]
            batch_idx = batch_pos_idx + batch_neg_idx
            #shuffle positive and negative classes
            batch_idx = shuffle(batch_idx)
            
            x_batch = data["X_train"][batch_idx]
            x_aux_batch = data["X_aux"][batch_idx]
            y_batch = data["y"][batch_idx]

            yield ([x_batch, x_aux_batch], [y_batch])
            
def batch_generator_GRU(data, batch_size):
    while True:
        num_batches = int(np.ceil(data["X_val"].shape[0] / batch_size))
        for i in range(num_batches):
            start_idx = i*batch_size
            end_idx = (i+1)*batch_size
            x_batch = data["X_val"][start_idx:end_idx]
            x_aux_batch = data["X_aux"][start_idx:end_idx]
            y_batch = data["y"][start_idx:end_idx]
            yield ([x_batch, x_aux_batch], [y_batch])
            
def batch_generator_GRUD(data, batch_size):
    while True:
        num_batches = int(np.ceil(data["X"].shape[0]/ batch_size))
        for batch_idx in range(num_batches):
            start_idx = batch_idx*batch_size
            end_idx = (batch_idx+1)*batch_size
            x_values = data["X"][start_idx: end_idx]
            x_mask = data["M"][start_idx: end_idx]
            x_times = data["GRU_times"][start_idx:end_idx]
            x_lengths = [data["GRU_lengths"]][start_idx:end_idx]            

            max_len = data["GRU_lengths"].max()
            x_values = [pad_along_axis(x, max_len, axis=0) for x in x_values]
            x_mask = [pad_along_axis(x, max_len, axis=0) for x in x_mask.tolist()]
            x_times = [pad_along_axis(np.array(x), max_len, axis=0) for x in x_times.tolist()]
            x_times = np.expand_dims(np.array(x_times), axis=2)
            x_aux_batch = data["X_aux"][start_idx:end_idx]
            #to make sure it doesn't go out of index --> len - 1 would be the index position
            x_lengths = [[x - 1] for x in x_lengths]

            inputs = [convert_to_tensor(x_values), convert_to_tensor(x_mask), convert_to_tensor(x_times), convert_to_tensor(x_aux_batch), convert_to_tensor(x_lengths)]
            y_batch = data["y"][start_idx:end_idx]

            yield (inputs, [y_batch])

def APC_batch_generator(data, time_shift, batch_size):
    if (data["X"].shape[0] != data["y"].shape[0]) or (data["X"].shape[0] != data["X_aux"].shape[0]):
        raise ValueError('Args `X`, `x_aux` and `y` must have the same length.')
    if len(.shape) != 2:
        raise ValueError(
            'Arg `y` must have a shape of (num_samples, num_classes). ' +
            'You can use `keras.utils.to_categorical` to convert a class vector ' +
            'to a binary class matrix.'
        )
    if batch_size < 1:
        raise ValueError('Arg `batch_size` must be a positive integer.')
    while True:    
        num_batches = int(np.ceil(data["X"].shape[0] / batch_size))
        for batch_idx in range(num_batches):
            start_idx = batch_idx*batch_size
            end_idx = (batch_idx+1)*batch_size
            x_batch = data["X"][start_idx:end_idx]
            if time_shift == 0:
                x_batch = x_batch
                y_1_batch = x_batch
            else:
                x_batch = x_batch[:, :-time_shift, :]
                y_1_batch = x_batch[:, time_shift:, :]
            x_aux_batch = data["X_aux"][start_idx:end_idx]
            y_batch = data["y"][start_idx:end_idx]
            yield ([x_batch, x_aux_batch], [y_1_batch, y_batch])
            
def APC_GRUD_batch_generator(data, time_shift, batch_size):
    while True:
        num_batches = int(np.ceil(data["X"].shape[0]/ batch_size))
        for batch_idx in range(num_batches):
            start_idx = batch_idx*batch_size
            end_idx = (batch_idx+1)*batch_size
            x_values_ = data["X"][start_idx:end_idx]
            x_mask = data["M"][start_idx:end_idx]
            x_times = data["GRU_times"][start_idx:end_idx]
            x_lengths = data["GRU_lengths"][start_idx:end_idx]
            x_aux_batch = data["X_aux"][start_idx:end_idx]
            y_batch = data["y"][start_idx:end_idx]
            
            max_len = x_lengths.max() 
            x_lengths = [[x - 2] for x in x_lengths]
            
            if time_shift == 0:
                x_values = x_values_
                y_1_batch = x_values
                x_mask = x_mask.tolist()
                x_times = x_times.tolist()
            else:
                x_values = [x[:-time_shift, :] for x in x_values_]
                y_1_batch = [x[time_shift:, :] for x in x_values_]
                x_mask = [x[:-time_shift, :] for x in x_mask.tolist()]
                x_times = [x[:-time_shift] for x in x_times.tolist()]
                
            x_values = [pad_along_axis(x, max_len, axis=0) for x in x_values]
            y_1_batch = [pad_along_axis(y, max_len, axis=0) for y in y_1_batch]
            x_mask = [pad_along_axis(x, max_len, axis=0) for x in x_mask]
            x_times = [pad_along_axis(np.array(x), max_len, axis=0) for x in x_times]
            x_times = np.expand_dims(np.array(x_times), axis=2)
            
            inputs = [convert_to_tensor(x_values), convert_to_tensor(x_mask), convert_to_tensor(x_times), convert_to_tensor(x_aux_batch), convert_to_tensor(x_lengths)]
            
            yield (inputs, [convert_to_tensor(y_1_batch), y_batch])