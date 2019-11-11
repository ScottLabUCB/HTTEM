import numpy as np
import skimage.io as skio
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def binarize_map(y_predOG,threshold):
    y_pred = y_predOG.copy()
    if y_pred.flags.owndata != True:
        raise RuntimeError('Data copy did not work - stopped before data overwrite occurred')
    for y in y_pred:
        y[y > threshold] = 1
        y[y <= threshold] = 0
    return y_pred


def tabulate_metrics(y_pred,y_true):
    """Returns five lists containing the zero_one_loss, balanced_accuracy_score
    precision, recall, and f1 score for each segmentation map that has been
    input. Inputs are the predicted maps and true maps."""
    zol_list = []
    bas_list = []
    prcsn_list = []
    rcll_list = []
    f1_list = []
    for idx, seg in enumerate(y_true):
        zol = metrics.zero_one_loss(seg.flatten(),y_pred[idx,:,:,:].flatten()) #fix INDEXING KATE NEED TO DETERMINE INPUT shape
        zol_list.append(zol)
        bas = metrics.accuracy_score(seg.flatten(),y_pred[idx,:,:,:].flatten())
        bas_list.append(bas)
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(seg.flatten(),y_pred[idx,:,:,:].flatten())
        prcsn_list.append(precision)
        rcll_list.append(recall)
        f1_list.append(f1)
    return zol_list,bas_list,prcsn_list,rcll_list,f1_list


def containsParticles(maps):
    """Function takes in segmentation maps in Keras tensor format and returns a list containing whether
    each map contained a particle written as a boolean"""
    particlelist = []
    count = 0
    for segmap in maps:
        if 1.0 in segmap:
            particlelist.append(True)
            count += 1
        else:
            particlelist.append(False)
    print('Fraction of maps which contained particles: {0}'.format(count/maps.shape[0]))
    return particlelist

def finalDataframe(predictions,imgtensor,maptensor,defocusDF, threshold = 0.6):
    """Create final data frame for a given set of input images.
    Inputs are the predictions made by the ml model, the input images in a keras tensor,
    the answer key maps tensor, and the defocus dataframe.
    Threshold sets the probability which must be achieved for the particle class"""
    expmntlist = experimentLabelsList()
    metricDF = expmntListDF(expmntlist)
    predictions = binarize_map(predictions,threshold)
    metrics = tabulate_metrics(predictions,maptensor)
    intMetrics = intensityMetrics(imgtensor)
    particlelist = containsParticles(maptensor)
    metricDF['zero-one-loss'] = metrics[0]
    metricDF['accuracy'] = metrics[1]
    metricDF['precision'] = metrics[2]
    metricDF['recall'] = metrics[3]
    metricDF['f1'] = metrics[4]
    metricDF['max'] = intMetrics[0]
    metricDF['min'] = intMetrics[1]
    metricDF['average']= intMetrics[2]
    metricDF['contains particles']= particlelist
    DF = pd.merge(defocusDF,metricDF,on='keys',how='inner',sort=True,validate='1:m')
    return DF

def find_thresh(maps,predicted,thresh = 0.7,step = 0.05,layer = 1,cutoff=0.0001):
    predBin = binarize_map(predicted[:,:,:,layer],thresh)
    acc = metrics.accuracy_score(maps[:,:,:,layer].flatten(),predBin.flatten())
    dif = 1
    loops = 30
    predBin = binarize_map(predicted[:,:,:,layer],thresh+step)
    up = metrics.accuracy_score(maps[:,:,:,layer].flatten(),predBin.flatten())
    predBin = binarize_map(predicted[:,:,:,layer],thresh-step)
    down = metrics.accuracy_score(maps[:,:,:,layer].flatten(),predBin.flatten())
    if round(up*1000) >= round(acc*1000):
        thresh += step
        print('threshold going up')
        while dif >= cutoff and loops > 0:
            if loops != 30:
                thresh += step
            if thresh >= 1:
                break
            predBin = binarize_map(predicted[:,:,:,layer],thresh)
            acc2 = metrics.accuracy_score(maps[:,:,:,layer].flatten(),predBin.flatten())
            dif = acc2 - acc
            loops -= 1
            if dif >= cutoff:
                acc = acc2
            print(acc2,thresh)
        thresh -= step
    elif round(down*1000) >= round(acc*1000):
        thresh -= step
        print('threshold going down')
        while dif > cutoff and loops > 0:
            if loops != 30:
                thresh -= step
            if thresh >= 1:
                print('break')
                break
            predBin = binarize_map(predicted[:,:,:,layer],thresh)
            acc2 = metrics.accuracy_score(maps[:,:,:,layer].flatten(),predBin.flatten())
            dif = acc2 - acc
            loops -= 1
            if dif > cutoff:
                acc = acc2
            print(acc2,thresh)
        thresh += step
    else:
        pass
    return(acc,thresh)

def plot_confusion_matrix(Y, Y_pred, Y_labels,save=False):
    """Creates a confusion matrix for the different classes given a true labels,predictions, the dataset
    and the desired trained classfier"""
    cfm = metrics.confusion_matrix(Y, Y_pred)
    cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
    df_cfm = pd.DataFrame(data = cfm, columns=Y_labels, index=Y_labels)
    plt.subplots(figsize=(5,5))
    ax = sns.heatmap(df_cfm, annot=True,cmap='rainbow')
    ax.set(ylabel='True label',xlabel='Predicted label')
    if save == True:
        fname = input('Specify filename to save figure to: ')
        fig = ax.get_figure()
        fig.savefig(fname)
