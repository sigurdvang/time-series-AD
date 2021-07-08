import more_itertools as mit
import numpy as np

"""
This file contains code used for anomaly score thresholding and anomaly sequence identifying. 
The methods in it are based on the research of Hundman et al. which can be found in here: 
https://arxiv.org/abs/1802.04431.

The code is also based on the papers companion repo, which can be found here: 
https://github.com/khundman/telemanom
"""

def find_epsilon(errors):
    """
    Method that finds threshold of anomaly scores
    args:
        erros: list of anomaly scores
    returns:
        anomaly threshold
    """
    e_s = errors

    max_score = -10000000
    mean_e_s = np.mean(e_s)
    sd_e_s = np.std(e_s)
    
    sd_threshold = None 
    # if one cannot find a suitable epsilon, the first one should be used
    best_epsilon =  mean_e_s + sd_e_s * 2.5
    
    for z in np.arange(2.5, 12, 0.5):
        epsilon = mean_e_s + sd_e_s * z
        pruned_e_s = e_s[e_s < epsilon]

        i_anom = np.argwhere(e_s >= epsilon).reshape(-1,)
        buffer = np.arange(1, 50)
        i_anom = np.sort(np.concatenate((i_anom,
                                        np.array([i+buffer for i in i_anom])
                                         .flatten(),
                                        np.array([i-buffer for i in i_anom])
                                         .flatten())))
        i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
        i_anom = np.sort(np.unique(i_anom))

        if len(i_anom) > 0:
            groups = [list(group) for group in mit.consecutive_groups(i_anom)]
            E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]


            mean_perc_decrease = (mean_e_s - np.mean(pruned_e_s)) / mean_e_s
            sd_perc_decrease = (sd_e_s - np.std(pruned_e_s)) / sd_e_s
            score = (mean_perc_decrease + sd_perc_decrease) #/ (len(E_seq) ** 2 + len(i_anom))

            # sanity checks / guardrails
            if score >= max_score and len(i_anom) < (len(e_s) * 0.5):
                max_score = score
                sd_threshold = z
                best_epsilon = epsilon
                
    return best_epsilon

def partition_anomalous_subsequences(scores, gt_anomalies, threshold, advance=None):
    """
    Finds anomalous subsequences in anomaly scores, using given threshold
    args:
        scores: list of anomaly scores
        gt_anomalies: ground truth anomalies, if there are any
        advance: window one can look in advance
    """
    
    def get_label(pos):
        for i in range(len(gt_anomalies['start'])):
            start_i, end_i = gt_anomalies['start'][i], gt_anomalies['end'][i]
            if start_i <= pos <= end_i:
                return True
        return False
    
    scores = np.asarray(scores)
    labels = [get_label(i) for i in range(len(scores))]
    labels = np.asarray(labels)
    predicts = scores > threshold
    
    actual = labels > 0.1
    anomaly_state = False
    anomaly_count = 0
    latency = 0
    
    # Added advance in case model predicts anomaly 'in advance' within a small window
    # Advance should be small
    if advance is not None:
        for i in range(len(scores)):
            if any(actual[max(i - advance, 0) : i + 1]) and predicts[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predicts[j]:
                            predicts[j] = True
                            latency += 1
            elif not actual[i]:
                anomaly_state = False
            if anomaly_state:
                predicts[i] = True
    
    return predicts, labels
    

def calc_pointwise_metrics(predict, actual):
    """
    calculate evaluation metrics predict and actual.
    Args:
            predict (np.ndarray): the predict label
            actual (np.ndarray): np.ndarray
    returns f1, precision, recall, TP, TN, FP, FN
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN