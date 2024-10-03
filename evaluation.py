import numpy as np
import torch
from sklearn.metrics import brier_score_loss
from utils import logits_labels_from_dataloader


def ECE_calc(samples_certainties, num_bins=15, adaptive=False):
    '''https://github.com/idogalil/benchmarking-uncertainty-estimation-performance'''
    indices_sorting_by_confidence = torch.argsort(samples_certainties[:, 0], descending=False)  # Notice the reverse sorting
    samples_certainties = samples_certainties[indices_sorting_by_confidence]
    samples_certainties = samples_certainties.transpose(0, 1)
    if adaptive:
        certainties = samples_certainties[0]
        N = certainties.shape[0]
        step_size = int(N / (num_bins - 1))
        bin_boundaries = [certainties[i].item() for i in range(0, certainties.shape[0], step_size)]
        bin_boundaries[0] = 0
        bin_boundaries[-1] = certainties[-1]
        bin_boundaries = torch.tensor(bin_boundaries)
    else:
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)        
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bins_accumulated_error = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        bin_indices = torch.logical_and(samples_certainties[0] <= bin_upper, samples_certainties[0] > bin_lower)
        if bin_indices.sum() == 0:
            continue  # This is an empty bin
        bin_confidences = samples_certainties[0][bin_indices]
        bin_accuracies = samples_certainties[1][bin_indices]
        bin_avg_confidence = bin_confidences.mean()
        bin_avg_accuracy = bin_accuracies.mean()
        bin_error = torch.abs(bin_avg_confidence - bin_avg_accuracy)
        bins_accumulated_error += bin_error * bin_confidences.shape[0]

    expected_calibration_error = bins_accumulated_error / samples_certainties.shape[1]
    return expected_calibration_error


def AUROC(samples_certainties, sort=True):
    '''https://github.com/idogalil/benchmarking-uncertainty-estimation-performance'''
    # Calculating AUROC in a similar way gamma correlation is calculated. The function can easily return both.
    if sort:
        indices_sorting_by_confidence = torch.argsort(samples_certainties[:, 0], descending=True)
        samples_certainties = samples_certainties[indices_sorting_by_confidence]
    total_samples = samples_certainties.shape[0]
    incorrect_after_i = np.zeros((total_samples))

    for i in range(total_samples - 1, -1, -1):
        if i == total_samples - 1:
            incorrect_after_i[i] = 0
        else:
            incorrect_after_i[i] = incorrect_after_i[i+1] + (1 - int(samples_certainties[i+1][1]))
            # Note: samples_certainties[i+1][1] is the correctness label for sample i+1

    n_d = 0  # amount of different pairs of ordering
    n_s = 0  # amount of pairs with same ordering
    incorrect_before_i = 0
    for i in range(total_samples):
        if i != 0:
            incorrect_before_i += (1 - int(samples_certainties[i-1][1]))
        if samples_certainties[i][1]:
            # if i is a correct prediction, i's ranking 'agrees' with all the incorrect that are to come
            n_s += incorrect_after_i[i]
            # i's ranking 'disagrees' with all incorrect predictions that preceed it (i.e., ranked more confident)
            n_d += incorrect_before_i
        else:
            # else i is an incorrect prediction, so i's ranking 'disagrees' with all the correct predictions after
            n_d += (total_samples - i - 1) - incorrect_after_i[i]  # (total_samples - i - 1) = all samples after i
            # i's ranking 'agrees' with all correct predictions that preceed it (i.e., ranked more confident)
            n_s += i - incorrect_before_i

    return n_s / (n_s + n_d)


def metrics_from_dataloader(model, dataloader):
    logits, labels = logits_labels_from_dataloader(model, dataloader)
    metrics = metrics_from_logits_labels(logits, labels)    
    return metrics

def metrics_from_logits_labels(logits, labels):
    probs = torch.softmax(logits, dim=1)
    metrics = metrics_from_probas_labels(probs, labels)
    return metrics

def metrics_from_probas_labels(probs, labels):
    certainties, y_pred = probs.max(axis=1)
    correct = (y_pred == labels)
    metrics = metrics_from_certainties_correct(certainties, correct)
    return metrics

def metrics_from_certainties_correct(certainties, correct):
    metrics = {}
    samples_certainties = torch.stack((certainties.cpu(), correct.cpu()), dim=1)
    metrics['ECE_15'] = ECE_calc(samples_certainties, num_bins=15).item()
    metrics['AdaECE_15'] = ECE_calc(samples_certainties, num_bins=15, adaptive=True).item()
    metrics['Accuracy'] = (samples_certainties[:,1].sum() / samples_certainties.shape[0]).item() * 100
    metrics['Average_Confidence'] = certainties.mean().item()
    metrics['AUROC'] = AUROC(samples_certainties)    
    metrics['Brier_top'] = brier_score_loss(correct.numpy(), certainties.numpy())
    return metrics


def eval_binary_method(method, model, logits_test, labels_test):
    probs = torch.softmax(logits_test, axis=1)
    if 'tva' in method:
        certainties, y_pred = probs.max(axis=1)
        if 'Beta' in method:
            certainties_scaled = torch.tensor(model.transform(certainties, num_samples=20)).mean(0) # careful with OOM for high num_samples
        else:
            certainties_scaled = torch.tensor(model.transform(certainties))
        correct = (y_pred == labels_test)
        metrics_test = metrics_from_certainties_correct(certainties_scaled, correct)
    else:
        if 'Beta' in method:
            probs_scaled = torch.tensor(model.transform(probs, num_samples=20)).mean(0) # careful with OOM for high num_samples
        else:
            probs_scaled = torch.tensor(model.transform(probs))
        metrics_test = metrics_from_probas_labels(probs_scaled, labels_test)
        
    return metrics_test