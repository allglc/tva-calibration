import torch
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from uncertainty_metrics import AURC_calc, AUROC, ECE_calc, calc_adaptive_bin_size

from utils import logits_labels_from_dataloader


def metrics_from_probas_labels(probs, labels):
    
    # compute uncertainty metrics
    certainties, y_pred = probs.max(axis=1)
    correct = (y_pred == labels)
    metrics = metrics_from_certainties_correct(certainties, correct)

    return metrics


def metrics_from_certainties_correct(certainties, correct):
    metrics = {}
    samples_certainties = torch.stack((certainties.cpu(), correct.cpu()), dim=1)
    
    # compute uncertainty metrics
    metrics['ECE_15'] = ECE_calc(samples_certainties, num_bins=15)[0].item()
    metrics['ECE_100'] = ECE_calc(samples_certainties, num_bins=100)[0].item()
    metrics['AdaECE_15'] = ECE_calc(samples_certainties, num_bins=15, bin_boundaries_scheme=calc_adaptive_bin_size)[0].item()
    metrics['AdaECE_100'] = ECE_calc(samples_certainties, num_bins=100, bin_boundaries_scheme=calc_adaptive_bin_size)[0].item()
    metrics['Accuracy'] = (samples_certainties[:,1].sum() / samples_certainties.shape[0]).item() * 100
    metrics['AUROC'] = AUROC(samples_certainties)
    metrics['AUROC_sklearn'] = roc_auc_score(correct.numpy(), certainties.numpy())
    metrics['AURC'] = AURC_calc(samples_certainties)
    metrics['Brier_top'] = brier_score_loss(correct.numpy(), certainties.numpy())
    metrics['BCE'] = log_loss(correct.numpy(), certainties.numpy())
    metrics['Average_Confidence'] = certainties.mean().item()

    return metrics


def metrics_from_logits_labels(logits, labels):
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)
    metrics = metrics_from_probas_labels(probs, labels)
    return metrics
    
    
def metrics_from_dataloader(model, dataloader):
    logits, labels = logits_labels_from_dataloader(model, dataloader)
    metrics = metrics_from_logits_labels(logits, labels)    
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