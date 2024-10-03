import numpy as np
import torch
from torch import nn

import imax_calib.calibration as imax_calibration
import imax_calib.utils as imax_utils


class ModelWithTemperatureOriginal(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperatureOriginal, self).__init__()
        self.model = model.eval()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Next: optimize the temperature w.r.t. NLL
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=5000)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        return self
    
    
class Patel2021():
    def __init__(self, setting, num_classes, method, seed=928163):
        
        self.cfg = dict(
            # All
            cal_setting=setting,  # CW, sCW or top1
            num_bins=15,
            n_classes=num_classes,
            # Binning
            Q_method=method, # imax, eqsize or eqmass
            Q_binning_stage="raw",  # bin the raw logodds or the 'scaled' logodds
            Q_binning_repr_scheme="sample_based",
            Q_bin_repr_during_optim="pred_prob_based",
            Q_rnd_seed=seed,
            Q_init_mode="kmeans")
        
    def fit(self, logits_val, labels_val):
        valid_logits = logits_val.numpy()
        valid_labels = nn.functional.one_hot(labels_val).numpy()
        valid_probs = imax_utils.to_softmax(valid_logits)
        valid_logodds = imax_utils.quick_logits_to_logodds(valid_logits, probs=valid_probs)

        self.calibrator_obj = imax_calibration.learn_calibrator(self.cfg, logits=valid_logits, logodds=valid_logodds, y=valid_labels)
        
        
    def predict(self, logits_test):
        test_logits = logits_test.numpy()
        test_probs = imax_utils.to_softmax(test_logits)
        test_logodds = imax_utils.quick_logits_to_logodds(test_logits, probs=test_probs)
        
        cal_logits, cal_logodds, cal_probs, assigned = self.calibrator_obj(test_logits, test_logodds)
        if self.cfg['cal_setting'] == 'top1':
            certainties_scaled = torch.tensor(cal_probs)
            _, y_pred = logits_test.max(axis=1) # y_pred is the same as previously
            return certainties_scaled, y_pred
        else:
            probs = torch.tensor(cal_probs)
            return probs
            
        
