import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.utils import shuffle
from copy import deepcopy
import itertools

from utils import get_model_response, expected_calibration_error, random_sampling, params_check
from data_utils import load_dataset_custom

sys.path.append(os.path.abspath('../'))
from calibrators import fit_binary_method, fit_scaling_model


def get_p_content_free(params, train_sentences, train_labels, test_labels, content_free_inputs=('N/A')):
    """Query model with content free input, return its prediction probability for each label"""

    _, all_p_y = get_model_response(params, None, None, train_sentences, train_labels, None, None, content_free_inputs, test_labels, normalize=False)

    p_y = np.mean(np.array(all_p_y), axis=0)
    p_y = p_y / np.sum(p_y)  # normalize
    return p_y


def convert_to_list(items, is_int=False):
    if is_int:
        return [int(s.strip()) for s in items.split(",")]
    else:
        return [s.strip() for s in items.split(",")]


def get_data(params):
    val_size = params['val_size']
    calib_size = params['calib_size']
    curr_seed = params['val_seed']
    
    freeze_test_set = True

    ### load data
    all_train_sentences, all_train_labels, all_test_sentences, all_test_labels, all_val_sentences, all_val_labels = load_dataset_custom(params)

    ### sample test set
    if params['subsample_test_set'] is None:
        test_sentences, test_labels = all_test_sentences, all_test_labels
        print(f"selecting full test set ({len(all_test_labels)} examples)")
    else:
        if freeze_test_set:
            np.random.seed(0)  # always use seed 0 result if freeze
        else:
            np.random.seed(params['seed'])
        test_sentences, test_labels = random_sampling(all_test_sentences, all_test_labels, params['subsample_test_set'])
        print(f"selecting {len(test_labels)} subsample of test set")

    ### sample few-shot training examples
    # np.random.seed(params['seed'])
    np.random.seed(curr_seed)
    train_sentences, train_labels = random_sampling(all_train_sentences, all_train_labels, params['num_shots'])
    
    ## sample validation set and calibration set
    np.random.seed(params['seed'])
    val_sentences, val_labels = random_sampling(all_val_sentences, all_val_labels, val_size+calib_size)
    calib_sentences, calib_labels = val_sentences[val_size:], val_labels[val_size:]
    val_sentences, val_labels = val_sentences[:val_size], val_labels[:val_size]
    print(f"selecting {len(val_labels)} subsample of validation set for LinC")
    print(f"selecting {len(calib_labels)} subsample of validation set for calibration")

    
    ### Evaluate the performance and save all results
    # obtaining model's response on test examples
    print(f"getting raw resp for val, calib and test sentences")

    params_check(params, all_train_sentences, all_train_labels, train_sentences, train_labels, val_sentences,
                    val_labels, test_sentences, test_labels)
    # get probas
    _, test_probas = get_model_response(params, None, None, train_sentences, train_labels, None, None, test_sentences, test_labels)
    _, val_probas = get_model_response(params, None, None, train_sentences, train_labels, None, None, val_sentences, val_labels)
    _, calib_probas = get_model_response(params, None, None, train_sentences, train_labels, None, None, calib_sentences, calib_labels)
    
    # calculate P_cf
    p_cf = get_p_content_free(params, train_sentences, train_labels, test_labels, content_free_inputs=["N/A", "", "[MASK]"])  ##type: numpy array e.g. [0.13829783 0.86170214] for SST2
    
    return val_probas, val_labels, calib_probas, calib_labels, test_probas, test_labels, p_cf


def learn_LinC(val_probas, val_labels, W_ConC, num_classes, epochs, lr, tva):
    
    W = Variable(W_ConC, requires_grad=True)
    b = Variable(torch.zeros([num_classes, 1]), requires_grad=True)
        
    if tva:
        diagonal_mask = torch.eye(W_ConC.size(0), W_ConC.size(1), dtype=torch.bool)
        off_diagonal_mask = ~diagonal_mask
        off_diag_reg, bias_reg, diag_reg= 1, 1, 0.1
            
    optimizer = torch.optim.SGD([W, b], lr=lr)

    for epoch in range(epochs):
        val_probas, val_labels = shuffle(val_probas, val_labels, random_state=0)

        for probas, label in zip(val_probas, val_labels):
            optimizer.zero_grad()
            probas = torch.tensor(probas) / torch.sum(torch.tensor(probas))  # normalize to 1 (should already be the case up to a tolerance)
            calibrated_probas = torch.matmul(W.float(), torch.unsqueeze(probas, dim=-1).float()) + b.float()
            if tva:
                calibrated_probas = torch.softmax(calibrated_probas.reshape(1, len(calibrated_probas)), axis=1) # BCE takes probas, not "logits"
                confidence, y_pred = torch.max(calibrated_probas, axis=1)
                correct = (y_pred == label).float()
                loss = nn.functional.binary_cross_entropy(confidence, correct)
                # Regularization
                loss += off_diag_reg * (off_diagonal_mask * W).square().sum()
                loss += bias_reg * b.square().sum()
                loss += diag_reg * (diagonal_mask * (W - W_ConC)).square().mean()
            else:
                loss = nn.functional.cross_entropy(calibrated_probas.reshape(1, len(calibrated_probas)),
                                torch.tensor(label).reshape(1))
            loss.backward()
            if not (torch.isnan(W.grad).any() or torch.isnan(b.grad).any()):
                optimizer.step()
    return W.detach(), b.detach()


def calibrate_LinC(probas, W, b):
    probas_calib = []
    for p in probas:
        p = torch.tensor(p) / torch.sum(torch.tensor(p))  # normalize to 1 (should already be the case up to a tolerance)
        calibrated_probas = torch.matmul(W.float(), torch.unsqueeze(p, dim=-1).float()) + b.float()
        calibrated_probas = torch.softmax(calibrated_probas, axis=0)
        probas_calib.append(calibrated_probas[:,0].tolist())
    probas_calib = np.array(probas_calib)
    return probas_calib


class LogitsDataset(torch.utils.data.Dataset):
    def __init__(self, logits, labels):
        super(LogitsDataset, self).__init__()
        self.logits = logits
        self.labels = labels
    def __len__(self):
        return self.logits.shape[0]

    def __getitem__(self, index):
        logits = self.logits[index, :]
        labels = self.labels[index]
        return logits, labels
    
    
def post_hoc_calib_learn_transform(method, calib_probas, calib_labels, test_probas, num_classes):
    calib_probas, calib_labels, test_probas = torch.tensor(calib_probas, dtype=torch.float32), torch.tensor(calib_labels), torch.tensor(test_probas, dtype=torch.float32)
    # binary methods
    if 'netcal' in method:
        model = fit_binary_method(method, calib_probas, calib_labels, num_classes=num_classes, run_softmax=False)
        if 'tva' in method:
            certainties, y_pred = test_probas.max(axis=1)
            certainties_scaled = torch.tensor(model.transform(certainties))
            # hack to transform confidences into probas to compute ECE with their code
            probs_scaled = torch.zeros(test_probas.shape[0], num_classes)
            probs_scaled[np.arange(probs_scaled.shape[0]), y_pred] = certainties_scaled.float() + 1e-6 # add epsilon to ensure correct argmax when confidence=0
        else:
            probs_scaled = torch.tensor(model.transform(test_probas))
    # apply HBtva on top of VS
    elif method == 'VS_HBtva':
        # learn VS
        logits = torch.log(calib_probas) # get "logits"
        dataset_logits_calib = LogitsDataset(logits, calib_labels)
        dataloader_logits_calib = DataLoader(dataset_logits_calib, batch_size=1)
        model_VS = fit_scaling_model('vector', dataloader_logits_calib, num_classes, binary_loss=False, regularization=False, temperature_ref=1, num_epochs=200)
        probs_scaled = model_VS(torch.log(test_probas).cuda()).detach().cpu()
        probs_scaled = torch.softmax(probs_scaled, dim=1) # scaling methods output logits, need to convert to probas
        # learn HB on top of VS
        calib_logits_VS = model_VS(torch.log(calib_probas).cuda()).detach().cpu()
        model_hbtva = fit_binary_method('netcal_HB_tva_eqsize', calib_logits_VS, calib_labels, num_classes=num_classes, run_softmax=True) # scaling methods output logits, need run_softmax
        # apply HB on top of VS
        certainties, y_pred = probs_scaled.max(axis=1)
        certainties_scaled = torch.tensor(model_hbtva.transform(certainties))
        # hack to transform confidences into probas to compute ECE with their code
        probs_scaled = torch.zeros(test_probas.shape[0], num_classes)
        probs_scaled[np.arange(probs_scaled.shape[0]), y_pred] = certainties_scaled.float() + 1e-6 # add epsilon to ensure correct argmax when confidence=0
    # scaling methods
    elif 'VS' in method:
        logits = torch.log(calib_probas) # get "logits"
        dataset_logits_calib = LogitsDataset(logits, calib_labels)
        dataloader_logits_calib = DataLoader(dataset_logits_calib, batch_size=1)
        model_VS = fit_scaling_model('vector', dataloader_logits_calib, num_classes, binary_loss=('tva' in method), regularization=('reg' in method), temperature_ref=1, num_epochs=200)
        probs_scaled = model_VS(torch.log(test_probas).cuda()).detach().cpu()
        probs_scaled = torch.softmax(probs_scaled, dim=1) # scaling methods output logits, need to convert to probas
            
    return probs_scaled.numpy()
    
                    
def main(models, datasets, all_shots, num_seeds, subsample_test_set, approx, use_saved_results, bs,
         lr, val_seed, val_size, epochs, calib_size, tva, array_id):
    
    # 1. Initialize parameters
    default_params = {
        'conditioned_on_correct_classes': True,
        'subsample_test_set': subsample_test_set,
        'val_size': val_size,
        'lr': lr,
        'epochs': epochs,
        'val_seed': val_seed,
        'approx': approx,
        'bs': bs,
        'tva': tva,
        'calib_size': calib_size}
    # list of all experiment parameters to run
    all_params = []
    for model in models:
        for dataset in datasets:
            for num_shots in all_shots:
                for seed in range(num_seeds):
                    p = deepcopy(default_params)
                    p['model'] = model
                    p['dataset'] = dataset
                    p['seed'] = seed
                    p['num_shots'] = num_shots
                    p['expr_name'] = f"{p['dataset']}_{p['model']}_{p['num_shots']}shot_{repr(p['subsample_test_set'])}_subsample_seed{p['seed']}"
                    all_params.append(p)
    
    for params in all_params:
        df = pd.DataFrame()
        val_probas, val_labels, calib_probas, calib_labels, test_probas, test_labels, p_cf = get_data(params)
        num_classes = test_probas.shape[1]
        lr = params['lr']
        epochs = params['epochs']
        tva = params['tva']

        print(f"\nRunning {params['expr_name']}")

        # 2. Run calibration baselines
        W_ConC = torch.inverse(torch.eye(num_classes) * torch.tensor(p_cf))
        val_probas_ConC = calibrate_LinC(val_probas, W_ConC, torch.zeros([num_classes, 1])) # for post-hoc calibration
        calib_probas_ConC = calibrate_LinC(calib_probas, W_ConC, torch.zeros([num_classes, 1])) # for post-hoc calibration
        test_probas_ConC = calibrate_LinC(test_probas, W_ConC, torch.zeros([num_classes, 1]))
        
        W, b = learn_LinC(val_probas, val_labels, W_ConC, num_classes, epochs, lr, tva)
        val_probas_LinC = calibrate_LinC(val_probas, W, b) # for post-hoc calibration
        calib_probas_LinC = calibrate_LinC(calib_probas, W, b) # for post-hoc calibration
        test_probas_LinC = calibrate_LinC(test_probas, W, b)
            
            
        # 3. Evaluate baselines and post-hoc calibration
        for method, probas in {
            'original': test_probas,
            'ConC': test_probas_ConC,
            'LinC': test_probas_LinC}.items():
            
            acc = np.mean(np.argmax(probas, axis=1) == test_labels)
            ECE10 = expected_calibration_error(np.array(probas), np.array(test_labels), M=10)[0]
            ECE15 = expected_calibration_error(np.array(probas), np.array(test_labels), M=15)[0]

            df_ = pd.DataFrame([{**params, 'method': method, 'accuracy': acc, 'ECE10': ECE10, 'ECE15': ECE15}])
            df = pd.concat([df, df_], axis=0)

        for postHoc_method in ['netcal_HB_tva_eqsize']:
            test_probas_postHoc = post_hoc_calib_learn_transform(postHoc_method, val_probas, val_labels, test_probas, num_classes)
            test_probas_ConC_postHoc = post_hoc_calib_learn_transform(postHoc_method, val_probas_ConC, val_labels, test_probas_ConC, num_classes)
            test_probas_LinC_postHoc = post_hoc_calib_learn_transform(postHoc_method, val_probas_LinC, val_labels, test_probas_LinC, num_classes)

            for method_base, probas in {
                'original': test_probas_postHoc,
                'ConC': test_probas_ConC_postHoc,
                'LinC': test_probas_LinC_postHoc
                }.items():
                acc = np.mean(np.argmax(probas, axis=1) == test_labels)
                ECE10 = expected_calibration_error(np.array(probas), np.array(test_labels), M=10)[0]
                ECE15 = expected_calibration_error(np.array(probas), np.array(test_labels), M=15)[0]

                df_ = pd.DataFrame([{**params, 'method': f'{method_base}_postHoc_{postHoc_method}', 'accuracy': acc, 'ECE10': ECE10, 'ECE15': ECE15}])
                df = pd.concat([df, df_], axis=0)   
        
        # 4. Save results
        f_path = f'benchmark_tva_{params["model"]}_{params["dataset"]}_NEW.csv'
        if os.path.exists(f_path):
            df_0 = pd.read_csv(f_path)
            df = pd.concat([df_0, df], axis=0)
        df.to_csv(f_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--models', dest='models', action='store', required=True,
                        help='name of model(s), e.g., GPT2-XL')
    parser.add_argument('--datasets', dest='datasets', action='store', required=True,
                        help='name of dataset(s), e.g., agnews')
    parser.add_argument('--num_seeds', dest='num_seeds', action='store', required=True,
                        help='num seeds for the training set', type=int)
    parser.add_argument('--all_shots', dest='all_shots', action='store', required=True,
                        help='num training examples to use')
    # LinC arguments
    parser.add_argument('--lr', dest='lr', action='store', required=True, help='learning rate alpha', type=float)
    parser.add_argument('--val_seed', dest='val_seed', action='store', required=True,
                        help='seed to select the random set of validation demonstrations', type=int)
    parser.add_argument('--epochs', dest='epochs', action='store', required=True, help='total numbr of epochs T',
                        type=int)
    parser.add_argument('--val_size', dest='val_size', action='store', required=True,
                        help='size of validation set i.e. number of validation prompts', type=int)
    # other arguments
    parser.add_argument('--subsample_test_set', dest='subsample_test_set', action='store', required=False, type=int,
                        default=None, help='size of test set to use to speed up eval. None means using all test set')
    parser.add_argument('--bs', dest='bs', action='store', required=False, type=int, default=None,
                        help='batch size for model queries. For OpenAI API, capped at 20. For local running, set this to max out your GPU memory.')
    # flags
    parser.add_argument('--use_saved_results', dest='use_saved_results', action='store_const', const=True,
                        default=False,
                        help='whether to load the results from pickle files and not run the model')
    parser.add_argument('--approx', dest='approx', action='store_const', const=True, default=False,
                        help='whether to set token prob to zero if not in top 100')
    parser.add_argument('--calib_size', dest='calib_size', action='store', required=True,
                            help='size of calibration set for post-hoc calibration', type=int)
    parser.add_argument('--tva', dest='tva', action='store_const', const=True, default=False,
                        help='whether to use TVA')
    parser.add_argument('--array_id', dest='array_id', type=int, help='SLURM array id', default=None)
    
    args = parser.parse_args()
    args = vars(args)

    args['models'] = convert_to_list(args['models'])
    args['all_shots'] = convert_to_list(args['all_shots'], is_int=True)
    
    if args['array_id'] is not None:
        list_datasets = ['sst5','agnews','trec','dbpedia']
        args['datasets'] = [list_datasets[args['array_id']]]
    else:
        args['datasets'] = convert_to_list(args['datasets'])
        

    main(**args)