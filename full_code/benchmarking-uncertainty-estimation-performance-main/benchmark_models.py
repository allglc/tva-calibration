import sys
from utils import log_utils
import sklearn.model_selection
import tqdm
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import argparse
from timeit import default_timer as timer
from torch.utils.data import Dataset, DataLoader, Subset
from utils.uncertainty_metrics import *
from utils.temperature_scaling import ModelWithTemperature
import timm
from timm.data import resolve_data_config, create_transform


config_parser = parser = argparse.ArgumentParser(description='Benchmarking uncertainty performance config', add_help=False)
parser.add_argument('--data_dir', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64)')

parser.add_argument('--models', nargs='+', type=str, help='a list of model names available on the timm repo')

parser.add_argument('--checkpoint_path', default='', type=str, help='path to checkpoint file')
parser.add_argument('--use_class_weights', action='store_true')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_model_and_transforms(model_name, checkpoint_path):
    model = timm.create_model(model_name, pretrained=True, checkpoint_path=checkpoint_path).eval().to(device)
    # Creating the model specific data transformation
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return model, transform


def metrics_calculations(samples_certainties, num_bins_ece=15):
    # Note: we assume here the certainty scores in samples_certainties are probabilities.
    results = {}
    results['Accuracy'] = (samples_certainties[:,1].sum() / samples_certainties.shape[0]).item() * 100
    results['AUROC'] = AUROC(samples_certainties)
    results['Coverage_for_Accuracy_99'] = coverage_for_desired_accuracy(samples_certainties, accuracy=0.99, start_index=200)
    ece, mce = ECE_calc(samples_certainties, num_bins=num_bins_ece)
    results[f'ECE_{num_bins_ece}'] = ece
    results['AURC'] = AURC_calc(samples_certainties)
    return results


def extract_model_info(model, dataloader, pbar_name='Extracting data for model', class_acc=None):                   
    num_batches = len(dataloader.batch_sampler)
    total_correct = 0
    total_samples = 0
    # samples_certainties holds a tensor of size (N, 2) of N samples, for each its certainty and whether it was a
    # correct prediction.
    # Position 0 is the confidences and 1 is the correctness
    samples_certainties = torch.empty((0, 2))
    targets = torch.empty(0, dtype=torch.int)
    timer_start = timer()
    with torch.no_grad():
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=sys.stdout) as pbar:
            dl_iter = iter(dataloader)
            for batch_idx in range(num_batches):
                x, y = next(dl_iter)
                x = x.to(device)
                y = y.to(device)
                y_scores = model.forward(x)
                y_scores = torch.softmax(y_scores, dim=1)
                y_pred = torch.max(y_scores, dim=1)
                certainties = y_pred[0]
                correct = y_pred[1] == y
                total_correct += correct.sum().item()
                total_samples += x.shape[0]
                accuracy = (total_correct / total_samples) * 100

                samples_info = torch.stack((certainties.cpu(), correct.cpu()))
                samples_certainties = torch.vstack((samples_certainties, samples_info.transpose(0, 1)))
                targets = torch.cat((targets, y_pred[1].cpu())) # use prediction

                pbar.set_description(f'{pbar_name}. accuracy:{accuracy:.3f}% (Elapsed time:{timer() - timer_start:.3f} sec)')
                pbar.update()

            if class_acc is not None:
                print('Scale sample certainties.')
                acc_weights = torch.tensor([class_acc[y] for y in targets]) / class_acc.max()
                print('max acc', class_acc.max())
                samples_certainties[:, 0] = acc_weights * samples_certainties[:, 0]

            indices_sorting_by_confidence = torch.argsort(samples_certainties[:, 0], descending=True)
            samples_certainties = samples_certainties[indices_sorting_by_confidence]
            results = metrics_calculations(samples_certainties)
            return results


def extract_temperature_scaled_metrics(model, transform, valid_size=5000, model_name=None, class_acc=None):
    assert valid_size > 0
    dataset = torchvision.datasets.ImageFolder(args.data_dir, transform=transform)
    test_indices, valid_indices = sklearn.model_selection.train_test_split(np.arange(len(dataset)),
                                                                           train_size=len(dataset) - valid_size,
                                                                           stratify=dataset.targets)
    valid_loader = torch.utils.data.DataLoader(dataset, pin_memory=True, batch_size=args.batch_size,
                                               sampler=SubsetRandomSampler(valid_indices), num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset, pin_memory=True, batch_size=args.batch_size,
                                               sampler=SubsetRandomSampler(test_indices), num_workers=4)
    model = ModelWithTemperature(model)
    print(f'Performing temperature scaling')
    model.set_temperature(valid_loader)
    if model_name:
        pbar_name = f'Extracting data for {model_name} after temperature scaling'
    else:
        pbar_name = f'Extracting data for model after temperature scaling'
    model_results_TS = extract_model_info(model, test_loader, pbar_name=pbar_name, class_acc=class_acc)
    # To make sure all temperature scaled metrics have a distinct name, add _TS at its end
    model_results_TS = {f'{key}_TS': value for key, value in model_results_TS.items()}
    return model_results_TS


def compute_class_acc(model, dataloader_train, model_name):

    f_path = os.path.realpath(os.path.dirname(__file__)) + f'/class_acc/class_acc_{model_name}.npy'
    if os.path.exists(f_path):
        print('Load class-specific accuracies.')
        with open(f_path, 'rb') as f:
            acc_per_class = np.load(f)
    else:
        print('Compute class-specific accuracies on training set.')
        correct_pred = torch.empty(0, dtype=torch.bool)
        pred = torch.empty(0, dtype=torch.int)
        label = torch.empty(0, dtype=torch.int)
        for batch in tqdm.tqdm(dataloader_train):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                logits = model.forward(x)

            correct_pred = torch.cat((correct_pred, (logits.argmax(dim=1) == y).cpu()))
            pred = torch.cat((pred, logits.argmax(dim=1).cpu()))
            label = torch.cat((label, y.cpu()))
            
        n_classes = 1000
        acc_per_class = np.empty(n_classes)
        for class_idx in range(n_classes):
            acc_per_class[class_idx] = correct_pred[pred == class_idx].float().mean().item()
        with open(f_path, 'wb') as f:
            np.save(f, acc_per_class)

    return acc_per_class


def models_comparison(models_names: list, file_name='./results.csv', model_checkpoint_path='', use_class_weights=False):
    headers = ['Architecture', 'Accuracy', 'AUROC', 'AUROC_TS', 'ECE_15', 'ECE_15_TS', 'Coverage_for_Accuracy_99',
               'Coverage_for_Accuracy_99_TS', 'AURC', 'AURC_TS']
    logger = log_utils.Logger(file_name=file_name, headers=headers, overwrite=False)
    for model_name in models_names:
        model, transform = create_model_and_transforms(model_name, model_checkpoint_path)
        dataset = torchvision.datasets.ImageFolder(args.data_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        if use_class_weights:
            dataset_train = torchvision.datasets.ImageFolder(args.data_dir[:-3]+'train', transform=transform)
            dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
            class_acc = compute_class_acc(model, dataloader_train, model_name)
        else:
            class_acc = None
        model_results = extract_model_info(model, dataloader, pbar_name=f'Extracting data for {model_name}', class_acc=class_acc)
        # Temperature scaling
        temperature_scaled_model_results = extract_temperature_scaled_metrics(model, transform, model_name=model_name, class_acc=class_acc)
        model_results = {**model_results, **temperature_scaled_model_results}
        if use_class_weights:
            model_name += '_useClassWeights'
        model_results['Architecture'] = model_name
        # Log results
        logger.log(model_results)



if __name__ == '__main__':
    torch.manual_seed(1)
    np.random.seed(1)
    args = parser.parse_args()
    if args.use_class_weights:
        print('Using class weights.')
    models_comparison(args.models, file_name='./results.csv', model_checkpoint_path=args.checkpoint_path, use_class_weights=args.use_class_weights)