import os
import argparse
from pathlib import Path
import sklearn.model_selection
from sklearn.isotonic import IsotonicRegression as SKLIsotonicRegression
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
from torchvision.models import (
    vgg16, VGG16_Weights, 
    resnet18, ResNet18_Weights,
    resnet34, ResNet34_Weights,
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
    efficientnet_b7, EfficientNet_B7_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    efficientnet_v2_m, EfficientNet_V2_M_Weights, 
    efficientnet_v2_l, EfficientNet_V2_L_Weights,   
    vit_b_32, ViT_B_32_Weights,
    vit_b_16, ViT_B_16_Weights,
    vit_l_32, ViT_L_32_Weights,
    vit_l_16, ViT_L_16_Weights,
    vit_h_14, ViT_H_14_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    convnext_small, ConvNeXt_Small_Weights,
    convnext_base, ConvNeXt_Base_Weights,
    convnext_large, ConvNeXt_Large_Weights,
    swin_t, Swin_T_Weights,
    swin_s, Swin_S_Weights, 
    swin_b, Swin_B_Weights,
    swin_v2_t, Swin_V2_T_Weights,
    swin_v2_s, Swin_V2_S_Weights,
    swin_v2_b, Swin_V2_B_Weights
)
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
import numpy as np
import pandas as pd
import time
import timm

import sys
sys.path.append('./benchmarking-uncertainty-estimation-performance-main/utils')
from focal_calibration.Net.resnet import resnet50 as cifar_resnet50, resnet110 as cifar_resnet110
from focal_calibration.Net.wide_resnet import wide_resnet_cifar as cifar_wide_resnet
from focal_calibration.Net.densenet import densenet121 as cifar_densenet121
from focal_calibration.Net.resnet import resnet50 as cifar_resnet50, resnet110 as cifar_resnet110
from focal_calibration.Data import cifar100 as cifar100_loader, cifar10 as cifar10_loader

sys.path.append('./Mix-n-Match-Calibration')
from util_calibration import mir_calibrate

from baseline_calibrators import ModelWithTemperatureOriginal, Patel2021
from calibrators import fit_scaling_model, fit_binary_method
from evaluation import metrics_from_logits_labels, metrics_from_certainties_correct, metrics_from_probas_labels, eval_binary_method
from utils import logits_labels_from_dataloader, convert_state_dict, LogitsDataset, CLIPClassifier

path_results = os.path.dirname(os.getcwd()) + '/results/'

path_imagenet = 'imagenet'
path_imagenet_21k = 'imagenet21k_val'

BATCH_SIZE = 64

models_and_weights_imagenet = {
    'VGG16': (vgg16, VGG16_Weights.DEFAULT),
    'ResNet-18': (resnet18, ResNet18_Weights.DEFAULT),
    'ResNet-34': (resnet34, ResNet34_Weights.DEFAULT),
    'ResNet-50': (resnet50, ResNet50_Weights.DEFAULT),
    'ResNet-101': (resnet101, ResNet101_Weights.DEFAULT),
    'EffNet-B7': (efficientnet_b7, EfficientNet_B7_Weights.DEFAULT),
    'EffNetV2-S': (efficientnet_v2_s, EfficientNet_V2_S_Weights.DEFAULT), 
    'EffNetV2-M': (efficientnet_v2_m, EfficientNet_V2_M_Weights.DEFAULT),
    'EffNetV2-L': (efficientnet_v2_l, EfficientNet_V2_L_Weights.DEFAULT), 
    'ViT-B/32': (vit_b_32, ViT_B_32_Weights.DEFAULT), 
    'ViT-B/16': (vit_b_16, ViT_B_16_Weights.DEFAULT), 
    'ViT-L/32': (vit_l_32, ViT_L_32_Weights.DEFAULT), 
    'ViT-L/16': (vit_l_16, ViT_L_16_Weights.DEFAULT), 
    'ViT-H/14': (vit_h_14, ViT_H_14_Weights.DEFAULT), 
    'ConvNeXt-T': (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT),
    'ConvNeXt-S': (convnext_small, ConvNeXt_Small_Weights.DEFAULT), 
    'ConvNeXt-B': (convnext_base, ConvNeXt_Base_Weights.DEFAULT),
    'ConvNeXt-L': (convnext_large, ConvNeXt_Large_Weights.DEFAULT), 
    'Swin-T': (swin_t, Swin_T_Weights.DEFAULT),
    'Swin-S': (swin_s, Swin_S_Weights.DEFAULT),
    'Swin-B': (swin_b, Swin_B_Weights.DEFAULT),
    'SwinV2-T': (swin_v2_t, Swin_V2_T_Weights.DEFAULT),
    'SwinV2-S': (swin_v2_s, Swin_V2_S_Weights.DEFAULT),
    'SwinV2-B': (swin_v2_b, Swin_V2_B_Weights.DEFAULT),
    'clip-vit-base-patch16': None,
    'clip-vit-large-patch14': None,
    'clip-vit-base-patch32': None,
}

models_and_weights_path_cifar10 = {
    'CIFAR10_ResNet-50': (cifar_resnet50, 'CIFAR10/resnet50_brier_score_350.model'),
    'CIFAR10_ResNet-110': (cifar_resnet110, 'CIFAR10/resnet110_brier_score_350.model'),
    'CIFAR10_WRN': (cifar_wide_resnet, 'CIFAR10/wide_resnet_brier_score_550.model'),
    'CIFAR10_DenseNet': (cifar_densenet121, 'CIFAR10/densenet121_brier_score_350.model'),
    'clip-vit-base-patch16': (None, None),
    'clip-vit-large-patch14': (None, None),
    'clip-vit-base-patch32': (None, None),
}

models_and_weights_path_cifar100 = {
    'CIFAR100_ResNet-50': (cifar_resnet50, 'CIFAR100/resnet50_brier_score_350.model'),
    'CIFAR100_ResNet-110': (cifar_resnet110, 'CIFAR100/resnet110_brier_score_430.model'),
    'CIFAR100_WRN': (cifar_wide_resnet, 'CIFAR100/wide_resnet_brier_score_350.model'),
    'CIFAR100_DenseNet': (cifar_densenet121, 'CIFAR100/densenet121_brier_score_350.model'),
    'clip-vit-base-patch16': (None, None),
    'clip-vit-large-patch14': (None, None),
    'clip-vit-base-patch32': (None, None),
}

model_names_imagenet21k = [
    'vit_base_patch16_224_miil_in21k',
    'mobilenetv3_large_100_miil_in21k',
    'tresnet_m_miil_in21k',
    'mixer_b16_224_miil_in21k'
]

model_names_PLM = ['t5', 'roberta', 't5-large', 'roberta-large']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-v', '--calib_size_IN', type=int, default=25000)
    parser.add_argument('-d', '--dataset', type=str, default='ImageNet') # CIFAR10 CIFAR100 ImageNet ImageNet21k amazon_food dynasent mnli yahoo_answers_topics
    args = parser.parse_args()
    seed = args.seed
    calib_size_IN = args.calib_size_IN
    dataset_name = args.dataset
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    
    if dataset_name == 'ImageNet':
        list_models = models_and_weights_imagenet.keys()
    elif dataset_name == 'CIFAR10':
        list_models = models_and_weights_path_cifar10.keys()
    elif dataset_name == 'CIFAR100':
        list_models = models_and_weights_path_cifar100.keys()
    elif dataset_name == 'ImageNet21k':
        list_models = model_names_imagenet21k
    elif dataset_name in ['amazon_food', 'dynasent', 'mnli', 'yahoo_answers_topics']:
        list_models = model_names_PLM
    else:
        raise ValueError('Unknown dataset')
                                                                     
    list_methods = [
        'original',
        # Scaling methods
        'TS', 'TS_tva', ## 'TS_original',
        'VS', 'VS_reg_tva', 'VS_reg', 'VS_tva',
        'Dir-ODIR', 'Dir-ODIR_reg_tva', 'Dir-ODIR_tva', 'Dir-ODIR_reg',
        # Binary methods
        'netcal_HB_eqsize', 'netcal_HB_tva_eqmass', 'netcal_HB_tva_eqsize', 'netcal_HB_eqmass',
        'netcal_Iso', 'netcal_Iso_tva',
        'netcal_Beta', 'netcal_Beta_tva',
        'netcal_BBQ', 'netcal_BBQ_tva',
        'netcal_ENIR', 'netcal_ENIR_tva',
        # Concurrent methods
        'IRM',
        'Patel2021_sCW_imax', 
        'Patel2021_top1_imax', 
        ]
    
    for model_name in list_models:
        print(f'Processing {model_name}...')
        
        if dataset_name in ['CIFAR10', 'CIFAR100', 'ImageNet', 'ImageNet21k']:
            # LOAD CLASSIFIER
            if dataset_name == 'ImageNet':
                if 'clip' in model_name:
                    classifier = CLIPClassifier(model_name, dataset_name)
                    transforms = T.Compose(
                            [T.Resize((224, 224)), 
                            T.ToTensor()])
                else:
                    architecture, weights = models_and_weights_imagenet[model_name]
                    classifier = architecture(weights=weights).eval().cuda()
                    transforms = weights.transforms()
                num_classes = 1000
                num_epochs = 200
                calib_size = calib_size_IN
            elif 'CIFAR' in dataset_name:
                if dataset_name == 'CIFAR10':
                    num_classes = 10
                    architecture, weights_path = models_and_weights_path_cifar10[model_name]
                elif dataset_name == 'CIFAR100':
                    num_classes = 100
                    architecture, weights_path = models_and_weights_path_cifar100[model_name]
                if 'clip' in model_name:
                    classifier = CLIPClassifier(model_name, dataset_name)
                else:
                    classifier = architecture(num_classes=num_classes, temp=1.0).eval().cuda()
                    classifier.load_state_dict(convert_state_dict(torch.load('./focal_calibration_models/' + weights_path)))
                    model_name = model_name.split('_')[1]
                num_epochs = 200
                calib_size = 5000
            elif dataset_name == 'ImageNet21k':
                classifier = timm.create_model(model_name, pretrained=True).eval().cuda()
                transforms = timm.data.create_transform(**timm.data.resolve_data_config({}, model=classifier))
                num_classes = 10450
                num_epochs = 20
                calib_size = 261250
            
            
            # GET LOGITS AND LABELS
            file_name = f'../results/logits_labels/{dataset_name}_{model_name.replace("/", "-")}_logits_labels.pt'
            if os.path.exists(file_name):
                print('loading logits and labels from file')
                all_logits, all_labels = torch.load(file_name)
            else:
                print('creating logits and labels from scratch')
                start_time_data = time.time()
                if 'ImageNet' in dataset_name:
                    if dataset_name == 'ImageNet':
                        path_IN = path_imagenet
                    elif dataset_name == 'ImageNet21k':
                        path_IN = path_imagenet_21k
                    dataset_val = ImageFolder(path_IN, transform=transforms)
                    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, shuffle=False)
                    all_logits, all_labels = logits_labels_from_dataloader(classifier, dataloader_val, in21k=(dataset_name=='ImageNet21k'))
                elif 'CIFAR' in dataset_name:
                    if dataset_name == 'CIFAR10':
                        _, valid_loader = cifar10_loader.get_train_valid_loader(batch_size=BATCH_SIZE, augment=False, random_seed=1, data_dir=Path(os.path.expandvars('$DSDIR/')), clip_model=('clip' in model_name)) # seed=1 in original github, Should not change it because model training data model used this seed
                        test_loader = cifar10_loader.get_test_loader(batch_size=BATCH_SIZE, data_dir=Path(os.path.expandvars('$DSDIR/')), clip_model=('clip' in model_name))
                    elif dataset_name == 'CIFAR100':
                        _, valid_loader = cifar100_loader.get_train_valid_loader(batch_size=BATCH_SIZE, augment=False, random_seed=1, data_dir=Path(os.path.expandvars('$DSDIR/')), clip_model=('clip' in model_name)) # seed=1 in original github, Should not change it because model training data model used this seed
                        test_loader = cifar100_loader.get_test_loader(batch_size=BATCH_SIZE, data_dir=Path(os.path.expandvars('$DSDIR/')), clip_model=('clip' in model_name))
                    logits_val, labels_val = logits_labels_from_dataloader(classifier, valid_loader)
                    logits_test, labels_test = logits_labels_from_dataloader(classifier, test_loader)
                    all_logits = torch.cat([logits_val, logits_test], axis=0)
                    all_labels = torch.cat([labels_val, labels_test], axis=0)
                print(f'Calibration data created in {time.time() - start_time_data:.0f} seconds')
                torch.save(((all_logits, all_labels)), file_name)
        
        elif dataset_name in ['amazon_food', 'dynasent', 'mnli', 'yahoo_answers_topics']:
            
            logits_calib = torch.tensor(np.load(f"../results/PLMCalibration/ood/{dataset_name}/{model_name}/Vanilla/calib/0/alllogits.npy"), dtype=torch.float32)
            labels_calib = torch.tensor(np.load(f"../results/PLMCalibration/ood/{dataset_name}/{model_name}/Vanilla/calib/0/alllabels.npy"))
            logits_test = torch.tensor(np.load(f"../results/PLMCalibration/ood/{dataset_name}/{model_name}/Vanilla/test/0/alllogits.npy"), dtype=torch.float32)
            labels_test = torch.tensor(np.load(f"../results/PLMCalibration/ood/{dataset_name}/{model_name}/Vanilla/test/0/alllabels.npy"))
            all_logits = torch.cat([logits_calib, logits_test], axis=0)
            all_labels = torch.cat([labels_calib, labels_test], axis=0)
            
            calib_size = len(labels_calib)
            if dataset_name == 'amazon_food':
                num_classes = 3
                num_epochs = 10
            elif dataset_name == 'dynasent':
                num_classes = 3
                num_epochs = 10
            elif dataset_name == 'mnli':
                num_classes = 3
                num_epochs = 10
            elif dataset_name == 'yahoo_answers_topics':
                num_classes = 10
                num_epochs = 10
            
        # CREATE CALIBRATION DATA
        test_indices, calib_indices = train_test_split(np.arange(all_logits.shape[0]), train_size=all_logits.shape[0] - calib_size,
                                                       stratify=all_labels, random_state=seed)
        logits_calib, labels_calib = all_logits[calib_indices], all_labels[calib_indices]
        logits_test, labels_test = all_logits[test_indices], all_labels[test_indices]
        dataset_logits_calib = LogitsDataset(logits_calib, labels_calib)
        dataloader_logits_calib = DataLoader(dataset_logits_calib, batch_size=512)

        
        # CALIBRATE
        for method in list_methods:
            try:
                print('\t', method)
                start_time_method = time.time()
                if method == 'original':
                    metrics_test = metrics_from_logits_labels(logits_test, labels_test)
                elif method == 'TS_original':
                    model = ModelWithTemperatureOriginal(classifier).cuda()
                    model.set_temperature(valid_loader)
                    logits_scaled = model.temperature_scale(logits_test.cuda()).detach().cpu()
                    metrics_test = metrics_from_logits_labels(logits_scaled, labels_test)
                    metrics_test['temperature'] = model.temperature.item()
                elif method == 'TS':
                    model = fit_scaling_model('temperature', dataloader_logits_calib, num_classes, binary_loss=False, regularization=False, num_epochs=num_epochs)
                    logits_scaled = model(logits_test.cuda()).detach().cpu()
                    metrics_test = metrics_from_logits_labels(logits_scaled, labels_test)
                    metrics_test['temperature'] = model.temp.item()
                elif method == 'TS_tva':
                    model = fit_scaling_model('temperature', dataloader_logits_calib, num_classes, binary_loss=True, regularization=False, num_epochs=num_epochs)
                    logits_scaled = model(logits_test.cuda()).detach().cpu()
                    metrics_test = metrics_from_logits_labels(logits_scaled, labels_test)
                    temp_tva = model.temp.item()
                    metrics_test['temperature'] = temp_tva
                elif method == 'VS':
                    model = fit_scaling_model('vector', dataloader_logits_calib, num_classes, binary_loss=False, regularization=False, num_epochs=num_epochs)
                    logits_scaled = model(logits_test.cuda()).detach().cpu()
                    metrics_test = metrics_from_logits_labels(logits_scaled, labels_test)
                elif method == 'VS_tva':
                    model = fit_scaling_model('vector', dataloader_logits_calib, num_classes, binary_loss=True, regularization=False, num_epochs=num_epochs, temperature_ref=temp_tva)
                    logits_scaled = model(logits_test.cuda()).detach().cpu()
                    metrics_test = metrics_from_logits_labels(logits_scaled, labels_test)
                elif method == 'VS_reg':
                    model = fit_scaling_model('vector', dataloader_logits_calib, num_classes, binary_loss=False, regularization=True, num_epochs=num_epochs, temperature_ref=temp_tva)
                    logits_scaled = model(logits_test.cuda()).detach().cpu()
                    metrics_test = metrics_from_logits_labels(logits_scaled, labels_test)
                elif method == 'VS_reg_tva':
                    model = fit_scaling_model('vector', dataloader_logits_calib, num_classes, binary_loss=True, regularization=True, num_epochs=num_epochs, temperature_ref=temp_tva)
                    logits_scaled = model(logits_test.cuda()).detach().cpu()
                    metrics_test = metrics_from_logits_labels(logits_scaled, labels_test)
                elif method == 'Dir-ODIR':
                    model = fit_scaling_model('dirichlet', dataloader_logits_calib, num_classes, binary_loss=False, regularization=False, num_epochs=num_epochs)
                    logits_scaled = model(logits_test.cuda()).detach().cpu()
                    metrics_test = metrics_from_logits_labels(logits_scaled, labels_test)
                elif method == 'Dir-ODIR_tva':
                    model = fit_scaling_model('dirichlet', dataloader_logits_calib, num_classes, binary_loss=True, regularization=False, num_epochs=num_epochs, temperature_ref=temp_tva)
                    logits_scaled = model(logits_test.cuda()).detach().cpu()
                    metrics_test = metrics_from_logits_labels(logits_scaled, labels_test)
                elif method == 'Dir-ODIR_reg':
                    model = fit_scaling_model('dirichlet', dataloader_logits_calib, num_classes, binary_loss=False, regularization=True, num_epochs=num_epochs, temperature_ref=temp_tva)
                    logits_scaled = model(logits_test.cuda()).detach().cpu()
                    metrics_test = metrics_from_logits_labels(logits_scaled, labels_test)
                elif method == 'Dir-ODIR_reg_tva':
                    model = fit_scaling_model('dirichlet', dataloader_logits_calib, num_classes, binary_loss=True, regularization=True, num_epochs=num_epochs, temperature_ref=temp_tva)
                    logits_scaled = model(logits_test.cuda()).detach().cpu()
                    metrics_test = metrics_from_logits_labels(logits_scaled, labels_test)
                elif 'netcal' in method:
                    model = fit_binary_method(method, logits_calib, labels_calib, num_classes)
                    metrics_test = eval_binary_method(method, model, logits_test, labels_test)
                elif method.split('_')[0] == 'Patel2021':
                    model = Patel2021(method.split('_')[1], num_classes, method.split('_')[2], seed=seed)
                    model.fit(logits_calib, labels_calib)
                    if method.split('_')[1] == 'sCW':
                        probs_scaled = model.predict(logits_test)
                        metrics_test = metrics_from_probas_labels(probs_scaled, labels_test)
                    elif method.split('_')[1] == 'top1':
                        certainties_scaled, y_pred = model.predict(logits_test)
                        correct = (y_pred == labels_test)
                        metrics_test = metrics_from_certainties_correct(certainties_scaled, correct)
                elif method == 'IRM':
                    probs = mir_calibrate(logits_calib.numpy(), one_hot(labels_calib, num_classes).numpy(), logits_test.numpy())
                    metrics_test = metrics_from_probas_labels(torch.tensor(probs), labels_test)
                else:
                    raise ValueError(f'Unknown method: {method}')
            except Exception as e:
                print(f'Error for {method}: {e}')
                metrics_test = {}
            exec_time = time.time() - start_time_method

            df_res_test = pd.DataFrame([{'dataset': dataset_name, 'calib_size': calib_size, 
                                         'num_epochs': num_epochs, 'model': model_name, 'method': method, **metrics_test, 'execution_time': exec_time}])
            f_path_test = path_results + f'benchmark_calibration_{dataset_name}_calibSize{calib_size}_seed{seed}.csv'

            if os.path.exists(f_path_test):
                df_0 = pd.read_csv(f_path_test)
                df_res_test = pd.concat([df_0, df_res_test], axis=0)
            df_res_test.to_csv(f_path_test, index=False)

