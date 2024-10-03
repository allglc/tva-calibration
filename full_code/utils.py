import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision import transforms as T
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
from torch import nn


def logits_labels_from_dataloader(model, dataloader, in21k=False):
    if in21k: # classifier has fal11 labels, dataset winter21
        tree_fall11 = torch.load('./imagenet21k_miil_tree_fall11.pth')
        list_names_fall11 = list(tree_fall11['class_description'].keys())
        tree_winter21 = torch.load('./imagenet21k_miil_tree_winter21.pth')
        list_names_winter21 = list(tree_winter21['class_description'].keys())
        idx_winter21 = []
        for i, name in enumerate(list_names_fall11):
            if name in list_names_winter21: idx_winter21.append(i)

    logits_list = []
    labels_list = []
    for input, label in dataloader:
        input = input.cuda()
        with torch.no_grad():
            logits = model(input)
            if in21k: logits = logits[:, idx_winter21]
        logits_list.append(logits)
        labels_list.append(label)
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    return logits.cpu(), labels.cpu()


def convert_state_dict(state_dict):
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/4
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict
       
            
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


def load_dataloaders_CIFAR(dataset_path, batch_size):
    ''' from https://github.com/huyvnphan/PyTorch_CIFAR10/blob/master/data.py '''

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)
    
    transform_train = T.Compose(
        [T.RandomCrop(32, padding=4),
         T.RandomHorizontalFlip(),
         T.ToTensor(),
         T.Normalize(mean, std)])
    dataset_val_dataAug = CIFAR10(root=dataset_path, train=False, transform=transform_train) # val set with train transforms
    dataset_train, _ = torch.utils.data.random_split(dataset_val_dataAug, [5000, 5000], generator=torch.Generator().manual_seed(42)) # 1st half of validation set with data augmentation
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    transform_val = T.Compose(
        [T.ToTensor(),
         T.Normalize(mean, std)])
    dataset_val = CIFAR10(root=dataset_path, train=False, transform=transform_val)
    _, dataset_val2 = torch.utils.data.random_split(dataset_val, [5000, 5000], generator=torch.Generator().manual_seed(42)) # 2nd half of validation set without data augmentation
    dataloader_val = DataLoader(dataset_val2, batch_size=batch_size, shuffle=False, pin_memory=True)

    return dataloader_train, dataloader_val


def load_dataloaders_ImageNet(dataset_path, batch_size):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transforms = T.Compose(
        [T.Resize((224, 224)), 
        T.ToTensor(),
        T.Normalize(mean, std)])
    
    imagenet_data_train = ImageFolder(dataset_path+'/train', transform=transforms)
    data_loader_train = DataLoader(imagenet_data_train, batch_size=batch_size, shuffle=True)    
    
    imagenet_data_val = ImageFolder(dataset_path+'/val', transform=transforms)
    data_loader_val = DataLoader(imagenet_data_val, batch_size=batch_size, shuffle=False)
    
    return data_loader_train, data_loader_val


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax.
    From https://github.com/NVlabs/stylegan3/blob/main/dnnlib/util.py"""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value):
        self[name] = value

    def __delattr__(self, name: str):
        del self[name]
        
        

def load_val_dataloader_ImageNet21(dataset_path, batch_size):
    # from https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/src_files/data_loading/data_loader.py

    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    val_dataset = ImageFolder(dataset_path, transform=val_transform)
    print("length val dataset: {}".format(len(val_dataset)))

    # Pytorch Data loader
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

    # val_loader = PrefetchLoader(val_loader)
    return val_loader


class CLIPClassifier(nn.Module):
    def __init__(self, model_name, data='ImageNet'):
        super().__init__()
        self.CLIP_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to('cuda')
        self.CLIP_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        
        if data == 'ImageNet':
            df = pd.read_csv('./imagenet_categories_synset.csv')
            classnames = df['words'].apply(lambda x:x.split(',')[0]).to_list()
        elif data == 'CIFAR10':
            classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        elif data == 'CIFAR100':
            classnames = [
                'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
                'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
                'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
                'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
                'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
                'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
                'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
            ]
        self.text_prompts = [f'A photo of a {c}.' for c in classnames]
        
    def forward(self, images):
        with torch.no_grad():
            images = images * 255
            inputs = self.CLIP_processor(text=self.text_prompts, images=images, return_tensors="pt", padding=True)
            inputs = inputs.to('cuda')
            outputs = self.CLIP_model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        # probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        return logits_per_image