import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision import transforms as T


def logits_labels_from_dataloader(model, dataloader):
    logits_list = []
    labels_list = []
    for input, label in dataloader:
        input = input.cuda()
        with torch.no_grad():
            logits = model(input)
        logits_list.append(logits)
        labels_list.append(label)
    logits = torch.cat(logits_list).cpu()
    labels = torch.cat(labels_list).cpu()
    return logits, labels

            
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