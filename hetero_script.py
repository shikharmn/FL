import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from omegaconf import OmegaConf
import numpy as np

config = """
logs_dir: './runs_model2'
dataset: 'MNIST'
classes: 10
batch_size: 256
device: 'cuda:0'
epochs: 40
steps:
  total: 4000
  local: 16
uniform:
  workers: 10
"""

cfg = OmegaConf.create(config)

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

def label_avg(check_list):
    averaged_label = 0
    stats = np.zeros((10,1))
    for i in range(min_len):
        _, label = dataset_train[check_list[i]]
        stats[label] += 1.0
        averaged_label += label/min_len
    print(1.0 * stats/stats.sum())
    return averaged_label

def fusion(models, optimizers):
    state_dicts = [model.state_dict() for model in models]
    model_averaged = CNN_FedAvg().to(device)
    for key in state_dicts[0]:
        model_averaged.state_dict()[key] = sum([state_dict[key] for state_dict in state_dicts]) / cfg.uniform.workers
    for idx in range(len(models)):
        models[idx].load_state_dict(model_averaged.state_dict())
    optimizers = [optim.SGD(model.parameters(), lr = 0.002) for model in models]
    return models, optimizers, model_averaged

def fusion_2(models, optimizers):
    parameters = [model.named_parameters() for model in models]
    model_averaged = CNN_FedAvg().to(device)
    dict_params_avg = dict(model_averaged.named_parameters())
    dicts_params = [dict(parameter) for parameter in parameters]

    for name, _ in parameters[0]:
        if name in dict_params_avg:
            dict_params_avg[name].data.copy_(sum([dict_params[name].data for dict_params in dicts_params]))

    model_averaged.load_state_dict(dict_params_avg)
    for idx in range(len(models)):
        models[idx].load_state_dict(dict_params_avg)
    optimizers = [optim.SGD(model.parameters(), lr = 0.002) for model in models]

    return models, optimizers, model_averaged

def shuffle_along_axis(a, axis=0):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

class CNN_FedAvg(nn.Module):

    def __init__(self, classes=10):
        super(CNN_FedAvg, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv2d_3 = torch.nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(3136, 512)
        self.linear_2 = nn.Linear(512, classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.relu(self.linear_1(x))
        # x = self.linear_2(x)
        x = self.softmax(self.linear_2(x))
        return x

#load the data
dataset_train = datasets.MNIST('./data', train=True, download=True,
                               transform=transforms.Compose(
                                   [transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])
                               )
dataset_test = datasets.MNIST('./data', train=False, download=True,
                              transform=transforms.Compose(
                                   [transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])
                               )

train_loader = torch.utils.data.DataLoader(dataset_train,
                                           batch_size=cfg.batch_size,
                                           shuffle=False, drop_last=False)
test_loader = torch.utils.data.DataLoader(dataset_test,
                                          batch_size=cfg.batch_size,
                                          shuffle=False)

data_idx = [[] for i in range(10)]

for i, (_, label) in enumerate(tqdm(dataset_train)):
    data_idx[label].append(i)
lens = []
for i in range(10): lens.append(len(data_idx[i]))
min_len = (min(lens)) // cfg.batch_size * cfg.batch_size    # drop_last=True

for i in range(10): data_idx[i] = data_idx[i][:min_len]
data_idx = np.asarray(data_idx)

idx_shuffles = [min_len // sim_fraction for sim_fraction in [min_len+1,10,2,1]]

idx_splits_shuffled = [shuffle_along_axis(data_idx.copy()[:,:idx_shuffle]) for idx_shuffle in idx_shuffles]

idx_shuffled_data = []
for idx_shuffle, idx_split_shuffled in zip(idx_shuffles, idx_splits_shuffled):
    new_data = data_idx
    if idx_shuffle != 0:
        new_data[:,:idx_shuffle] = idx_split_shuffled
    idx_shuffled_data.append(new_data.copy())

hetero_dataloaders = [[] for i in range(4)]
for idx_sim,idxs in enumerate(idx_shuffled_data):
    for label in range(10):
        split_dataset = Subset(dataset_train,idxs[label])
        hetero_dataloaders[idx_sim].append(DataLoader(split_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=True, drop_last=False))

for similarity,train_loaders in zip(["10"],hetero_dataloaders[1:2]):
    for local_steps in [4, 16, 64]:
        client_models = [CNN_FedAvg() for i in range(cfg.uniform.workers)]

        writer = SummaryWriter(cfg.logs_dir + '/hetero_fixed_2' + similarity + '_' + str(local_steps))
        criterion = nn.NLLLoss()

        for i in range(cfg.uniform.workers): client_models[i] = client_models[i].to(cfg.device)
        optimizers = [optim.SGD(model.parameters(), lr = 0.002) for model in client_models]

        data_iters = [iter(dataloader) for dataloader in train_loaders]
        for model in client_models: model.to(device)
        for step in tqdm(range(cfg.steps.total)):
            for model in client_models: model.train()

            # Get data from the dataloaders
            try:
                data_pairs = [next(data_iter) for data_iter in data_iters]
            except StopIteration:
                data_iters = [iter(dataloader) for dataloader in train_loaders]
                data_pairs = [next(data_iter) for data_iter in data_iters]

            data_pairs = [(data.to(device), target.to(device)) for data, target in data_pairs]

            # Train
            for optimizer in optimizers: optimizer.zero_grad()
            outputs = [model(data.squeeze(1)) for model, (data, _) in zip(client_models, data_pairs)]
            preds = [output.max(1, keepdim=True)[1] for output in outputs]
            for idx, (_, labels) in enumerate(data_pairs):
                writer.add_scalar(f'debug/worker_{idx}/gt_mean', labels.float().mean().item(), step)
                writer.add_scalar(f'debug/worker_{idx}/gt_std', labels.float().std().item(), step)
                writer.add_scalar(f'debug/worker_{idx}/pred_mean', preds[idx].float().mean().item(), step)
                writer.add_scalar(f'debug/worker_{idx}/pred_std', preds[idx].float().std().item(), step)
            losses = [criterion(output, target) for output, (_, target) in zip(outputs, data_pairs)]
            for loss in losses: loss.backward()
            for optimizer in optimizers: optimizer.step()

            for idx, loss in enumerate(losses):
                writer.add_scalar('train/Loss_' + str(idx), loss.item(), step)

            # Fusion
            if step % local_steps == 0 and idx != 0:
                client_models, optimizers, averaged_model = fusion_2(client_models, optimizers)

            if step % local_steps == 0 or step % local_steps == local_steps - 1:
              # Testing
                averaged_model.eval()
                test_loss = 0
                correct = 0
                preds = []
                with torch.no_grad():
                    for data, target in tqdm(test_loader):
                        data, target = data.to(device), target.to(device)
                        output = averaged_model(data.squeeze(1))
                        test_loss += criterion(output, target).item()
                        pred = output.max(1, keepdim=True)[1]
                        preds.append(pred.cpu().numpy())
                        correct += pred.eq(target.view_as(pred)).sum().item()

                acc = 100. * correct / len(test_loader.dataset)
                preds = np.concatenate(preds)
                writer.add_scalar('test/pred_mean', preds.mean(), step)
                writer.add_scalar('test/pred_std', preds.std(), step)
                writer.add_scalar('test/Loss', test_loss, step)
                writer.add_scalar('test/Accuracy', acc, step)