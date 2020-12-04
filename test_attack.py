from preactresnet import PreActResNet18
import torch
import torch.nn.functional as F
import time

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from attack import Args_pgd, attack_pgd, attack_pgd_01, attack_pgd_02, attack_pgd_my, attack_pgd_my_1
import copy

from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def dataset_norm(dataset):
    if dataset == 'mnist':
        mean = (0.0)
        std = (1.0)
    elif dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
        std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255
    return mean, std


def get_loader(dataset, train=True, batch_size=100, data_augm=False, num_workers=2, pin_memory=True):
    """load dataset: mnist, cifar"""
    mean, std = dataset_norm(dataset)

    if dataset == 'mnist':
        transform = transforms.ToTensor()
        dataset = datasets.MNIST("../data", train=train, download=True, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=pin_memory)

    elif dataset == 'cifar10':
        transform_list = []
        if train and data_augm:
            transform_list.append(transforms.RandomCrop(32, padding=4, padding_mode='reflect'))
            transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean, std))

        transform = transforms.Compose(transform_list)
        dataset = datasets.CIFAR10(root='../data', train=train, download=True, transform=transform)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=pin_memory)

    return dataloader


def test(test_loader, attack_func=None, attack_args=None):
    model.eval()
    test_loss = 0
    test_acc = 0
    test_robust_loss = 0
    test_robust_acc = 0
    test_n = 0

    start_time = time.time()
    for X, y in test_loader:
        X, y = X.cuda(), y.cuda()

        # Random initialization
        if attack_func is None:
            delta = torch.zeros_like(X)
        else:
            delta = attack_func(model, X, y, criterion, attack_args)
        with torch.no_grad():
            robust_output = model(X + delta)
            robust_loss = criterion(robust_output, y)

            output = model(X)
            loss = criterion(output, y)

            test_robust_loss += robust_loss.item() * y.size(0)
            test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            test_n += y.size(0)

    test_loss = test_loss / test_n
    test_acc = test_acc / test_n
    test_robust_loss = test_robust_loss / test_n
    test_robust_acc = test_robust_acc / test_n

    test_time = time.time() - start_time
    print('%.1f  \t %.4f  \t %.4f  \t %.4f  \t %.4f' % (
          test_time, test_loss, test_acc, test_robust_loss, test_robust_acc))


model_name = 'mnist_net'
dataset = 'cifar10'

model = PreActResNet18()
if torch.cuda.device_count() > 1:#判断是不是有多个GPU
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model = model.cuda()

model_load = torch.load('cifar10_pgd10/' + 'model_19.pth')
model.load_state_dict(model_load)
# print(model_load['test_robust_acc'])

criterion = F.cross_entropy

test_loader = get_loader(dataset, train=False, batch_size=128, data_augm=False, num_workers=2)

args_test = Args_pgd('pgd')
# args_test.check_args_value()

run_num = 5

# print('------------baseline 1-------------------')
# for i in range(run_num):
#     test(test_loader, attack_pgd_01, args_test)

# print('------------baseline 2-------------------')
# for i in range(run_num):
#     test(test_loader, attack_pgd_02, args_test)

args_test2 = copy.deepcopy(args_test)
mean, std = dataset_norm(dataset)
# print(mean, std)
mu = torch.tensor(mean).view(-1, 1, 1).cuda()
std = torch.tensor(std).view(-1, 1, 1).cuda()
args_test2.upper_limit = ((1 - mu) / std)
args_test2.lower_limit = ((0 - mu) / std)
args_test2.epsilon = (args_test2.epsilon) / std
args_test2.alpha = args_test2.alpha / std
args_test2.attack_init = 'zero'
# args_test2.early_stop = True
args_test2.check_args_value()

print('------------pgd--------------------------')
for i in range(run_num):
    test(test_loader, attack_pgd, args_test2)

print('------------pgd_01--------------------------')
for i in range(run_num):
    test(test_loader, attack_pgd_01, args_test2)

print('------------pgd_02--------------------------')
for i in range(run_num):
    test(test_loader, attack_pgd_02, args_test2)

print('------------pgd my-------------------')
for i in range(run_num):
    test(test_loader, attack_pgd_my, args_test2)

print('------------pgd my 1-------------------')
for i in range(run_num):
    test(test_loader, attack_pgd_my_1, args_test2)