from __future__ import print_function, division
import math
import torch
import torchvision
from pathlib import Path
from torchvision.datasets import ImageFolder
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
import torch.nn as nn
import torch.nn.init as init
import os
import gc
import os.path as osp

import subprocess
import pickle
import numpy as np
import random


def get_data_file(filename, data_dir, label2id, unlabel=False):
    data = []
    targets = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data.append(os.path.join(data_dir, line.strip()))
            targets.append(label2id[line.strip().split("/")[1]])
    if unlabel:
        return np.array(data)

    return np.array(data), np.array(targets)


def get_data_file_unlabeled(filename, data_dir, label2id, unlabel=False):
    data = []
    targets = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data.append(os.path.join(data_dir, line.strip()))
            targets.append(label2id[line.strip().split("/")[1]])
    if unlabel:
        return np.array(data)

    return np.array(data), np.array(targets)


def get_data_file_cifar(data_dir, base_session, index, train, unlabel=False, class_list=None, unlabels_num=None, return_ulb=False, labels_num=None, dataset='cifar100', random=True, add_drift=False, drift_ratio=0.2, drift_type='all'):

    def SelectfromDefault(data, targets, index, num_per_class=None, return_ulb=False):
        data_tmp = []
        targets_tmp = []
        udata_tmp = []
        utargets_tmp = []
        
        for i in index:
            ind_cl = np.where(targets == i)[0]
            if num_per_class is not None:
                if len(data_tmp) == 0:
                    data_tmp = data[ind_cl][:num_per_class]
                    targets_tmp = targets[ind_cl][:num_per_class]
                    udata_tmp = data[ind_cl][num_per_class:]
                    utargets_tmp = targets[ind_cl][num_per_class:]
                else:
                    data_tmp = np.vstack((data_tmp, data[ind_cl][:num_per_class]))
                    targets_tmp = np.hstack((targets_tmp, targets[ind_cl][:num_per_class]))
                    udata_tmp = np.vstack((udata_tmp, data[ind_cl][num_per_class:]))
                    utargets_tmp = np.hstack((utargets_tmp, targets[ind_cl][num_per_class:]))
            else:
                if len(data_tmp) == 0:
                    data_tmp = data[ind_cl]
                    targets_tmp = targets[ind_cl]
                else:
                    data_tmp = np.vstack((data_tmp, data[ind_cl]))
                    targets_tmp = np.hstack((targets_tmp, targets[ind_cl]))
        
        if return_ulb:
            return data_tmp, targets_tmp, udata_tmp, utargets_tmp
        
        return data_tmp, targets_tmp

    def NewClassSelector(data, targets, index, num_per_class=None):
        data_tmp = []
        targets_tmp = []
        ind_list = [int(i) for i in index]
        ind_np = np.array(ind_list, dtype=int)
        
        if len(ind_np) == 25:
            index = ind_np.reshape((5,5))
            for i in index:
                ind_cl = i
                if len(data_tmp) == 0:
                    data_tmp = data[ind_cl]
                    targets_tmp = targets[ind_cl]
                else:
                    data_tmp = np.vstack((data_tmp, data[ind_cl]))
                    targets_tmp = np.hstack((targets_tmp, targets[ind_cl]))
        else:
            data_tmp, targets_tmp = data[ind_np], targets[ind_np]
        
        return data_tmp, targets_tmp

    def NewClassSelectorForUnlabels(data, targets, index, class_list, num_per_class=None):
        data = np.array(data)
        targets = np.array(targets)
        
        all_index = np.concatenate([np.where(targets == i)[0] for i in class_list])
        
        ind_np = np.array([int(i) for i in index])
        
        unlabels_index = np.setdiff1d(all_index, ind_np)
        
        unlabels_data, unlabels_targets = data[unlabels_index], targets[unlabels_index]

        if num_per_class is not None:
            for i in class_list:
                ind_cl = np.where(unlabels_targets == i)[0]
                if len(ind_cl) > num_per_class:
                    ind_cl = np.random.choice(ind_cl, num_per_class, replace=False)
                if len(data_tmp) == 0:
                    data_tmp = unlabels_data[ind_cl]
                    targets_tmp = unlabels_targets[ind_cl]
                else:
                    data_tmp = np.vstack((data_tmp, unlabels_data[ind_cl]))
                    targets_tmp = np.hstack((targets_tmp, unlabels_targets[ind_cl]))
        else:
            data_tmp, targets_tmp = unlabels_data, unlabels_targets

        return data_tmp, targets_tmp
    
    def NewClassSelectorForLabelsAndUnlabels(data, targets, index, class_list, num_per_class=None):
        data = np.array(data)
        targets = np.array(targets)
        
        ind_np = np.array([int(i) for i in index])

        all_labels_data, all_labels_targets = data[ind_np], targets[ind_np]

        all_index = np.concatenate([np.where(targets == i)[0] for i in class_list])
        labels_index = np.concatenate([ind_np[all_labels_targets == i] for i in class_list])

        unlabels_index = np.setdiff1d(all_index, labels_index)
        
        unlabels_data, unlabels_targets = data[unlabels_index], targets[unlabels_index]
        labels_data, labels_targets = data[labels_index], targets[labels_index]

        return labels_data, labels_targets, unlabels_data, unlabels_targets
        

    if dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True)
        testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True)
    elif dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)
    else:
        raise ValueError('dataset must be cifar10 or cifar100')
    
    if return_ulb:
        if random:
            data, targets, u_data, u_targets = SelectfromDefault(trainset.data, np.array(trainset.targets), index, num_per_class=labels_num, return_ulb=return_ulb)
        else:
            data, targets, u_data, u_targets = NewClassSelectorForLabelsAndUnlabels(trainset.data, np.array(trainset.targets), index, class_list, num_per_class=labels_num)

        if add_drift and train:
            data = apply_drift_to_data(data, drift_ratio=drift_ratio, drift_type=drift_type)
            u_data = apply_drift_to_data(u_data, drift_ratio=drift_ratio, drift_type=drift_type)
        
        return data, targets, u_data, u_targets

    if unlabel:
        if unlabels_num is not None:
            num_per_class = unlabels_num // len(class_list) + 1
        else:
            num_per_class = None
        data, targets = NewClassSelectorForUnlabels(trainset.data, np.array(trainset.targets), index, class_list, num_per_class)
    else:
        if train:
            if base_session:
                data, targets = SelectfromDefault(trainset.data, np.array(trainset.targets), index)
            else:
                data, targets = NewClassSelector(trainset.data, np.array(trainset.targets), index)
        else:
            if base_session:
                data, targets = SelectfromDefault(testset.data, np.array(testset.targets), index)
            else:
                data, targets = NewClassSelector(testset.data, np.array(testset.targets), index)

    assert len(data) == len(targets)

    if add_drift:
        data = apply_drift_to_data(data, drift_ratio=drift_ratio, drift_type=drift_type)
    
    return data, targets

def get_data_file_mnist(data_dir, base_session, index, train, unlabel=False, class_list=None, unlabels_num=None, return_ulb=False, labels_rate=None, dataset='bloodmnist', random=True):


    def SelectfromDefault(data, targets, index, rt_per_class=None, return_ulb=False):
        data_tmp = []
        targets_tmp = []
        udata_tmp = []
        utargets_tmp = []
        
        for i in index:
            ind_cl = np.where(targets == i)[0]
            if rt_per_class is not None:
                num_per_class = int(len(data[ind_cl])*rt_per_class)
                if len(data_tmp) == 0:
                    data_tmp = data[ind_cl][:num_per_class]
                    targets_tmp = targets[ind_cl][:num_per_class]
                    udata_tmp = data[ind_cl][num_per_class:]
                    utargets_tmp = targets[ind_cl][num_per_class:]
                else:
                    data_tmp = np.vstack((data_tmp, data[ind_cl][:num_per_class]))
                    targets_tmp = np.hstack((targets_tmp, targets[ind_cl][:num_per_class]))
                    udata_tmp = np.vstack((udata_tmp, data[ind_cl][num_per_class:]))
                    utargets_tmp = np.hstack((utargets_tmp, targets[ind_cl][num_per_class:]))
            else:
                if len(data_tmp) == 0:
                    data_tmp = data[ind_cl]
                    targets_tmp = targets[ind_cl]
                else:
                    data_tmp = np.vstack((data_tmp, data[ind_cl]))
                    targets_tmp = np.hstack((targets_tmp, targets[ind_cl]))
        
        if return_ulb:
            return data_tmp, targets_tmp, udata_tmp, utargets_tmp
        
        return data_tmp, targets_tmp

    if dataset == 'bloodmnist':
        from medmnist import BloodMNIST
        trainset = BloodMNIST(split="train", download=True, as_rgb=True)
        testset = BloodMNIST(split="test", download=True, as_rgb=True)
    elif dataset == 'pathmnist':
        from medmnist import PathMNIST
        trainset = PathMNIST(split="train", download=True, as_rgb=True)
        testset = PathMNIST(split="test", download=True, as_rgb=True)
    else:
        raise ValueError('dataset must be pathmnist or bloodmnist')
    train_data = trainset.imgs
    train_target = trainset.labels.reshape(-1,)
    test_data = testset.imgs
    test_target = testset.labels.reshape(-1,)

    if return_ulb:
        if random:
            data, targets, u_data, u_targets = SelectfromDefault(train_data, train_target, index, rt_per_class=labels_rate, return_ulb=return_ulb)
        else:
            raise ValueError('not impl')
            data, targets, u_data, u_targets = NewClassSelectorForLabelsAndUnlabels(trainset.data, np.array(trainset.targets), index, class_list, num_per_class=labels_num)
        
        return data, targets, u_data, u_targets

    if unlabel:
        raise ValueError('not impl')
        if unlabels_num is not None:
            num_per_class = unlabels_num // len(class_list) + 1
        else:
            num_per_class = None
        data, targets = NewClassSelectorForUnlabels(trainset.data, np.array(trainset.targets), index, class_list, num_per_class)
    else:
        if train:
            if base_session:
                data, targets = SelectfromDefault(train_data, train_target, index)
            else:
                raise ValueError('not impl')
                data, targets = NewClassSelector(trainset.data, np.array(trainset.targets), index)
        else:
            if base_session:
                data, targets = SelectfromDefault(test_data, test_target, index)
            else:
                raise ValueError('not impl')
                data, targets = NewClassSelector(testset.data, np.array(testset.targets), index)

    assert len(data) == len(targets)
    
    return data, targets

def find_classes(directory: Union[str, Path]) -> Tuple[List[str], Dict[str, int]]:
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(directory, class_to_idx, percentage=-1, extensions=None, is_valid_file=None, include_lb_to_ulb=True, lb_index=None):   
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return x.lower().endswith(extensions)
    
    lb_idx = {}
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            random.shuffle(fnames)
            if percentage != -1:
                fnames = fnames[:int(len(fnames) * percentage)]
            if percentage != -1:
                lb_idx[target_class] = fnames
            for fname in fnames:
                if not include_lb_to_ulb:
                    if fname in lb_index[target_class]:
                        continue
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    gc.collect()
    return instances, lb_idx


def get_label2id(filename):
    label_set = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            if line not in label_set.keys():
                label_set[line] = len(label_set)
    return label_set


def savepickle(data, file_path):
    mkdir_p(osp.dirname(file_path), delete=False)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def unpickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def mkdir_p(path, delete=False, print_info=True):
    if path == '': return

    if delete:
        subprocess.call(('rm -r ' + path).split())
    if not osp.exists(path):
        if print_info:
            print('mkdir -p  ' + path)
        subprocess.call(('mkdir -p ' + path).split())


def get_mean_and_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)


def find_and_delete_max(tensor):
    original_shape = tensor.shape
    row_map = list(range(original_shape[0]))
    col_map = list(range(original_shape[1]))
    delete_sequence = []

    while tensor.numel() > 0:
        max_value = torch.max(tensor)
        max_idx = (tensor == max_value).nonzero(as_tuple=False)[0]
        row, col = max_idx[0].item(), max_idx[1].item()

        original_row, original_col = row_map[row], col_map[col]
        delete_sequence.append((original_row, original_col))

        tensor = torch.cat((tensor[:row, :], tensor[row+1:, :]), dim=0)
        tensor = torch.cat((tensor[:, :col], tensor[:, col+1:]), dim=1)
        
        del row_map[row]
        del col_map[col]

    return delete_sequence


def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):

    from torch.optim.lr_scheduler import LambdaLR
    def _lr_lambda(current_step):

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def generate_random_orthogonal_matrix(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()

    return orth_vec


def generate_etf_vector(in_channels, num_classes):

    orth_vec = generate_random_orthogonal_matrix(in_channels, num_classes)
    i_nc_nc = torch.eye(num_classes)
    one_nc_nc = torch.mul(torch.ones(num_classes, num_classes), (1 / num_classes))
    etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                        math.sqrt(num_classes / (num_classes - 1)))
    
    return etf_vec

@torch.no_grad()
def mixup_one_target(x, y, alpha=1.0, is_bias=False):

    x, u = x.chunk(2, dim=0)
    y, p = y.chunk(2, dim=0)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    if is_bias:
        lam = max(lam, 1 - lam)

    index = torch.randperm(x.size(0)).to(x.device)

    mixed_x = lam * x + (1 - lam) * u[index]
    mixed_y = lam * y + (1 - lam) * p[index]
    return mixed_x, mixed_y, lam

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def apply_drift_to_data(data, drift_ratio=0.3, drift_type='all', seed=None):

    if seed is not None:
        np.random.seed(seed)
    
    data = data.copy()
    n_samples = len(data)
    n_drifted = int(n_samples * drift_ratio)
    
    drifted_indices = np.random.choice(n_samples, size=n_drifted, replace=False)
    
    for idx in drifted_indices:
        img = data[idx].astype(np.float32)
        
        if drift_type == 'all':
            num_drifts = np.random.randint(1, 4)
            available_drifts = ['brightness', 'contrast', 'noise', 'blur', 'salt_pepper', 'occlusion', 'color']
            selected_drifts = np.random.choice(available_drifts, size=num_drifts, replace=False)
        else:
            selected_drifts = [drift_type] if drift_type != 'both' else ['brightness', 'noise']

        for drift in selected_drifts:
            if drift == 'brightness':
                brightness_delta = np.random.uniform(-40, 40)
                img = img + brightness_delta
                img = np.clip(img, 0, 255)
            
            elif drift == 'contrast':
                alpha = np.random.uniform(0.5, 1.5)
                mean_intensity = np.mean(img)
                img = (img - mean_intensity) * alpha + mean_intensity
                img = np.clip(img, 0, 255)
            
            elif drift == 'noise':
                noise_std = np.random.uniform(15, 30)
                noise = np.random.normal(0, noise_std, img.shape)
                img = img + noise
                img = np.clip(img, 0, 255)
            
            elif drift == 'blur':
                from scipy.ndimage import gaussian_filter
                sigma = np.random.uniform(1.0, 2.5, size=3)
                img = gaussian_filter(img, sigma=(sigma[0], sigma[1], 0))
                img = np.clip(img, 0, 255)
            
            elif drift == 'salt_pepper':
                salt_prob = np.random.uniform(0.01, 0.05)
                pepper_prob = np.random.uniform(0.01, 0.05)
                
                salt_mask = np.random.random(img.shape) < salt_prob
                pepper_mask = np.random.random(img.shape) < pepper_prob
                
                img[salt_mask] = 255
                img[pepper_mask] = 0
                img = np.clip(img, 0, 255)
            
            elif drift == 'occlusion':
                h, w = img.shape[:2]
                num_blocks = np.random.randint(1, 4)
                
                for _ in range(num_blocks):
                    block_h = np.random.randint(int(h * 0.1), int(h * 0.3))
                    block_w = np.random.randint(int(w * 0.1), int(w * 0.3))
                    start_h = np.random.randint(0, h - block_h)
                    start_w = np.random.randint(0, w - block_w)
                    
                    occlusion_value = np.random.choice([0, np.random.uniform(100, 150)])
                    img[start_h:start_h+block_h, start_w:start_w+block_w, :] = occlusion_value
                
                img = np.clip(img, 0, 255)
            
            elif drift == 'color':
                color_shift = np.random.uniform(0.5, 1.5, size=3)
                img = img * color_shift
                img = np.clip(img, 0, 255)
        
        data[idx] = img.astype(np.uint8)
    
    return data