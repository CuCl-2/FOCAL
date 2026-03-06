from pathlib import Path
import argparse
import os
import random
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score
from torch.cuda.amp import autocast, GradScaler
from torch import nn
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from finetune_dataset import getdataset
from models.vit1d import vit_tiny, vit_small, vit_middle, vit_base

parser = argparse.ArgumentParser(description='FOCAL Linear Probing / Fine-tuning')
parser.add_argument('--dataset', default='ptbxl_super_class', type=str)
parser.add_argument('--ratio', default=100, type=int,
                    help='percentage of training data to use (1 / 10 / 100)')
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch-size', default=256, type=int)
parser.add_argument('--test-batch-size', default=256, type=int)
parser.add_argument('--learning-rate', default=1e-3, type=float)
parser.add_argument('--weight-decay', default=1e-4, type=float)
parser.add_argument('--pretrain_path', default='your_pretrained_encoder.pth', type=str,
                    help='path to FOCAL pretrained ECG encoder (.pth)')
parser.add_argument('--checkpoint-dir', default='./checkpoint_finetune/', type=Path)
parser.add_argument('--backbone', default='vit_tiny', type=str,
                    choices=['vit_tiny', 'vit_small', 'vit_middle', 'vit_base'])
parser.add_argument('--num_leads', default=12, type=int)
parser.add_argument('--name', default='LinearProbing', type=str,
                    help='exp name; use "linear" in name to freeze backbone')

_VIT_FACTORY = {
    'vit_tiny':   vit_tiny,
    'vit_small':  vit_small,
    'vit_middle': vit_middle,
    'vit_base':   vit_base,
}


def main():
    args = parser.parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = True
    print(f'Dataset: {args.dataset}')

    data_split_path = 'your_path/FOCAL/finetune/data_split'
    data_meta_path  = 'your_path/downstream'

    if 'ptbxl' in args.dataset:
        data_path = f'{data_meta_path}/ptbxl'
        split_dir = os.path.join(data_split_path, f'ptbxl/{args.dataset[6:]}')
        train_dataset = getdataset(data_path,
                                   os.path.join(split_dir, f'{args.dataset}_train.csv'),
                                   mode='train', dataset_name='ptbxl', ratio=args.ratio)
        val_dataset   = getdataset(data_path,
                                   os.path.join(split_dir, f'{args.dataset}_val.csv'),
                                   mode='val', dataset_name='ptbxl')
        test_dataset  = getdataset(data_path,
                                   os.path.join(split_dir, f'{args.dataset}_test.csv'),
                                   mode='test', dataset_name='ptbxl')

    elif args.dataset == 'CPSC2018':
        data_path = f'{data_meta_path}/icbeb2018/records500'
        split_dir = os.path.join(data_split_path, args.dataset)
        train_dataset = getdataset(data_path,
                                   os.path.join(split_dir, f'{args.dataset}_train.csv'),
                                   mode='train', dataset_name='icbeb', ratio=args.ratio)
        val_dataset   = getdataset(data_path,
                                   os.path.join(split_dir, f'{args.dataset}_val.csv'),
                                   mode='val', dataset_name='icbeb')
        test_dataset  = getdataset(data_path,
                                   os.path.join(split_dir, f'{args.dataset}_test.csv'),
                                   mode='test', dataset_name='icbeb')

    elif args.dataset == 'CSN':
        data_path = f'{data_meta_path}/downstream/'
        split_dir = os.path.join(data_split_path, args.dataset)
        train_dataset = getdataset(data_path,
                                   os.path.join(split_dir, f'{args.dataset}_train.csv'),
                                   mode='train', dataset_name='chapman', ratio=args.ratio)
        val_dataset   = getdataset(data_path,
                                   os.path.join(split_dir, f'{args.dataset}_val.csv'),
                                   mode='val', dataset_name='chapman')
        test_dataset  = getdataset(data_path,
                                   os.path.join(split_dir, f'{args.dataset}_test.csv'),
                                   mode='test', dataset_name='chapman')

    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    args.labels_name = train_dataset.labels_name
    num_classes = train_dataset.num_classes

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader   = torch.utils.data.DataLoader(
        val_dataset,   batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(
        test_dataset,  batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Build SpatialTemporalViT and load pretrained FOCAL encoder weights
    model = _VIT_FACTORY[args.backbone](num_leads=args.num_leads)
    ckpt  = torch.load(args.pretrain_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    print(f'Loaded pretrained encoder from {args.pretrain_path}')

    if 'linear' in args.name:
        for param in model.parameters():
            param.requires_grad = False
        print('Backbone frozen for linear probing.')

    model.reset_head(num_classes=num_classes)
    model.head.weight.requires_grad = True
    model.head.bias.requires_grad   = True

    model = model.to('cuda')
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[40], gamma=0.1)
    criterion = nn.BCEWithLogitsLoss()

    # Resume checkpoint if exists
    ckpt_name = (f'{args.name}-{args.backbone}-B-{args.batch_size}'
                 f'{args.dataset}R-{args.ratio}.pth')
    resume_path = args.checkpoint_dir / ckpt_name
    start_epoch = 0
    if resume_path.is_file():
        ckpt = torch.load(resume_path, map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])

    scaler = GradScaler()
    log = {k: [] for k in ['epoch', 'val_auc', 'test_auc']}
    class_log = {'val_log': [], 'test_log': []}

    for epoch in tqdm(range(start_epoch, args.epochs)):
        model.train()
        for ecg, target in tqdm(train_loader, leave=False):
            optimizer.zero_grad()
            with autocast():
                output = model(ecg.to('cuda'))
                loss   = criterion(output, target.to('cuda'))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        val_auc,  _, _, val_class  = infer(model, val_loader,  args)
        test_auc, _, _, test_class = infer(model, test_loader, args)

        log['epoch'].append(epoch)
        log['val_auc'].append(val_auc)
        log['test_auc'].append(test_auc)
        class_log['val_log'].append(val_class)
        class_log['test_log'].append(test_class)

    prefix = f'{args.checkpoint_dir}/{args.name}-{args.backbone}-B-{args.batch_size}{args.dataset}R-{args.ratio}'
    pd.DataFrame(log).to_csv(f'{prefix}.csv', index=False)
    pd.concat(class_log['val_log'],  axis=0).to_csv(f'{prefix}-val-class.csv',  index=False)
    pd.concat(class_log['test_log'], axis=0).to_csv(f'{prefix}-test-class.csv', index=False)

    print(f'Best val AUC:  {max(log["val_auc"]):.4f}')
    print(f'Best test AUC: {max(log["test_auc"]):.4f}')


@torch.no_grad()
def infer(model, loader, args):
    model.eval()
    y_pred, y_true = [], []

    for ecg, target in tqdm(loader, leave=False):
        pred = model(ecg.to('cuda'))
        y_true.append(target.cpu().numpy())
        y_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    auc    = roc_auc_score(y_true, y_pred, average='macro')

    max_f1s, accs = [], []
    for i in range(y_pred.shape[1]):
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_pred[:, i])
        denom = recall + precision
        f1s   = np.divide(2 * recall * precision, denom,
                          out=np.zeros_like(denom), where=(denom != 0))
        best_thresh = thresholds[np.argmax(f1s)]
        max_f1s.append(np.max(f1s) * 100)
        accs.append(accuracy_score(y_true[:, i], y_pred[:, i] > best_thresh) * 100)

    metric_dict = {name: [roc_auc_score(y_true[:, i], y_pred[:, i])]
                   for i, name in enumerate(args.labels_name)}

    return auc, np.mean(max_f1s), np.mean(accs), pd.DataFrame(metric_dict)


if __name__ == '__main__':
    main()
