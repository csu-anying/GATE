import time
from pathlib import Path
import argparse
import os
import random
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, \
    precision_recall_curve, average_precision_score
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from torch import nn
import torch
from matplotlib import pyplot as plt
from finetune_dataset import getdataset
from models.graph_classifier import ECG_Graph_Classifier

parser = argparse.ArgumentParser(description='Graph-text Finetuning')
# ptbxl_super_class icbeb
parser.add_argument('--dataset', default='ptbxl_super_class',
                    type=str, help='dataset name')
parser.add_argument('--ratio', default='1',
                    type=int, help='training data ratio')
parser.add_argument('--seq_len', default=5000,
                    type=int, help='the length of data')
parser.add_argument('--patchsize', default=500,
                    type=int, help='patchsize')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--test-batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate', default=1e-2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--weight-decay', default=1e-8, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--pretrain_path', default="../checkpoints/pretrain/GATE_best_ckpt.pth",
                    type=str,
                    help='path to pretrain weight directory')
parser.add_argument('--checkpoint_dir', default='../checkpoints/finetune/',
                    type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--backbone', default='GATE', type=str, metavar='B',
                    help='backbone name')
parser.add_argument('--num_leads', default=12, type=int, metavar='B',
                    help='number of leads')
parser.add_argument('--name', default='LinearProbing', type=str, metavar='B',
                    help='LinearProbing or Finetuning')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from collections import OrderedDict


def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_key = k[7:]  # 去掉 'module.'
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict


def main():
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    batch_size = int(args.batch_size)
    test_batch_size = int(args.test_batch_size)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.empty_cache()
    device_id = torch.cuda.device_count()
    torch.manual_seed(42)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = True
    print(f'this task use {args.dataset} dataset')

    data_split_path = './data_split'
    data_meta_path = 'your_path/data'

    if 'ptbxl' in args.dataset:
        # set the path where you store the ptbxl dataset
        data_path = f'{data_meta_path}/PTB_XL/raw'
        data_split_path = os.path.join(data_split_path, f'ptbxl/{args.dataset[6:]}')

        train_csv_path = f'{args.dataset}_train.csv'
        train_csv_path = os.path.join(data_split_path, train_csv_path)
        val_csv_path = f'{args.dataset}_val.csv'
        val_csv_path = os.path.join(data_split_path, val_csv_path)
        test_csv_path = f'{args.dataset}_test.csv'
        test_csv_path = os.path.join(data_split_path, test_csv_path)

        train_dataset = getdataset(data_path, train_csv_path, mode='train', dataset_name='ptbxl', ratio=args.ratio,
                                   backbone=args.backbone)
        val_dataset = getdataset(data_path, val_csv_path, mode='val', dataset_name='ptbxl',
                                 backbone=args.backbone)
        test_dataset = getdataset(data_path, test_csv_path, mode='test', dataset_name='ptbxl',
                                  backbone=args.backbone)

        args.labels_name = train_dataset.labels_name
        num_classes = train_dataset.num_classes

    elif args.dataset == 'icbeb':
        # set the path where you store the CPSC2018 dataset, the CPSC2018 dataset folder should be icbeb2018/records500/...
        data_path = f'{data_meta_path}/ICBEB/records500'
        data_split_path = os.path.join(data_split_path, args.dataset)

        train_csv_path = f'{args.dataset}_train.csv'
        train_csv_path = os.path.join(data_split_path, train_csv_path)
        val_csv_path = f'{args.dataset}_val.csv'
        val_csv_path = os.path.join(data_split_path, val_csv_path)
        test_csv_path = f'{args.dataset}_test.csv'
        test_csv_path = os.path.join(data_split_path, test_csv_path)

        train_dataset = getdataset(data_path, train_csv_path, mode='train', dataset_name='icbeb', ratio=args.ratio,
                                   backbone=args.backbone)
        val_dataset = getdataset(data_path, val_csv_path, mode='val', dataset_name='icbeb',
                                 backbone=args.backbone)
        test_dataset = getdataset(data_path, test_csv_path, mode='test', dataset_name='icbeb',
                                  backbone=args.backbone)

        args.labels_name = train_dataset.labels_name
        num_classes = train_dataset.num_classes

    elif args.dataset == 'chapman':
        # set the path where you store the CSN dataset, the CSN dataset folder should be chapman/...
        data_path = f'{data_meta_path}/'
        data_split_path = os.path.join(data_split_path, args.dataset)

        train_csv_path = f'{args.dataset}_train.csv'
        train_csv_path = os.path.join(data_split_path, train_csv_path)
        val_csv_path = f'{args.dataset}_val.csv'
        val_csv_path = os.path.join(data_split_path, val_csv_path)
        test_csv_path = f'{args.dataset}_test.csv'
        test_csv_path = os.path.join(data_split_path, test_csv_path)

        train_dataset = getdataset(data_path, train_csv_path, mode='train', dataset_name='chapman', ratio=args.ratio,
                                   backbone=args.backbone)
        val_dataset = getdataset(data_path, val_csv_path, mode='val', dataset_name='chapman',
                                 backbone=args.backbone)
        test_dataset = getdataset(data_path, test_csv_path, mode='test', dataset_name='chapman',
                                  backbone=args.backbone)

        args.labels_name = train_dataset.labels_name
        num_classes = train_dataset.num_classes

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    ckpt_path = args.pretrain_path
    ckpt = torch.load(ckpt_path, map_location='cpu')
    ckpt = remove_module_prefix(ckpt)
    # Load config
    config = yaml.load(open("../pretrain/config.yaml", "r"), Loader=yaml.FullLoader)
    # Initialize model and load pre-trained weights
    model = ECG_Graph_Classifier(config['network'], num_classes)
    model.load_state_dict(ckpt, strict=False)
    print(
        f'load pretrained model from {args.pretrain_path}, the backbone is {args.backbone}, using {args.num_leads} leads')
    # Optionally freeze layers of the model
    if 'Linear' in args.name:
        print("yes!")
        for _, p in model.named_parameters():
            p.requires_grad = False
        for _, p in model.classifier.named_parameters():
            p.requires_grad = True
    model.eval()

    model = model.to(device)
    print(model)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[40],
                                                     gamma=0.1,
                                                     last_epoch=-1)
    criterion = nn.BCEWithLogitsLoss()

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / (
            args.backbone + '-checkpoint-' + 'B-' + str(batch_size) + args.dataset + '.pth')).is_file():
        ckpt = torch.load(args.checkpoint_dir / (
                args.backbone + '-checkpoint-' + 'B-' + str(batch_size) + args.dataset + 'R-' + str(
            args.ratio) + '.pth'),
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        start_epoch = 0
        print("start epoch")

    global_step = 0

    log = {
        'epoch': [],
        'val_acc': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'val_auc': [],
        'test_acc': [],
        'test_f1': [],
        'test_precision': [],
        'test_recall': [],
        'test_auc': []
    }
    class_log = {
        'val_log': [],
        'test_log': []
    }

    scaler = GradScaler()
    train_time = 0
    infer_time = 0
    best_test_auc = 0
    for epoch in range(start_epoch, args.epochs):
        print(f"===================Epoch {epoch}===========================")
        time0 = time.time()
        model.train()
        for step, (ecg, target) in tqdm(enumerate(train_loader, start=epoch * len(train_loader))):
            optimizer.zero_grad()
            with autocast():
                output = model(ecg.to(device))
                # print(output.shape)
                loss = criterion(output, target.to(device))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        duration = time.time() - time0
        train_time += duration
        # Time
        start = time.time()
        val_acc, val_f1, val_precision, val_recall, val_auc, val_metric_class = infer(model, val_loader, args)
        infers = time.time() - start
        infer_time += infers
        print(
            f"val acc {val_acc}, val precision {val_precision}, val recall {val_recall}, val f1 {val_f1}, val auc {val_auc}")
        test_acc, test_f1, test_precision, test_recall, test_auc, test_metric_class = infer(model, test_loader, args)
        print(
            f"test acc {test_acc}, test precision {test_precision}, test recall {test_recall}, test f1 {test_f1}, test auc {test_auc}")

        if 'Finetuning' in args.name and test_auc > best_test_auc:
            print("Saving==")
            best_test_auc = test_auc
            torch.save(model.state_dict(), f'{args.checkpoint_dir}/' + 'finetune_final_total.pth')

        log['epoch'].append(epoch)
        log['val_acc'].append(val_acc)
        log['val_f1'].append(val_f1)
        log['val_precision'].append(val_precision)
        log['val_recall'].append(val_recall)
        log['val_auc'].append(val_auc)
        log['test_acc'].append(test_acc)
        log['test_f1'].append(test_f1)
        log['test_precision'].append(test_precision)
        log['test_recall'].append(test_recall)
        log['test_auc'].append(test_auc)

        class_log['val_log'].append(val_metric_class)
        class_log['test_log'].append(test_metric_class)

        scheduler.step()

    csv = pd.DataFrame(log)
    csv.columns = ['epoch', 'val_acc',
                   'val_f1', 'val_precision',
                   'val_recall', 'val_auc',
                   'test_acc',
                   'test_f1', 'test_precision',
                   'test_recall', 'test_auc']

    val_class_csv = pd.concat(class_log['val_log'], axis=0)
    test_class_csv = pd.concat(class_log['test_log'], axis=0)
    val_class_csv.to_csv(f'{args.checkpoint_dir}/' + args.name + '-' + args.backbone + '-B-' + str(
        batch_size) + args.dataset + 'R-' + str(args.ratio) + '-val-class.csv', index=False)
    test_class_csv.to_csv(f'{args.checkpoint_dir}/' + args.name + '-' + args.backbone + '-B-' + str(
        batch_size) + args.dataset + 'R-' + str(args.ratio) + '-test-class.csv', index=False)

    csv.to_csv(f'{args.checkpoint_dir}/' + args.name + '-' + args.backbone + '-B-' + str(
        batch_size) + args.dataset + 'R-' + str(args.ratio) + '.csv', index=False)

    print(f'max val acc: {max(log["val_acc"])}\n \
            max val f1: {max(log["val_f1"])}\n \
            max val precision: {max(log["val_precision"])}\n \
            max val recall: {max(log["val_recall"])}\n \
            max val auc: {max(log["val_auc"])}\n \
            max test acc: {max(log["test_acc"])}\n \
            max test f1: {max(log["test_f1"])}\n \
            max test precision: {max(log["test_precision"])}\n \
            max test recall: {max(log["test_recall"])}\n \
            max test auc: {max(log["test_auc"])}\n')
    print('final train time = %.4f' % (train_time / int(args.epochs)))
    print('final infer time = %.4f' % (infer_time / int(args.epochs)))

    # Save final predictions for error analysis
    print("\nSaving final predictions for error analysis...")
    pred_save_path = f'{args.checkpoint_dir}/predictions/{args.backbone}'
    infer(model, test_loader, args, save_predictions=True, save_path=pred_save_path)
    print(f"Predictions saved to {pred_save_path}")
    # plot each metric in one subplot
    plt.figure(figsize=(30, 10))
    plt.subplot(1, 3, 1)
    plt.plot(log['epoch'], log['val_acc'], label='val_acc')
    plt.plot(log['epoch'], log['test_acc'], label='test_acc')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(log['epoch'], log['val_f1'], label='val_f1')
    plt.plot(log['epoch'], log['test_f1'], label='test_f1')
    plt.legend()
    # plt.subplot(2, 2, 3)
    # since we donot compute precision and recall in there. so this figure is not useful.
    # plt.plot(log['epoch'], log['val_precision'], label='val_precision')
    # plt.plot(log['epoch'], log['test_precision'], label='test_precision')
    # plt.plot(log['epoch'], log['val_ecall'], label='val_recall')
    # plt.plot(log['epoch'], log['test_recall'], label='test_recall')
    # plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(log['epoch'], log['val_auc'], label='val_auc')
    plt.plot(log['epoch'], log['test_auc'], label='test_auc')
    plt.legend()
    plt.savefig(f'{args.checkpoint_dir}/' + args.name + '-' + args.backbone + '-B-' + str(
        batch_size) + args.dataset + 'R-' + str(args.ratio) + '.png')
    plt.close()


@torch.no_grad()
def infer(model, loader, args, save_predictions=False, save_path=None):
    # evaluate

    model.eval()

    y_pred = []

    y_true = []

    for ecg, target in tqdm(loader):

        input_label_list = target.to(device)

        predictions = model(ecg.to(device))
        y_true.append(input_label_list.cpu().detach().numpy())

        for index, val in enumerate(predictions):
            y_pred.append(val.cpu().detach().numpy().reshape(1, -1))

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    # Save predictions for error analysis
    if save_predictions and save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, 'y_true.npy'), y_true)
        np.save(os.path.join(save_path, 'y_pred.npy'), y_pred)
    auc = roc_auc_score(y_true, y_pred, average='macro')

    max_f1s = []
    accs = []
    ap_scores = []

    for i in range(y_pred.shape[1]):
        gt_np = y_true[:, i]
        pred_np = y_pred[:, i]
        precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
        numerator = 2 * recall * precision
        denom = recall + precision
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
        max_f1 = np.max(f1_scores)
        max_f1_thresh = thresholds[np.argmax(f1_scores)]
        max_f1s.append(max_f1)
        accs.append(accuracy_score(gt_np, pred_np > max_f1_thresh))

        ap_scores.append(average_precision_score(gt_np, pred_np))

    max_f1s = [i * 100 for i in max_f1s]
    accs = [i * 100 for i in accs]
    f1 = np.array(max_f1s).mean()
    acc = np.array(accs).mean()
    recall_test = np.array(ap_scores).mean()

    # we donot compute precision and recall in there.
    precision = 0

    class_name = args.labels_name

    metric_dict = {element: [] for element in class_name}

    for i in range(len(list(metric_dict.keys()))):
        key = list(metric_dict.keys())[i]
        metric_dict[key].append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    metric_class = pd.DataFrame(metric_dict)

    return acc, f1, precision, recall_test * 100, auc * 100, metric_class


if __name__ == '__main__':
    main()
