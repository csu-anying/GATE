# package import
import os
from typing import Type

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import yaml as yaml

from utils.util_loss import clip_loss, graph_contrastive_loss, cosine_consistency_loss
from zeroshot.zeroshot_eval import zeroshot_eval
import wandb


class trainer_Graph_Text:
    def __init__(self, model,
                 optimizer, device, model_name, **args):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name
        self.train_batch_size = args['batch_size']
        self.max_epochs = args['max_epochs']
        self.num_workers = args['num_workers']
        self.checkpoint_interval = args['checkpoint_interval']
        self.val_batch_size = args['val_batch_size']

    def train_graph_text(self, train_dataset, val_dataset, args_zeroshot_eval):
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size,
                                  num_workers=self.num_workers,
                                  drop_last=True, shuffle=True)

        val_loader = DataLoader(val_dataset, batch_size=self.val_batch_size,
                                num_workers=self.num_workers,
                                drop_last=True, shuffle=False)
        model_checkpoints_folder = os.path.join('../checkpoints/pretrain/')
        if not os.path.exists(model_checkpoints_folder):
            print(f'Creating directory "{model_checkpoints_folder}" for saving checkpoint!')
            os.makedirs(model_checkpoints_folder)
        else:
            print(f'Directory "{model_checkpoints_folder}" already exists for saving checkpoint!')

        # automatically resume from checkpoint if it exists
        print('#########################################')
        print('Checking for checkpoint...')
        if os.path.exists(model_checkpoints_folder + self.model_name + '_checkpoint.pth'):
            ckpt = torch.load(model_checkpoints_folder + self.model_name + '_checkpoint.pth',
                              map_location='cpu')
            start_epoch = ckpt['epoch']
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print('Continuing training from checkpoint.')
        else:
            start_epoch = 0
            print('Starting training from epoch 0.')

        print('#########################################')
        print('Training started.')

        # scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5000,
            T_mult=1,
            eta_min=1e-8,
        )
        niter = 1

        scaler = GradScaler()

        f1_total = []
        acc_total = []
        auc_total = []

        zeroshot_csv = pd.DataFrame()
        best_auc = 0
        best_val_acc = 0

        for epoch_counter in range(start_epoch, self.max_epochs + 1):

            epoch_loss = 0
            epoch_uma_loss = 0
            epoch_cma_loss = 0
            epoch_acc1 = []
            epoch_acc5 = []
            self.model.train()
            for step, data in enumerate(train_loader):
                report = data['raw_text']
                ecg = data['ecg'].to(torch.float32).to(self.device).contiguous()

                self.optimizer.zero_grad()

                with autocast():
                    report_tokenize_output = self.model.module._tokenize(report)

                    input_ids = report_tokenize_output.input_ids.to(self.device).contiguous()
                    attention_mask = report_tokenize_output.attention_mask.to(self.device).contiguous()

                    output_dict = self.model(ecg, input_ids, attention_mask, mode='train')
                    ecg_emb, proj_ecg_emb, proj_text_emb = output_dict['ecg_emb'], \
                        output_dict['proj_ecg_emb'], \
                        output_dict['proj_text_emb']

                    # Aggregating embeddings across all batches (no dist.all_gather in single-device mode)
                    agg_proj_img_emb = proj_ecg_emb[0]
                    agg_proj_text_emb = proj_text_emb[0]
                    agg_proj_ecg_emb1 = ecg_emb[0]
                    agg_proj_ecg_emb2 = ecg_emb[1]
                    # print("img:{}, text:{}".format(agg_proj_img_emb.shape, agg_proj_text_emb.shape))
                    cma_loss, acc1, acc5 = clip_loss(agg_proj_img_emb, agg_proj_text_emb, device=self.device)
                    uma_loss = graph_contrastive_loss(agg_proj_ecg_emb1, agg_proj_ecg_emb2)
                    # Cosine Similarity Loss
                    con_loss = cosine_consistency_loss(agg_proj_img_emb, agg_proj_ecg_emb1, agg_proj_ecg_emb2)
                    loss = cma_loss + 0.2 * uma_loss + 0.1 * con_loss

                    print(f'loss is {loss.item()}, acc1 is {acc1.item()}, acc5 is {acc5.item()}, '
                          f'cma_loss is {cma_loss.item()}, uma_loss is {uma_loss.item()}')

                    # Log to wandb
                    wandb.log({
                        'train_step_uma_loss': uma_loss.item(),
                        'train_step_cma_loss': cma_loss.item(),
                        'train_step_con_loss': con_loss.item(),
                        'train_step_total_loss': loss.item(),
                        'train_step_acc1': acc1.item(),
                        'train_step_acc5': acc5.item()
                    })

                    epoch_loss += loss.item()
                    epoch_uma_loss += uma_loss.item()
                    epoch_cma_loss += cma_loss.item()
                    epoch_acc1.append(acc1.item())
                    epoch_acc5.append(acc5.item())

                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                    scheduler.step()
                niter += 1

            # Validation stage
            val_log = self.val(val_loader)

            # Log training and validation metrics
            epoch_acc1 = np.array(epoch_acc1).mean()
            epoch_acc5 = np.array(epoch_acc5).mean()

            epoch_iter = (len(train_dataset) // self.train_batch_size)
            print(
                f'{epoch_counter} epoch loss is {epoch_loss / epoch_iter}, acc1 is {epoch_acc1}, acc5 is {epoch_acc5}')

            # Log to wandb
            wandb.log({
                'train_epoch_loss': epoch_loss / epoch_iter,
                'train_epoch_uma_loss':epoch_uma_loss / epoch_iter,
                'train_epoch_cma_loss': epoch_cma_loss / epoch_iter,
                'train_epoch_acc1': epoch_acc1,
                'train_epoch_acc5': epoch_acc5,
                'val_cma_loss': val_log['val_cma_loss'],
                'val_epoch_loss': val_log['val_loss'],
                'val_epoch_acc1': val_log['val_acc1'],
                'val_epoch_acc5': val_log['val_acc5']
            })

            # Save best checkpoint based on validation accuracy
            if val_log['val_acc1'] > best_val_acc:
                best_val_acc = val_log['val_acc1']
                torch.save(self.model.state_dict(),
                           model_checkpoints_folder + self.model_name + f'_best_ckpt.pth')

            # Zero-shot eval======================================
            avg_f1, avg_acc, avg_auc = 0, 0, 0
            for set_name in args_zeroshot_eval['val_sets'].keys():
                f1, acc, auc, _, _, _, res_dict = \
                    zeroshot_eval(model=self.model,
                                  set_name=set_name,
                                  device=self.device,
                                  args_zeroshot_eval=args_zeroshot_eval)

                avg_f1 += f1
                avg_acc += acc
                avg_auc += auc

                # Log each val set zeroshot performance
                wandb.log({
                    f'{set_name}_f1': f1,
                    f'{set_name}_acc': acc,
                    f'{set_name}_AUROC': auc
                })

            avg_f1 = avg_f1 / len(args_zeroshot_eval['val_sets'].keys())
            avg_acc = avg_acc / len(args_zeroshot_eval['val_sets'].keys())
            avg_auc = avg_auc / len(args_zeroshot_eval['val_sets'].keys())

            # Log average performance
            wandb.log({
                'avg_f1': avg_f1,
                'avg_acc': avg_acc,
                'avg_auc': avg_auc
            })

            f1_total.append(f1)
            acc_total.append(acc)
            auc_total.append(auc)

            best_metric = avg_auc
            if best_metric > best_auc:
                best_auc = best_metric
                torch.save(self.model.state_dict(),
                           model_checkpoints_folder + self.model_name + f'_bestZeroShotAll_ckpt.pth')

            if epoch_counter % self.checkpoint_interval == 0:
                self.save_checkpoints(epoch_counter,
                                      model_checkpoints_folder + self.model_name + f'_{epoch_counter}_ckpt.pth')

        # Save final model and encoder
        torch.save(self.model.state_dict(),
                   model_checkpoints_folder + self.model_name + '_final_total.pth')

    def val(self, loader):
        print('=======================Starting validation=======================')
        self.model.eval()
        val_cma_loss = 0
        val_loss = 0
        val_epoch_acc1 = []
        val_epoch_acc5 = []

        for step, data in enumerate(loader):
            # get raw text
            report = data['raw_text']
            # get ecg
            ecg = data['ecg'].to(torch.float32).to(self.device).contiguous()

            report_tokenize_output = self.model.module._tokenize(report)

            input_ids = report_tokenize_output.input_ids.to(self.device).contiguous()
            attention_mask = report_tokenize_output.attention_mask.to(self.device).contiguous()

            with torch.no_grad():
                output_dict = self.model(ecg, input_ids, attention_mask, mode='test')
                proj_ecg_emb, proj_text_emb = output_dict['proj_ecg_emb'], output_dict['proj_text_emb']

                agg_proj_img_emb = proj_ecg_emb[0]
                agg_proj_text_emb = proj_text_emb[0]

                cma_loss, acc1, acc5 = clip_loss(agg_proj_img_emb, agg_proj_text_emb, device=self.device)

                val_cma_loss += cma_loss.item()
                val_loss += cma_loss.item()
                val_epoch_acc1.append(acc1.item())
                val_epoch_acc5.append(acc5.item())

        val_cma_loss /= len(loader)
        val_loss /= len(loader)
        val_epoch_acc1 = np.array(val_epoch_acc1).mean()
        val_epoch_acc5 = np.array(val_epoch_acc5).mean()

        print(f'Validation loss: {val_loss}, acc1: {val_epoch_acc1}, acc5: {val_epoch_acc5}')
        return {
            'val_cma_loss': val_cma_loss,
            'val_loss': val_loss,
            'val_acc1': val_epoch_acc1,
            'val_acc5': val_epoch_acc5
        }

    def save_checkpoints(self, epoch, checkpoint_path):
        print(f"Saving checkpoint at epoch {epoch}...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
