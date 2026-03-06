"""
FOCAL Trainer — distributed pre-training with DDP.

Training loop:
  1. Forward pass on each GPU: ECG → patch embeddings, report tags → tag embeddings,
     Semi-UOT → tag-specific ECG representations.
  2. All-gather tag_ecg_embs and tag_text_embs across GPUs (with padding for
     variable-length tag lists).
  3. Replace local slice with gradient-carrying tensors, detach others.
  4. Compute focal_loss on the full cross-GPU K_total × K_total matrix.
"""

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb

from utils_loss import focal_loss
from zeroshot_val import zeroshot_eval


def _cosine_warmup_schedule(optimizer, num_warmup_steps: int, num_total_steps: int):
    """Linear warmup then cosine decay scheduler."""
    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_total_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class FOCALTrainer:
    def __init__(self, model, optimizer, device, model_name, **args):
        self.model       = model
        self.optimizer   = optimizer
        self.device      = device   # rank
        self.model_name  = model_name

        self.train_batch_size    = args['batch_size']
        self.val_batch_size      = args['val_batch_size']
        self.max_epochs          = args['max_epochs']
        self.num_workers         = args['num_workers']
        self.checkpoint_interval = args['checkpoint_interval']
        # Maximum tags we allocate per report for cross-GPU gather (padding)
        self.max_tags_per_report = args.get('max_tags_per_report', 15)

    # ── Distributed gather helpers ──────────────────────────────────────────

    def _gather_variable_embs(self, local_embs: torch.Tensor,
                               local_mask: torch.Tensor):
        """
        All-gather variable-length tag embeddings across GPUs.

        Each GPU pads its embeddings to max_K slots with zeros and
        passes a boolean validity mask.  After gathering, the valid
        rows are extracted and the local slice is replaced with the
        gradient-carrying original tensor.

        Args:
            local_embs : (K_local, d) — local embeddings (with gradients)
            local_mask : (max_K,)     — True for valid rows
        Returns:
            global_embs : (K_total, d) — all valid embeddings across GPUs
                          gradient flows only through the local slice
        """
        world_size = dist.get_world_size()
        rank       = dist.get_rank()
        max_K, d   = local_mask.shape[0], local_embs.shape[1]

        # Padded buffer for this GPU
        padded = torch.zeros(max_K, d, device=local_embs.device,
                             dtype=local_embs.dtype)
        K_local = local_embs.shape[0]
        padded[:K_local] = local_embs.detach()

        # All-gather padded embeddings and masks
        all_padded = [torch.zeros_like(padded) for _ in range(world_size)]
        all_masks  = [torch.zeros_like(local_mask) for _ in range(world_size)]
        dist.all_gather(all_padded, padded)
        dist.all_gather(all_masks,  local_mask)

        # Extract valid rows; replace local slice with gradient version
        parts = []
        for i in range(world_size):
            valid = all_padded[i][all_masks[i]]   # (K_i, d) — detached
            if i == rank:
                valid = local_embs                 # (K_local, d) — with grad
            parts.append(valid)

        return torch.cat(parts, dim=0)   # (K_total, d)

    # ── Main training entry ─────────────────────────────────────────────────

    def train(self, train_dataset, val_dataset, args_zeroshot_eval):
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=False,
            sampler=DistributedSampler(train_dataset),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=False,
            sampler=DistributedSampler(val_dataset),
        )

        ckpt_dir = '../checkpoints/'
        if self.device == 0:
            os.makedirs(ckpt_dir, exist_ok=True)

        # Resume from checkpoint
        ckpt_path   = ckpt_dir + self.model_name + '_checkpoint.pth'
        start_epoch = 0
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            start_epoch = ckpt['epoch']
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if self.device == 0:
                print(f'Resumed from epoch {start_epoch}')
        else:
            if self.device == 0:
                print('Training from scratch (epoch 0)')

        # Scheduler: linear warmup (10%) + cosine decay
        steps_per_epoch = len(train_loader)
        total_steps     = self.max_epochs * steps_per_epoch
        warmup_steps    = int(0.10 * total_steps)
        scheduler = _cosine_warmup_schedule(
            self.optimizer, warmup_steps, total_steps)

        scaler   = GradScaler()
        best_auc = 0.0
        global_step = start_epoch * steps_per_epoch

        max_K = self.train_batch_size * self.max_tags_per_report

        for epoch in tqdm(range(start_epoch, self.max_epochs)):
            self.model.train()
            train_loader.sampler.set_epoch(epoch)

            epoch_loss = 0.0
            num_steps  = 0

            for data in tqdm(train_loader, leave=False):
                self.model.train()
                reports = data['raw_text']          # list[str], len = B
                ecg     = data['ecg'].to(torch.float32).to(self.device)

                self.optimizer.zero_grad()

                with autocast():
                    out = self.model(ecg, reports)
                    tag_ecg_embs  = out['tag_ecg_embs']    # (K_local, d)
                    tag_text_embs = out['tag_text_embs']   # (K_local, d)
                    K_local = tag_ecg_embs.shape[0]

                    # Build validity mask for padding
                    valid_mask = torch.zeros(max_K, dtype=torch.bool,
                                             device=self.device)
                    valid_mask[:K_local] = True

                    # Cross-GPU gather
                    global_ecg_embs  = self._gather_variable_embs(
                        tag_ecg_embs, valid_mask)
                    global_text_embs = self._gather_variable_embs(
                        tag_text_embs, valid_mask)

                    loss = focal_loss(
                        global_ecg_embs,
                        global_text_embs,
                        self.model.module.log_temperature,
                    )

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                scheduler.step()

                loss_val = loss.item()
                epoch_loss += loss_val
                num_steps  += 1
                global_step += 1

                if self.device == 0:
                    wandb.log({
                        'train_step_loss': loss_val,
                        'temperature':     self.model.module.log_temperature.exp().item(),
                        'lr':              self.optimizer.param_groups[0]['lr'],
                    }, step=global_step)

            # ── Validation ────────────────────────────────────────────────
            val_loss = self._val(val_loader)

            if self.device == 0:
                avg_train_loss = epoch_loss / max(1, num_steps)
                print(f'Epoch {epoch}: train_loss={avg_train_loss:.4f}  '
                      f'val_loss={val_loss:.4f}')
                wandb.log({
                    'train_epoch_loss': avg_train_loss,
                    'val_epoch_loss':   val_loss,
                    'epoch':            epoch,
                }, step=global_step)

                # Zero-shot evaluation
                avg_auc = 0.0
                for set_name in args_zeroshot_eval['val_sets']:
                    _, _, auc, _, _, _, _ = zeroshot_eval(
                        model=self.model,
                        set_name=set_name,
                        device=self.device,
                        args_zeroshot_eval=args_zeroshot_eval,
                    )
                    avg_auc += auc
                    wandb.log({f'{set_name}_AUROC': auc}, step=global_step)

                avg_auc /= len(args_zeroshot_eval['val_sets'])
                wandb.log({'avg_auc': avg_auc}, step=global_step)

                # Save best checkpoint
                if avg_auc > best_auc:
                    best_auc = avg_auc
                    torch.save(self.model.module.state_dict(),
                               ckpt_dir + self.model_name + '_best_ckpt.pth')
                    torch.save(self.model.module.ecg_encoder.state_dict(),
                               ckpt_dir + self.model_name + '_best_encoder.pth')

                if epoch % self.checkpoint_interval == 0:
                    self._save_checkpoint(
                        epoch, ckpt_dir + self.model_name + f'_{epoch}_ckpt.pth')

        # Save final weights
        if self.device == 0:
            torch.save(self.model.module.state_dict(),
                       ckpt_dir + self.model_name + '_final_total.pth')
            torch.save(self.model.module.ecg_encoder.state_dict(),
                       ckpt_dir + self.model_name + '_final_encoder.pth')

    # ── Validation ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _val(self, loader) -> float:
        self.model.eval()
        total_loss = 0.0
        num_steps  = 0

        for data in tqdm(loader, leave=False):
            reports = data['raw_text']
            ecg     = data['ecg'].to(torch.float32).to(self.device)

            out = self.model(ecg, reports)
            loss = focal_loss(
                out['tag_ecg_embs'],
                out['tag_text_embs'],
                self.model.module.log_temperature,
            )
            total_loss += loss.item()
            num_steps  += 1

        return total_loss / max(1, num_steps)

    # ── Checkpoint ──────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, path: str):
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
