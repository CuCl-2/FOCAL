import random
import os
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
import sys

sys.path.append("../utils")
from utils_trainer import FOCALTrainer
import utils_dataset
import utils_builder

import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def ddp_main():
    dist.init_process_group("nccl")
    torch.cuda.empty_cache()
    rank      = dist.get_rank()
    device_id = rank % torch.cuda.device_count()

    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

    if device_id == 0:
        wandb.init(
            project="FOCAL",
            name=config['wandb_name'],
            config={
                'lr':           config['optimizer']['params']['lr'],
                'weight_decay': config['optimizer']['params']['weight_decay'],
                'max_epochs':   config['trainer']['max_epochs'],
                'batch_size':   config['trainer']['batch_size'],
                'ecg_model':    config['network']['ecg_model'],
                'text_model':   config['network']['text_model'],
                'uot_eps':      config['network'].get('uot_eps', 0.1),
                'uot_tau':      config['network'].get('uot_tau', 1.0),
            },
        )

    torch.manual_seed(42)
    random.seed(0)
    np.random.seed(0)

    # ── Dataset ──────────────────────────────────────────────────────────────
    data_path = config['dataset']['data_path']
    dataset   = utils_dataset.ECG_TEXT_Dsataset(
        data_path=data_path,
        dataset_name=config['dataset']['dataset_name'],
    )
    train_dataset = dataset.get_dataset(train_test='train')
    val_dataset   = dataset.get_dataset(train_test='val')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = utils_builder.FOCAL(config['network'])

    # Optionally freeze early BERT layers to stabilise training
    if config['network'].get('free_layers') is not None:
        n_freeze = int(config['network']['free_layers'])
        for layer_idx in range(n_freeze):
            for param in model.lm_model.encoder.layer[layer_idx].parameters():
                param.requires_grad = False
        if device_id == 0:
            print(f'Froze first {n_freeze} BERT encoder layers.')

    model = model.to(device_id)
    model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        **config['optimizer']['params'],
        betas=(0.9, 0.999),
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = FOCALTrainer(
        model=model,
        optimizer=optimizer,
        device=rank,
        model_name=config['wandb_name'],
        **config['trainer'],
    )

    trainer.train(train_dataset, val_dataset, config['zeroshot'])


if __name__ == '__main__':
    ddp_main()
