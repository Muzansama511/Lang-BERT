
import os
import copy
import random
 
import numpy as np
import torch
 
import lightning as L
# from lightning.pytorch.loggers import LitLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
# ── your own modules — adjust these imports to match your actual filenames ──
from Dataset import get_dataloader, get_tiktokenizer
from Model.model import JEA


def main():
    # ── config (swap this block for Hydra later if you want sweeps) ──
    num_workers = 4
    max_steps = -1
    val_check_interval = 10000
    lr = 1e-4
    ema_decay = 0.99
    ema_update_every = 20
    precision = "32"
    accumulate_grad_batches = 1
    warmup = 10000
    hidden = 256
    num_layers = 8
    use_deepnet_init = False  # Set to True if you want to use DeepNet initialization
    have_teacher = False  # Set to True if you want to use a teacher encoder
    
    # ── tokenizer / special tokens ──
    enc, meta = get_tiktokenizer()
    cls_id = meta["CLS_ID"]
    sep_id = meta["SEP_ID"]
    mask_id = meta["MASK_ID"]
    pad_id = meta["PAD_ID"]
    vocab_size = enc.n_vocab
 
    # ── data ──
    train_loader = get_dataloader(train_or_val="train", batch_size=4, num_workers=num_workers, max_length=128, perc_to_use=0.01)
    val_loader = get_dataloader(train_or_val="val", batch_size=4, num_workers=num_workers, max_length=128, perc_to_use=0.0001)
 
    # ── model ──
    lit_model = JEA(hidden_dim=hidden, vocab_size=vocab_size, num_layers = num_layers, ema_decay=ema_decay, ema_update_every=ema_update_every, lr=lr, have_teacher=have_teacher, warmup_steps=warmup, use_deepnet_init=use_deepnet_init)
    
    # # ── logging ──
    # wandb_logger = WandbLogger(
    #     project="world-model-encoder",
    #     name="run-cls-mlm-v1",
    #     log_model=True,
    # )
 
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="step{step}-valloss{val_loss:.3f}",
        monitor="val_loss",
        save_top_k=3,
        save_last=True,
        every_n_train_steps=val_check_interval,
    )
 
    logger = TensorBoardLogger(
    save_dir="logs/",
    name="jea"
    )
 
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # ── trainer ──
    trainer = L.Trainer(
        max_steps=max_steps,
        accelerator="gpu",
        devices=1,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=100,
        # limit_val_batches=0.0,
        val_check_interval=val_check_interval,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir="checkpoints/",
    )

    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()