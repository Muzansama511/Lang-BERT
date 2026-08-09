import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchview import draw_graph
from torch.optim.lr_scheduler import LambdaLR
import math
from time import sleep
# from flash_attn.losses.cross_entropy import CrossEntropyLoss as FlashCrossEntropyLoss

try:
    from .BERTmodel import *
except:
    from BERTmodel import *

'''
The idea is that the Online Encoder gets the masked text, encodes and a loss acts on predicting the masked text. The Teacher Encoder gets the unmasked text and encodes it. The loss is computed between the two CLS encodings and the Teacher Encoder is the EMA of Online Encoder.
'''



class OnlineEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 12, use_deepnet_init: bool = True):
        super(OnlineEncoder, self).__init__()
        self.encoder = BERT(vocab_size=input_dim, hidden=hidden_dim, n_layers=num_layers, attn_heads=2, dropout=0.1, base=256, use_deepnet_init=use_deepnet_init)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_enc = self.encoder(x)
        x = self.predictor(x_enc)
        # print('x_enc', x_enc.shape, 'x', x.shape)
        return x_enc, x

class TeacherEncoder(OnlineEncoder):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 12, use_deepnet_init: bool = True):
        super(TeacherEncoder, self).__init__(input_dim, hidden_dim, output_dim, num_layers=num_layers, use_deepnet_init=use_deepnet_init)


class JEA(L.LightningModule):
    def __init__(self, hidden_dim: int = 64, vocab_size: int = 100280, num_layers: int = 12,  ema_decay: float = 0.99, ema_update_every: int = 100, lr: float = 1e-4, warmup_steps=500, compile_mode=True, have_teacher = False, use_deepnet_init=False):
        super(JEA, self).__init__()
        self.vocab_size = vocab_size
        # self.global_step = 0
        self.use_deepnet_init = use_deepnet_init
        if compile_mode:
            self.online_encoder = torch.compile(OnlineEncoder(self.vocab_size, hidden_dim, self.vocab_size, num_layers, use_deepnet_init=self.use_deepnet_init))
            if have_teacher:
                self.teacher_encoder = torch.compile(TeacherEncoder(self.vocab_size, hidden_dim, self.vocab_size, num_layers, use_deepnet_init=self.use_deepnet_init))
                for p in self.teacher_encoder.parameters():
                    p.requires_grad = False  # Teacher encoder is not trained directly
        else:
            self.online_encoder = (OnlineEncoder(self.vocab_size, hidden_dim, self.vocab_size, num_layers, use_deepnet_init=self.use_deepnet_init))
            if have_teacher:
                self.teacher_encoder = (TeacherEncoder(self.vocab_size, hidden_dim, self.vocab_size, num_layers, use_deepnet_init=self.use_deepnet_init))
                for p in self.teacher_encoder.parameters():
                    p.requires_grad = False  # Teacher encoder is not trained directly
        self.ema_decay = ema_decay
        self.ema_update_every = ema_update_every
        # self.CELoss = FlashCrossEntropyLoss(ignore_index=-100, reduction="mean", inplace_backward=True)
        self.CELoss = nn.CrossEntropyLoss(reduction="mean")
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.have_teacher = have_teacher
        self.save_hyperparameters(ignore=["model"])
    
    def loss_fn(self, online_encoding: torch.Tensor, online_output: torch.Tensor, teacher_encoding: torch.Tensor, mask: torch.Tensor, ground_true_labels: torch.Tensor) -> torch.Tensor:
        # Compute the loss between online and teacher outputs

        if self.hparams.have_teacher:
            CLS_loss = F.mse_loss(online_encoding[..., 0], teacher_encoding[..., 0], reduction='mean')
        else:
            CLS_loss = 0.0
        online_pred = online_output
        online_pred = torch.masked_select(online_pred, (mask.unsqueeze(-1).expand_as(online_pred)).to(bool))
        # print('online_pred', online_pred.shape)
        ground_true_labels = torch.masked_select(ground_true_labels, (mask).to(bool))
        # print('vs',self.vocab_size)
        online_pred = online_pred.reshape(-1, self.vocab_size)
        ground_true_labels = ground_true_labels.reshape(-1)
        # print('online, ground truth labels',online_pred.shape, ground_true_labels.shape)
        masked_loss = self.CELoss(online_pred, ground_true_labels)
        return {"CLS_loss": 0.0*CLS_loss, "masked_loss": masked_loss} # Currently set CLS_loss set to 0 to check masked recovery

        
    def forward(self, x_online: torch.Tensor, x_teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x_teacher == ground_true_labels
        # print('x_online', x_online[0,:10], 'x_teacher', x_teacher[0,:10], 'mask', mask[0,:10])
        online_encoding, online_output = self.online_encoder(x_online)
        # print('online_encoding', online_encoding.shape, 'online_output', online_output.shape)
        if self.hparams.have_teacher:
            with torch.no_grad():
                teacher_encoding, _ = self.teacher_encoder(x_teacher)
            return self.loss_fn(online_encoding, online_output, teacher_encoding, mask, x_teacher)
        else:
            return self.loss_fn(online_encoding, online_output, online_encoding, mask, x_teacher)

    def training_step(self, batch, batch_idx):
        x_online, ground_true_labels, mask = batch['output'], batch['input'], batch['consider_for_loss']
        out_dict = self.forward(x_online, ground_true_labels, mask)
        loss = sum([v for k,v in out_dict.items() if 'SUM_IGNORE' not in k])
        
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            )
        for key, value in out_dict.items():
            if 'LOG_IGNORE' in key:
                continue
            self.log(
                f"train_{key}",
                value,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )
            # input()
            # sleep(1)
        # self.global_step += 1
        return loss

    def validation_step(self, batch, batch_idx):
        x_online, ground_true_labels, mask = batch['output'], batch['input'], batch['consider_for_loss']
        out_dict = self.forward(x_online, ground_true_labels, mask)
        loss = sum([v for k,v in out_dict.items() if 'SUM_IGNORE' not in k])
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        for key, value in out_dict.items():
            if 'LOG_IGNORE' in key:
                continue
            self.log(
                f"val_{key}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
        return loss
    
    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.online_encoder.parameters(), lr=self.lr)
    #     return optimizer

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.online_encoder.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.95),   # lower beta2 helps early stability
            weight_decay=0.01,
        )

        def lr_lambda(step):
            warmup = self.hparams.warmup_steps
            total = self.trainer.estimated_stepping_batches
            if step < warmup:
                # linear warmup
                return step / max(1, warmup)
            # cosine decay after warmup
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",   # step-wise, not epoch-wise
                "frequency": 1,
            },
        }
        
    def on_before_optimizer_step(self, optimizer):
        # log per-parameter average gradient
        for name, param in self.named_parameters():
            # if 'linear_layers' in name:
            if param.grad is not None:
                self.log(
                    f"grad_mean/{name}",
                    param.grad.norm(), # gradient norm
                    on_step=True,
                    on_epoch=False,
                )
    # def on_train_epoch_end(self):
    #     print('Epoch end')
    
    @torch.no_grad()
    def update_teacher(self):
        for online_param, teacher_param in zip(self.online_encoder.parameters(), self.teacher_encoder.parameters()):
            teacher_param.data.mul_(self.ema_decay).add_(online_param.data * (1 - self.ema_decay))

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # print('batch_end')
        if self.hparams.have_teacher:
            if self.global_step % self.ema_update_every == 0:
                self.update_teacher()

if __name__ == "__main__":
    
    model = JEA(
        hidden_dim=128,
        vocab_size=100280,
        have_teacher=False,
        compile_mode=False
    )

    model.eval()

    x_online = torch.randint(
        0, model.vocab_size,
        (4, 128),
        dtype=torch.long,
    )

    x_teacher = torch.randint(
        0, model.vocab_size,
        (4, 128),
        dtype=torch.long,
    )

    mask = torch.ones(
        (4, 128),
        dtype=torch.bool,
    )

    graph = draw_graph(
        model,
        input_data=(x_online, x_teacher, mask),
        expand_nested=True,
    )

    graph.visual_graph.render(
        "jea_architecture_without_teacher",
        format="png",
    )