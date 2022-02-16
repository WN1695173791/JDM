import os
import logging
import copy
from tqdm import trange
from datetime import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from utils import ema
from lib.dataset import DataLooper 
from lib.sde import VPSDE
from lib.model.unet import UNet
from lib.trainer import DiffusionTrainer
from lib.sampler import DiffusionSampler


def train(config, logdir, resume=True):
    """Running a training pipeline"""
    # Dataset setup
    datalooper = DataLooper(
        config,
        batch_size=config.train.batch_size,
    )
    eval_datalooper = DataLooper(
        config,
        batch_size=config.eval.batch_size,
    )

    # Model setup
    if config.model.name == 'unet':
        net_model = UNet(
            config.dataset.x_ch,
            config.dataset.y_ch,
            config.model.ch,
            config.model.ch_mult,
            config.model.attn,
            config.model.num_res_blocks,
            config.model.dropout,
        )
    else:
        raise ValueError

    ema_model = copy.deepcopy(net_model)

    if config.parallel:
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)

    # SDE setup
    if config.sde.name == 'VPSDE':
        sde = VPSDE(
            config.sde.beta_min,
            config.sde.beta_max,
            config.sde.N,
        )
    else:
        raise ValueError

    # Trainer setup
    trainer = DiffusionTrainer(
        sde,
        net_model,
        config.model.pred_type,
    ).to(config.device)
    trainer.train()

    # Optimizer setup
    optim = torch.optim.Adam(
        net_model.parameters(),
        lr=config.train.lr,
    )
    warmup = config.train.warmup
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lambda step: min(step, warmup) / warmup,
    )

    # Sampler setup
    sampler = DiffusionSampler(
        sde,
        ema_model,
        config.model.pred_type,
    ).to(config.device)
    sampler.eval()
    
    # Log setup 
    sample_dir = os.path.join(logdir, 'samples')
    os.makedirs(sample_dir, exist_ok=True)
    writer = SummaryWriter(logdir)

    # Show model size
    model_size = sum(p.numel() for p in net_model.parameters())
    logging.info(f'Model Params : {model_size / 1024 / 1024:.2f}M')

    # Load checkpoint (if exists)
    try:
        assert resume
        ckpt = torch.load(os.path.join(logdir, f'ckpt_latest.pt'))
        net_model.load_state_dict(ckpt['net_model'])
        ema_model.load_state_dict(ckpt['ema_model'])
        optim.load_state_dict(ckpt['optimizer'])
        sched.load_state_dict(ckpt['scheduler'])
        init_step = ckpt['step'] + 1
        logging.info(f'Checkpoint loaded! Re-start from step {init_step}.')
    except:
        init_step = 0
        logging.info(f'No checkpoint found. Start from step {init_step}.')

    # Start training
    with trange(init_step, config.train.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # Train
            optim.zero_grad()
            x_0, y_0 = next(datalooper)
            x_0 = x_0.to(config.device)
            y_0 = y_0.to(config.device)
            c = torch.ones(config.train.batch_size) * 0.5
            c = torch.bernoulli(c)[:, None, None, None]
            x_loss, y_loss = trainer(x_0, y_0, c)
            loss = torch.cat([x_loss, y_loss], dim=1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(),
                config.train.grad_clip,
            )
            optim.step()
            sched.step()
            ema(net_model, ema_model, config.train.ema_decay)

            # Log
            writer.add_scalar('loss', loss, step)
            writer.add_scalar('x_loss', x_loss.mean(), step)
            writer.add_scalar('y_loss', y_loss.mean(), step)
            pbar.set_postfix(loss=f'{loss:.3f}')

            # Sample
            if config.train.sample_step > 0 and step % config.train.sample_step == 0:
                for option in ['joint', 'x->y', 'y->x']:
                    os.makedirs(os.path.join(sample_dir, option), exist_ok=True)
                    pbar.set_postfix_str(f'{option} start')
                    xs, ys = [], []
                    total_steps = config.eval.sample_size // config.eval.batch_size
                    for i in range(0, config.eval.sample_size, config.eval.batch_size):
                        x_0, y_0 = next(eval_datalooper)
                        x_0 = x_0.to(config.device)
                        y_0 = y_0.to(config.device)
                        x_T = torch.randn_like(x_0)
                        y_T = torch.randn_like(y_0)

                        if option == 'joint':
                            x_0 = None
                            y_0 = None
                        elif option == 'x->y':
                            y_0 = None
                        elif option == 'y->x':
                            x_0 = None

                        with torch.no_grad():
                            x, y = sampler(x_T, y_T, x_0, y_0, option, pbar)

                        xs.append((x.detach().cpu() + 1.) / 2)
                        ys.append((y.detach().cpu() + 1.) / 2)
                        pbar.set_postfix(option=f'({i+1}/{total_steps})')
                    xs = torch.cat(xs, dim=0)
                    ys = torch.cat(ys, dim=0)
                    save_image(
                        xs[:64],
                        os.path.join(sample_dir, option, f'x_{step}.png'),
                        nrow=8,
                    )
                    save_image(
                        ys[:64],
                        os.path.join(sample_dir, option, f'y_{step}.png'),
                        nrow=8,
                    )

            # Save
            if config.train.save_step > 0 and step % config.train.save_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'scheduler': sched.state_dict(),
                    'step': step,
                }
                torch.save(ckpt, os.path.join(logdir, f'ckpt_latest.pt'))

            # Archive
            if config.train.archive_step > 0 and step % config.train.archive_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'scheduler': sched.state_dict(),
                    'step': step,
                }
                torch.save(ckpt, os.path.join(logdir, f'ckpt_{step}.pt'))

    writer.close()


def eval(config, logdir):
    """Running an evaluation pipeline"""
    # Datalooper setup
    eval_datalooper = DataLooper(
        config,
        batch_size=config.eval.batch_size,
    )
    sample_size = config.eval.sample_size
    batch_size = config.eval.batch_size

    # Model setup
    if config.model.name == 'unet':
        model = UNet(
            config.dataset.x_ch,
            config.dataset.y_ch,
            config.model.ch,
            config.model.ch_mult,
            config.model.attn,
            config.model.num_res_blocks,
            config.model.dropout,
        )
    else:
        raise ValueError

    if config.parallel:
        model = torch.nn.DataParallel(model)
    
    # SDE setup
    if config.sde.name == 'VPSDE':
        sde = VPSDE(
            config.sde.beta_min,
            config.sde.beta_max,
            config.sde.N,
        )
    else:
        raise ValueError

    # Sampler setup
    sampler = DiffusionSampler(
        sde,
        model,
        config.model.pred_type,
    ).to(config.device)
    sampler.eval()

    # Show model size
    model_size = sum(p.numel() for p in model.parameters())
    logging.info(f'Model Params : {model_size / 1024 / 1024:.2f}M')

    # Load checkpoint
    ckpt = torch.load(
        os.path.join(logdir, f'ckpt_latest.pt'),
        map_location=config.device
    )
    logging.info(f'Checkpoint step : {ckpt["step"]}')
    model.load_state_dict(ckpt['ema_model'])

    # Directory setup
    eval_dir = os.path.join(logdir, 'eval')
    sample_dir = os.path.join(eval_dir, 'samples')
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    for option in ['joint', 'x->y', 'y->x']:
        os.makedirs(os.path.join(sample_dir, option), exist_ok=True)
        xs, ys = [], []
        with trange(0, sample_size, batch_size, dynamic_ncols=True) as pbar:
            for _ in pbar:
                x_0, y_0 = next(eval_datalooper)
                x_0 = x_0.to(config.device)
                y_0 = y_0.to(config.device)
                x_T = torch.randn_like(x_0)
                y_T = torch.randn_like(y_0)

                if option == 'joint':
                    x_0 = None
                    y_0 = None
                elif option == 'x->y':
                    y_0 = None
                elif option == 'y->x':
                    x_0 = None

                with torch.no_grad():
                    x, y = sampler(x_T, y_T, x_0, y_0, option, pbar)

                xs.append((x.detach().cpu() + 1.) / 2)
                ys.append((y.detach().cpu() + 1.) / 2)
        xs = torch.cat(xs, dim=0)
        ys = torch.cat(ys, dim=0)
        now = datetime.now()
        save_image(
            xs[:64],
            os.path.join(sample_dir, option, f'x_{now}.png'),
            nrow=8,
        )
        save_image(
            ys[:64],
            os.path.join(sample_dir, option, f'y_{now}.png'),
            nrow=8,
        )