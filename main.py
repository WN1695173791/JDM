import os
import copy
from absl import app, flags
import numpy as np
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from torchvision.utils import save_image
from torchvision.datasets import CIFAR10, MNIST
from tqdm import trange, tqdm

from utils import *
from lib.diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from lib.model.vit import ViT
from lib.model.unet import UNet


FLAGS = flags.FLAGS
FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='evaluate with various options')
flags.DEFINE_string('eval_option', 'generation', help='options for evaluation')
# Both ViT and UNet
flags.DEFINE_string('model_name', 'vit', help='vit or unet')
flags.DEFINE_integer('image_size', 32, help='image size')
flags.DEFINE_integer('num_classes', 10, help='number of classes')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
flags.DEFINE_float('x_dropout', 0.5, help='dropout rate of input x')
flags.DEFINE_float('y_dropout', 0.9, help='dropout rate of input y')
flags.DEFINE_bool('logsnr', True, help='sample t considering logsnr')
flags.DEFINE_string('y_modal', 'different', help='use same modal: same, else: different')
# ViT image_size, patch_size, num_classes, T, dim, depth, heads, mlp_dim
flags.DEFINE_integer('patch_size', 2, help='patch size')
flags.DEFINE_integer('dim', 64, help='hidden layer dimension')
flags.DEFINE_integer('depth', 12, help='depth of transformer module')
flags.DEFINE_integer('heads', 8, help='number of multi-attention head')
flags.DEFINE_integer('mlp_dim', 256, help='FFLN layer hidden dimension')
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lmbda', 1e0, help='Reflection rate of class loss/generator loss')
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 800000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10_EPS', help='log directory')
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
flags.DEFINE_integer('n_corrector', 1, help='the number of corrector steps per one predictor step')
flags.DEFINE_float('snr', 0.01, help='target snr from ScoreSDE. Refer to Table 5 in ScoreSDE.')
# Evaluation
flags.DEFINE_integer('save_step', 5000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10_train.npz', help='FID cache')

device = torch.device('cuda:0')


def to_mnist(labels, mnist_loopers, num_classes):
    assert len(labels.shape) == 1 # shape [B,]
    result = []
    for label in labels:
        temp = torch.Tensor(
            next(mnist_loopers[label.item()]),
        ).unsqueeze(0)
        temp = F.pad(temp, (2, 2, 2, 2), 'constant')
        result.append(temp)
    return torch.cat(result, dim=0)


def infiniteloop(dataloader, num_classes, option):
    assert option in ['onehot', 'mnist']

    if option == 'onehot':
        assert 0==1
        
    elif option == 'mnist':
        def infinite_mnist_loop(
            num: int,
            splitted_root='../../data/mnist/splitted'
        ):
            data = np.load(os.path.join(splitted_root, f'{num}.npz'))['data']
            while True:
                i = np.random.randint(data.shape[0])
                yield data[i]

        mnist_loopers = []

        for i in range(FLAGS.num_classes):
            looper = infinite_mnist_loop(i)
            mnist_loopers.append(looper)

        while True:
            for x, y in iter(dataloader):
                y = to_mnist(y, mnist_loopers, num_classes)
                yield x, y


def evaluate(sampler, model, option='joint_gen', break_num=1, verbose=False):

    model.eval()
    logdir = FLAGS.logdir if verbose else None
    
    if option == 'joint_gen':
        with torch.no_grad():
            images, labels = [], []
            for i in trange(0, FLAGS.sample_size, FLAGS.sample_size, desc=option):
                B = FLAGS.sample_size
                x_T = torch.randn(B, 3, FLAGS.img_size, FLAGS.img_size)
                y_T = torch.randn(B, 1, FLAGS.img_size, FLAGS.img_size)
                x, y = sampler(
                    x_T.to(device),
                    y_T.to(device),
                    option=option,
                    logdir=logdir,
                )
                images.append((x.detach().cpu() + 1.) / 2)
                labels.append((y.detach().cpu() + 1.) / 2)
            images = torch.cat(images, dim=0)
            labels = torch.cat(labels, dim=0)

    elif option == 'cls':
        assert isinstance(break_num, int) and break_num >= 1
        dataset = CIFAR10(
            root='../../data/cifar10/train/',
            train=True,
            download=True,
            transform=T.Compose([
                T.RandomHorizontalFlip(0.5),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=FLAGS.sample_size,
            shuffle=True,
            num_workers=FLAGS.num_workers,
            drop_last=True,
        )
        with torch.no_grad():
            images, labels = [], []
            for i, (img, _) in tqdm(enumerate(dataloader), desc=option, unit='step'):
                if i >= break_num:
                    break
                B = img.shape[0]
                x_T = torch.randn(B, 3, FLAGS.img_size, FLAGS.img_size)
                y_T = torch.randn(B, 1, FLAGS.img_size, FLAGS.img_size)
                x, y = sampler(
                    x_T.to(device),
                    y_T.to(device),
                    x_0=img.to(device),
                    option=option,
                    logdir=logdir,
                )
                images.append((x.detach().cpu() + 1.) / 2)
                labels.append((y.detach().cpu() + 1.) / 2)
            images = torch.cat(images, dim=0)
            labels = torch.cat(labels, dim=0)

    elif option == 'cond_gen':
        assert isinstance(break_num, int) and break_num >= 1
        dataset = MNIST(
            root='../../data/mnist/',
            train=True,
            download=True,
            transform=T.Compose([
                T.ToTensor(),
                T.Normalize(0.5, 0.5),
            ])
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=FLAGS.sample_size,
            shuffle=True,
            num_workers=FLAGS.num_workers,
            drop_last=True,
        )
        with torch.no_grad():
            images, labels = [], []
            for i, (img, _) in tqdm(enumerate(dataloader), desc=option, unit='step'):
                if i >= break_num:
                    break
                B = img.shape[0]
                img = F.pad(img, (2, 2, 2, 2), mode='constant', value=0.0)
                x_T = torch.randn(B, 3, FLAGS.img_size, FLAGS.img_size)
                y_T = torch.randn(B, 1, FLAGS.img_size, FLAGS.img_size)
                x, y = sampler(
                    x_T.to(device),
                    y_T.to(device),
                    y_0=img.to(device),
                    option=option,
                    logdir=logdir,
                )
                images.append((x.detach().cpu() + 1.) / 2)
                labels.append((y.detach().cpu() + 1.) / 2)
            images = torch.cat(images, dim=0)
            labels = torch.cat(labels, dim=0)

    else:
        raise NotImplementedError

    model.train()
    return images, labels


def train():

    # Dataset
    dataset = CIFAR10(
        root='../../data/cifar10/train/',
        train=True,
        download=True,
        transform=T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )
    split_and_save_mnist()
    datalooper = infiniteloop(dataloader, FLAGS.num_classes, 'mnist')

    # Model setup
    if FLAGS.model_name == 'vit':
        net_model = ViT(
            FLAGS.image_size,
            FLAGS.patch_size,
            FLAGS.num_classes,
            FLAGS.T,
            FLAGS.dim,
            FLAGS.depth,
            FLAGS.heads,
            FLAGS.mlp_dim,
            channels=3,
            dim_head=64,
            dropout=FLAGS.dropout
        )
    elif FLAGS.model_name == 'unet':
        net_model = UNet(
            FLAGS.image_size,
            FLAGS.T,
            FLAGS.num_classes,
            FLAGS.ch,
            FLAGS.ch_mult,
            FLAGS.attn,
            FLAGS.num_res_blocks,
            FLAGS.dropout,
            FLAGS.x_dropout,
            FLAGS.y_dropout,
        )
        
    net_model = torch.nn.DataParallel(net_model)
    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(
        net_model.parameters(),
        lr=FLAGS.lr,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lambda step: min(step, FLAGS.warmup) / FLAGS.warmup,
    )
    trainer = GaussianDiffusionTrainer(
        net_model,
        FLAGS.beta_1, 
        FLAGS.beta_T, 
        FLAGS.T,
        FLAGS.logsnr
    ).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model,
        FLAGS.beta_1,
        FLAGS.beta_T,
        FLAGS.T,
        FLAGS.n_corrector,
        FLAGS.snr,
        FLAGS.img_size,
        FLAGS.mean_type,
        FLAGS.var_type
    ).to(device)

    if FLAGS.parallel:
        trainer = torch.nn.DataParallel(trainer)
        ema_sampler = torch.nn.DataParallel(ema_sampler)

    # Log setup
    os.makedirs(os.path.join(FLAGS.logdir, 'sample'), exist_ok=True)
    for option in ['joint_gen', 'cls', 'cond_gen']:
        os.makedirs(os.path.join(FLAGS.logdir, 'sample', option), exist_ok=True)
    writer = SummaryWriter(FLAGS.logdir)

    # Backup all hyperparameters
    with open(os.path.join(FLAGS.logdir, 'flagfile.txt'), 'w') as f:
        f.write(FLAGS.flags_into_string())

    # Show model size
    model_size = sum(p.numel() for p in net_model.parameters())
    print(f'Model params: {model_size / 1024 / 1024:.2f}M')

    # Start training
    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            x_0, y_0 = next(datalooper)
            x_0 = x_0.to(device)
            y_0 = y_0.to(device)
            coin = np.random.uniform(low=0, high=1)
            if coin < 0.8:
                input_dropout_opt = None
            elif coin < 0.9:
                input_dropout_opt = 'x'
            else:
                input_dropout_opt = 'y'
            x_loss, y_loss = trainer(x_0, y_0, input_dropout_opt)
            loss = x_loss + FLAGS.lmbda * y_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(),
                FLAGS.grad_clip
            )
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)

            # log
            writer.add_scalar('x_loss', x_loss, step)
            writer.add_scalar('y_loss', y_loss, step)
            writer.add_scalar('loss', loss, step)
            pbar.set_postfix(loss=f'{loss:.3f}')

            # sample
            if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
                for option in ['joint_gen', 'cls', 'cond_gen']:
                    x_0, y_0 = evaluate(
                        ema_sampler,
                        net_model,
                        option=option,
                        break_num=1, # only for classification
                    )
                    save_image(
                        x_0[:64],
                        os.path.join(FLAGS.logdir, 'sample', option, f'x_{step}.png') 
                    )
                    save_image(
                        y_0[:64],
                        os.path.join(FLAGS.logdir, 'sample', option, f'y_{step}.png') 
                    )

            # save
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'scheduler': sched.state_dict(),
                    'step': step,
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, f'ckpt_{step}.pt'))
                torch.save(ckpt, os.path.join(FLAGS.logdir, f'ckpt_latest.pt'))
    writer.close()


def eval(option='joint_gen'):
    # Model setup
    if FLAGS.model_name == 'vit':
        model = ViT(
            FLAGS.image_size,
            FLAGS.patch_size,
            FLAGS.num_classes,
            FLAGS.T,
            FLAGS.dim,
            FLAGS.depth,
            FLAGS.heads,
            FLAGS.mlp_dim,
            channels=3,
            dim_head=64,
            dropout=FLAGS.dropout
        )
    elif FLAGS.model_name == 'unet':
        model = UNet(
            FLAGS.image_size,
            FLAGS.T,
            FLAGS.num_classes,
            FLAGS.ch,
            FLAGS.ch_mult,
            FLAGS.attn,
            FLAGS.num_res_blocks,
            FLAGS.dropout
        )

    sampler = GaussianDiffusionSampler(
        model,
        beta_1=FLAGS.beta_1,
        beta_T=FLAGS.beta_T,
        T=FLAGS.T,
        n_corrector=FLAGS.n_corrector,
        snr=FLAGS.snr,
        img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type,
        var_type=FLAGS.var_type
    ).to(device)

    model = torch.nn.DataParallel(model)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)
    
    # Load model and evaluate
    ckpt = torch.load(
        os.path.join(FLAGS.logdir, 'ckpt_latest.pt'),
        map_location=device
    )
    model.load_state_dict(ckpt['ema_model'])

    images, labels = evaluate(
        sampler,
        model,
        option=option,
        break_num=30,
    )

    now = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    save_image(
        images[:64].detach(),
        os.path.join(FLAGS.logdir, f'images_{option}_{now}.png'),
        nrow=8,
    )
    save_image(
        labels[:64].detach(),
        os.path.join(FLAGS.logdir, f'labels_{option}_{now}.png'),
        nrow=8,
    )


def main(argv):
    if FLAGS.train:
        train()
    elif FLAGS.eval:
        eval(option=FLAGS.eval_option)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    app.run(main)