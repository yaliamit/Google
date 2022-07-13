import argparse

import lightly
import pytorch_lightning as pl
import torch
import torchvision
from lightly.data import LightlyDataset
from lightly.data import SimCLRCollateFunction, BaseCollateFunction
from torch import nn

from backbones import Conv6, ResNetCifarGenerator
from dataloaders import get_clf_dataloaders
from losses import SimpleHingeLoss, ContrastiveHinge, ContrastiveHinge2, HingeFewerNegs, HingeNoNegs
from models import LinearProbingClassifier
from models import SimSiam, SimCLR, BYOL, DirectCopy, DirectCopyBP
from transform_external import distortion3_collate_fn


def main(args):
    print(args)


    gpus = [args.gpu_id] if torch.cuda.is_available() and args.gpu_id >= 0 else 0
    s = "cuda:" + str(gpus[0])
    device = torch.device(s)
    print(device)
    if args.deterministic:
        pl.seed_everything(args.seed)
    if args.backbone == 'resnet18':
        resnet = torchvision.models.resnet18()
        backbone = nn.Sequential(*list(resnet.children())[:-1])
    elif args.backbone == 'lightly_resnet18':
        resnet = lightly.models.ResNetGenerator('resnet-18')
        backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
    elif args.backbone == 'lightly_resnet18_large_head':
        resnet = lightly.models.ResNetGenerator('resnet-18')
        backbone = nn.Sequential(
            *list(resnet.children())[:-1],
        )
    elif args.backbone == 'resnet20':
        resnet = ResNetCifarGenerator('resnet-20')
        backbone = nn.Sequential(
            *list(resnet.children())[:-1],
        )
    elif args.backbone == 'conv6':
        backbone = Conv6()
    else:
        raise ValueError('Backbone not supported')

    if not args.loss:
        loss = None
    elif args.loss == 'ContrastiveHinge':
        loss = ContrastiveHinge(normalize=True)
    elif args.loss == 'ContrastiveHingeNN':
        loss = ContrastiveHinge(normalize=False)
    elif args.loss == 'ContrastiveHinge2':
        loss = ContrastiveHinge2(normalize=True)
    elif args.loss == 'ContrastiveHingeNN2':
        loss = ContrastiveHinge2(normalize=False)
    elif args.loss == 'HingeFewerNegs':
        if args.model in ['simsiam', 'byol', 'directcopy']:
            loss = HingeFewerNegs(future=1, normalize=True)
        else:
            loss = HingeFewerNegs(normalize=True)
    elif args.loss == 'HingeNNFewerNegs':
        if args.model in ['simsiam', 'byol', 'directcopy']:
            loss = HingeFewerNegs(future=1, normalize=False)
        else:
            loss = HingeFewerNegs(normalize=False)
    elif args.loss == 'SimpleHinge':
        loss = SimpleHingeLoss(margin=args.margin, normalize=True)
    elif args.loss == 'SimpleHingeNN':
        loss = SimpleHingeLoss(margin=args.margin, normalize=False)
    elif args.loss == 'HingeNoNegs':
        loss = HingeNoNegs(normalize=True)
    elif args.loss == 'HingeNNNoNegs':
        loss = HingeNoNegs(normalize=False)
    elif hasattr(lightly.loss, args.loss):
        loss = getattr(lightly.loss, args.loss)()
    elif hasattr(torch.nn, args.loss):
        loss = getattr(torch.nn, args.loss)()
    else:
        raise ValueError('Loss not supported')

    if args.model == 'simsiam':
        model = SimSiam(backbone, args.ssl_epochs, loss)
    elif args.model == 'simclr':
        model = SimCLR(backbone, args.ssl_epochs, loss)
    elif args.model == 'byol':
        model = BYOL(backbone, args.ssl_epochs, loss)
    elif args.model == 'directcopy':
        model = DirectCopy(backbone, args.ssl_epochs, loss, args.dc_cm_grad, args.dc_m, args.dc_mu, args.dc_epsilon)
    elif args.model == 'directcopybp':
        model = DirectCopyBP(backbone, args.ssl_epochs, device, loss, args.dc_cm_grad, args.dc_m, args.dc_mu,
                             args.dc_epsilon, args.perturb, args.symmetric, args.double_aug)
    else:
        raise ValueError('Model not supported')

    if args.collate == 'none':
        collate_fn = BaseCollateFunction(transform=torchvision.transforms.ToTensor())
    elif args.collate == 'simclr':
        collate_fn = SimCLRCollateFunction(input_size=32, gaussian_blur=0.)
    elif args.collate == 'distort3':
        collate_fn = distortion3_collate_fn()
    else:
        raise ValueError('Collate not supported')

    cifar10_train = torchvision.datasets.CIFAR10("./data", download=True, train=True)
    dataset_ssl = LightlyDataset.from_torch_dataset(cifar10_train)

    dataloader_ssl = torch.utils.data.DataLoader(
        dataset_ssl,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    # data for linear probing
    dataloader_clf_train, dataloader_clf_test = get_clf_dataloaders(batch_size=args.clf_batch_size,
                                                                    num_workers=args.num_workers)

    gpus = [args.gpu_id] if torch.cuda.is_available() and args.gpu_id >= 0 else 0

    if args.load:
        model.backbone.load_state_dict(torch.load(args.load))
    else:
        if args.model in ['directcopy', 'directcopybp']:
            log_dir = f'./ssl_logs/dc_mu_{args.dc_mu}_dc_eps_{args.dc_epsilon}_dc_cm_grad_{args.dc_cm_grad}__batchsz_{args.batch_size}'
        else:
            log_dir = f'./ssl_logs/{args.model}__batchsz_{args.batch_size}'
        trainer = pl.Trainer(default_root_dir=log_dir, max_epochs=args.ssl_epochs, gpus=gpus,log_every_n_steps=500)
        trainer.fit(model=model, train_dataloaders=dataloader_ssl)
        if args.save:
            torch.save(model.backbone.state_dict(), args.save)

    model.eval()
    classifier = LinearProbingClassifier(model.backbone)
    trainer = pl.Trainer(max_epochs=args.clf_epochs, gpus=gpus,log_every_n_steps=500)
    trainer.fit(
        classifier,
        dataloader_clf_train,
        dataloader_clf_test
    )
    print(f'Best validation accuracy: {classifier.best_acc:.5f} at epoch {classifier.best_epoch}, '
          f'last validation accuracy: {classifier.last_acc:.5f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--clf_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--deterministic", action='store_true')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ssl_epochs", type=int, default=400)
    parser.add_argument("--clf_epochs", type=int, default=200)
    parser.add_argument("--model", type=str, default="directcopybp",
                        choices=["simsiam", "simclr", "byol", "directcopy", "directcopybp"])
    parser.add_argument("--backbone", type=str, default="conv6",
                        choices=["resnet18", "lightly_resnet18", "lightly_resnet18_large_head", "resnet20", "conv6"])
    parser.add_argument("--collate", type=str, default="simclr", choices=["simclr", "distort3", "none"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--loss", type=str, default=None)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--perturb", type=float, default=None)
    parser.add_argument("--dc_m", type=float, default=0.996)
    parser.add_argument("--dc_mu", type=float, default=0.5)
    parser.add_argument("--dc_epsilon", type=float, default=0.3)
    parser.add_argument("--dc_cm_grad", action='store_true')
    parser.add_argument("--symmetric", action='store_true')
    parser.add_argument("--double_aug", action='store_true')
    parser.add_argument("--save", type=str, default="backbone_weights.pth")
    parser.add_argument("--load", type=str, default=None)
    args = parser.parse_args()

    main(args)
