import argparse

import lightly
import pytorch_lightning as pl
import torch
import torchvision
from lightly.data import LightlyDataset
from lightly.data.collate import SimCLRCollateFunction, BaseCollateFunction
from torch import nn
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import RichProgressBar
from backbones import Conv6, ResNetCifarGenerator
from collate import SimCLRCollateFunctionWithOriginal
from dataloaders import get_clf_dataloaders
from losses import ContrastiveHinge, ContrastiveHinge2, HingeFewerNegs, HingeNoNegs
from models import LinearProbingClassifier
from models import SimSiam, SimCLR, BYOL, DirectCopy, DirectCopyBP
from transform_external import distortion3_collate_fn


def inference_backbone_output_shape(backbone, input_size=32):
    device = next(backbone.parameters()).device
    val = torch.randn(2, 3, input_size, input_size).to(device)
    _, backbone_output_shape = backbone(val).flatten(start_dim=1).shape
    return backbone_output_shape


def get_backbone(backbone_name, input_size):
    if backbone_name == 'resnet18':
        resnet = torchvision.models.resnet18()
        backbone = nn.Sequential(*list(resnet.children())[:-1])
    elif backbone_name == 'lightly_resnet18':
        resnet = lightly.models.ResNetGenerator('resnet-18')
        backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
    elif backbone_name == 'lightly_resnet18_large_head':
        resnet = lightly.models.ResNetGenerator('resnet-18')
        backbone = nn.Sequential(
            *list(resnet.children())[:-1],
        )
    elif backbone_name == 'resnet20':
        resnet = ResNetCifarGenerator('resnet-20')
        backbone = nn.Sequential(
            *list(resnet.children())[:-1],
        )
    elif backbone_name == 'conv6':
        backbone = Conv6()
    else:
        raise ValueError(f'Backbone {backbone_name} not supported')

    backbone_out_shape = inference_backbone_output_shape(backbone, input_size=input_size)
    return backbone, backbone_out_shape


def get_loss(loss_name, model_name):
    if not loss_name:
        loss = None
    elif loss_name == 'ContrastiveHinge':
        loss = ContrastiveHinge(normalize=True)
    elif loss_name == 'ContrastiveHingeNN':
        loss = ContrastiveHinge(normalize=False)
    elif loss_name == 'ContrastiveHinge2':
        loss = ContrastiveHinge2(normalize=True)
    elif loss_name == 'ContrastiveHingeNN2':
        loss = ContrastiveHinge2(normalize=False)
    elif loss_name == 'HingeFewerNegs':
        if model_name in ['simsiam', 'byol', 'directcopy']:
            loss = HingeFewerNegs(future=1, normalize=True)
        else:
            loss = HingeFewerNegs(normalize=True)
    elif loss_name == 'HingeNNFewerNegs':
        if model_name in ['simsiam', 'byol', 'directcopy']:
            loss = HingeFewerNegs(future=1, normalize=False)
        else:
            loss = HingeFewerNegs(normalize=False)
    elif loss_name == 'HingeNoNegs':
        loss = HingeNoNegs(normalize=True)
    elif loss_name == 'HingeNNNoNegs':
        loss = HingeNoNegs(normalize=False)
    elif hasattr(lightly.loss, loss_name):
        loss = getattr(lightly.loss, loss_name)()
    elif hasattr(torch.nn, loss_name):
        loss = getattr(torch.nn, loss_name)()
    else:
        raise ValueError('Loss not supported')
    return loss


def get_model(args, backbone, loss, backbone_out_shape):
    if args.model == 'simsiam':
        model = SimSiam(backbone, backbone_out_shape, loss)
    elif args.model == 'simclr':
        model = SimCLR(backbone, backbone_out_shape, loss)
    elif args.model == 'byol':
        model = BYOL(backbone, backbone_out_shape, loss)
    elif args.model == 'directcopy':
        model = DirectCopy(backbone, backbone_out_shape, loss, args.dc_m, args.dc_mu, args.dc_epsilon)
    elif args.model == 'directcopybp':
        model = DirectCopyBP(backbone, backbone_out_shape, loss, symmetric_loss=(not args.dc_no_symmetric_loss),
                             mu=args.dc_mu, epsilon=args.dc_epsilon)
    else:
        raise ValueError('Model not supported')
    return model


def get_ssl_dataset(dataset_name):
    if dataset_name == 'stl10':
        ssl_dataloader = torchvision.datasets.STL10("./data", download=True, split='unlabeled')
        input_size = 96
    elif dataset_name == 'cifar10':
        ssl_dataloader = torchvision.datasets.CIFAR10("./data", download=True, train=True)
        input_size = 32
    elif dataset_name == 'cifar100':
        ssl_dataloader = torchvision.datasets.CIFAR100("./data", download=True, train=True)
        input_size = 32
    else:
        raise ValueError(f'Dataset {dataset_name} not supported')
    dataset_ssl = LightlyDataset.from_torch_dataset(ssl_dataloader)
    return dataset_ssl, input_size


def get_collate_fn(collate_fn_name, input_size):
    if collate_fn_name == 'none':
        collate_fn = BaseCollateFunction(transform=torchvision.transforms.ToTensor())
    elif collate_fn_name == 'simclr':
        collate_fn = SimCLRCollateFunction(input_size=input_size, gaussian_blur=0.)
    elif collate_fn_name == 'simclr_w_original':
        collate_fn = SimCLRCollateFunctionWithOriginal(input_size=input_size, gaussian_blur=0.)
    elif collate_fn_name == 'distort3':
        collate_fn = distortion3_collate_fn()
    else:
        raise ValueError('Collate not supported')
    return collate_fn


def main(args):
    print(args)
    if args.deterministic:
        pl.seed_everything(args.seed)

    dataset_ssl, input_size = get_ssl_dataset(args.dataset)
    backbone, backbone_out_shape = get_backbone(args.backbone, input_size)
    loss = get_loss(args.loss, args.model)
    model = get_model(args, backbone, loss, backbone_out_shape)

    collate_fn = get_collate_fn(args.collate, input_size)
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
    dataloader_clf_train, dataloader_clf_test = get_clf_dataloaders(dataset=args.dataset,
                                                                    batch_size=args.clf_batch_size,
                                                                    num_workers=args.num_workers)

    gpus = [args.gpu_id] if torch.cuda.is_available() and args.gpu_id >= 0 else 0

    if args.load:
        model.backbone.load_state_dict(torch.load(args.load))
    else:
        if args.model in ['directcopy', 'directcopybp']:
            log_dir = f'./ssl_logs/dc_mu_{args.dc_mu}_dc_eps_{args.dc_epsilon}__batchsz_{args.batch_size}'
        else:
            log_dir = f'./ssl_logs/{args.model}__batchsz_{args.batch_size}'
        trainer=pl.Trainer(default_root_dir=log_dir, max_epochs=args.ssl_epochs, gpus=gpus, callbacks=[RichProgressBar(leave=True)])
        trainer.fit(model=model, train_dataloaders=dataloader_ssl)
        if args.save:
            #torch.save(model,args.save)
            torch.save({'bb':model.backbone.state_dict(),'pp':model.projection_head.state_dict()}, args.save)

        # loss=0
        # count=0
        # for tt in dataloader_ssl:
        #     #print(tt)
        #     tloss=model.training_step(tt,0)
        #     print(tloss)
        #     loss+=tloss
        #     count+=1
        #
        # print('loss',loss/count)


    model.eval()
    classifier = LinearProbingClassifier(model.backbone, backbone_out_shape)
    trainer = pl.Trainer(max_epochs=args.clf_epochs, gpus=gpus,callbacks=[RichProgressBar(leave=True)])
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
    parser.add_argument("--dataset", type=str, default='cifar10', choices=['stl10', 'cifar10', 'cifar100'])
    parser.add_argument("--model", type=str, default="directcopybp",
                        choices=["simsiam", "simclr", "byol", "directcopy", "directcopybp"])
    parser.add_argument("--backbone", type=str, default="conv6",
                        choices=["resnet18", "lightly_resnet18", "lightly_resnet18_large_head", "resnet20", "conv6"])
    parser.add_argument("--collate", type=str, default="simclr",
                        choices=["simclr", "distort3", "none", "simclr_w_original"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--loss", type=str, default=None)
    parser.add_argument("--dc_m", type=float, default=0.996)
    parser.add_argument("--dc_mu", type=float, default=0.5)
    parser.add_argument("--dc_epsilon", type=float, default=0.3)
    parser.add_argument("--dc_no_symmetric_loss", action='store_true')
    parser.add_argument("--save", type=str, default="backbone_weights.pth")
    parser.add_argument("--load", type=str, default=None)
    args = parser.parse_args()

    main(args)
