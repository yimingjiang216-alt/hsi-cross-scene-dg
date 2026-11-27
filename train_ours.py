# -*- coding: utf-8 -*-
"""
train_ours.py — Training & Cross-Scene Evaluation Script for Proposed Method:
"Physics-Guided Domain Randomization and Spectral-Shape Invariant Representation
for Cross-Scene Hyperspectral Image Classification"

Supports one-click execution of all 3 Cross-Scene Tasks (Houston, Pavia, HyRANK)
and all 4 Ablation Study configurations (Exp 1 to Exp 4).
"""
import argparse
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as data

from utils_HSI import sample_gt, metrics, seed_worker
from datasets import get_dataset, HyperX
from con_losses import SupConLoss
from network.physics_shape_net import (
    PhysicsGuidedGenerator,
    SpectralShapeDisentangledDiscriminator,
    OrthogonalDisentangleLoss,
)

parser = argparse.ArgumentParser(description='PyTorch Physics-Shape Cross-Scene HSI DG (Ours)')
parser.add_argument('--save_path', type=str, default='./results_ours/')
parser.add_argument('--data_path', type=str, default='./datasets/Houston/')
parser.add_argument('--source_name', type=str, default='Houston13',
                    help='Source domain dataset name: Houston13 | paviaU | Dioni')
parser.add_argument('--target_name', type=str, default='Houston18',
                    help='Target domain dataset name: Houston18 | paviaC | Loukia')
parser.add_argument('--gpu', type=int, default=0,
                    help='Specify CUDA device (automatically falls back to CPU if unavailable)')

# Ablation Study Switches (Table 2 in Thesis)
parser.add_argument('--use_phys_aug', type=int, default=1,
                    help='Innovation 1: Wavelength-aware physical domain randomization (1=on, 0=off)')
parser.add_argument('--use_spec_diff', type=int, default=1,
                    help='Innovation 2: Spectral derivative shape stream & 1D gating (1=on, 0=off)')
parser.add_argument('--use_orth_loss', type=int, default=1,
                    help='Innovation 3: Orthogonal style-content disentanglement loss (1=on, 0=off)')
parser.add_argument('--lambda_orth', type=float, default=0.5,
                    help='Weight for orthogonal disentanglement loss L_orth')

group_train = parser.add_argument_group('Training')
group_train.add_argument('--patch_size', type=int, default=13)
group_train.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--momentum', type=float, default=0.9)
group_train.add_argument('--batch_size', type=int, default=256)
group_train.add_argument('--pro_dim', type=int, default=128)
group_train.add_argument('--test_stride', type=int, default=1)
parser.add_argument('--seed', type=int, default=233)
parser.add_argument('--l2_decay', type=float, default=1e-4)
parser.add_argument('--training_sample_ratio', type=float, default=0.8)
parser.add_argument('--re_ratio', type=int, default=5)
parser.add_argument('--max_epoch', type=int, default=400)
parser.add_argument('--log_interval', type=int, default=40)
parser.add_argument('--d_se', type=int, default=64)
parser.add_argument('--lambda_1', type=float, default=1.0)
parser.add_argument('--lambda_2', type=float, default=1.0)
parser.add_argument('--lr_scheduler', type=str, default='none')

group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=True)
group_da.add_argument('--radiation_augmentation', action='store_true', default=True)
group_da.add_argument('--mixture_augmentation', action='store_true', default=False)

args = parser.parse_args()


def get_compute_device(gpu_id):
    if gpu_id >= 0 and torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')


def evaluate(net, val_loader, device, tgt=False):
    ps = []
    ys = []
    for i, (x1, y1) in enumerate(val_loader):
        y1 = y1 - 1
        with torch.no_grad():
            x1 = x1.to(device)
            p1 = net(x1, mode='test')
            p1 = p1.argmax(dim=1)
            ps.append(p1.detach().cpu().numpy())
            ys.append(y1.numpy())
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    acc = np.mean(ys == ps) * 100
    if tgt:
        results = metrics(ps, ys, n_classes=int(ys.max() + 1))
        print(results['Confusion_matrix'], '\n', 'TPR:', np.round(results['TPR'] * 100, 2),
              '\n', 'OA:', round(results['Accuracy'], 2), 'Kappa:', round(results['Kappa'], 4))
    return acc


def evaluate_tgt(cls_net, device, loader, modelpath):
    saved_weight = torch.load(modelpath, map_location=device)
    cls_net.load_state_dict(saved_weight['Discriminator'])
    cls_net.eval()
    teacc = evaluate(cls_net, loader, device, tgt=True)
    return teacc


def experiment():
    device = get_compute_device(args.gpu)
    print(f"[Info] Running on device: {device} | Source: {args.source_name} -> Target: {args.target_name}")
    print(f"[Ablation] use_phys_aug={bool(args.use_phys_aug)} | use_spec_diff={bool(args.use_spec_diff)} | use_orth_loss={bool(args.use_orth_loss)}")

    hyperparams = vars(args)
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    root = os.path.join(args.save_path, args.source_name + 'to' + args.target_name)
    log_dir = os.path.join(
        root,
        f"lr{args.lr}_phys{args.use_phys_aug}_diff{args.use_spec_diff}_orth{args.use_orth_loss}_{time_str}"
    )
    os.makedirs(log_dir, exist_ok=True)
    df = pd.DataFrame([args])
    df.to_csv(os.path.join(log_dir, 'params.txt'))

    seed_worker(args.seed)
    data_folder = args.data_path if args.data_path.endswith('/') or args.data_path.endswith('\\') else args.data_path + '/'
    img_src, gt_src, LABEL_VALUES_src, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.source_name, data_folder)
    img_tar, gt_tar, LABEL_VALUES_tar, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.target_name, data_folder)

    sample_num_src = len(np.nonzero(gt_src)[0])
    sample_num_tar = len(np.nonzero(gt_tar)[0])
    tmp = args.training_sample_ratio * args.re_ratio * sample_num_src / sample_num_tar
    num_classes = int(gt_src.max())
    N_BANDS = img_src.shape[-1]
    hyperparams.update({
        'n_classes': num_classes,
        'n_bands': N_BANDS,
        'ignored_labels': IGNORED_LABELS,
        'device': device,
        'center_pixel': None,
        'supervision': 'full',
    })

    r = int(hyperparams['patch_size'] / 2) + 1
    img_src = np.pad(img_src, ((r, r), (r, r), (0, 0)), 'symmetric')
    img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gt_src = np.pad(gt_src, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    gt_tar = np.pad(gt_tar, ((r, r), (r, r)), 'constant', constant_values=(0, 0))

    train_gt_src, val_gt_src, _, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')
    test_gt_tar, _, _, _ = sample_gt(gt_tar, 1, mode='random')
    img_src_con, train_gt_src_con = img_src, train_gt_src
    val_gt_src_con = val_gt_src
    if tmp < 1:
        for i in range(args.re_ratio - 1):
            img_src_con = np.concatenate((img_src_con, img_src))
            train_gt_src_con = np.concatenate((train_gt_src_con, train_gt_src))
            val_gt_src_con = np.concatenate((val_gt_src_con, val_gt_src))

    hyperparams_train = hyperparams.copy()
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=hyperparams['batch_size'],
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
    )
    val_dataset = HyperX(img_src_con, val_gt_src_con, **hyperparams)
    val_loader = data.DataLoader(
        val_dataset,
        pin_memory=torch.cuda.is_available(),
        batch_size=hyperparams['batch_size'],
    )
    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
    test_loader = data.DataLoader(
        test_dataset,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=g,
        batch_size=hyperparams['batch_size'],
    )
    imsize = [hyperparams['patch_size'], hyperparams['patch_size']]

    D_net = SpectralShapeDisentangledDiscriminator(
        inchannel=N_BANDS,
        outchannel=args.pro_dim,
        num_classes=num_classes,
        patch_size=hyperparams['patch_size'],
        use_spec_diff=bool(args.use_spec_diff),
    ).to(device)
    D_opt = optim.Adam(D_net.parameters(), lr=args.lr, weight_decay=args.l2_decay)

    G_net = PhysicsGuidedGenerator(
        n=args.d_se,
        imdim=N_BANDS,
        imsize=imsize,
        zdim=10,
        device=device,
        use_phys_aug=bool(args.use_phys_aug),
    ).to(device)
    G_opt = optim.Adam(G_net.parameters(), lr=args.lr)

    cls_criterion = nn.CrossEntropyLoss()
    con_criterion = SupConLoss(device=device)
    orth_criterion = OrthogonalDisentangleLoss()

    best_acc = 0
    taracc, taracc_list = 0, []
    for epoch in range(1, args.max_epoch + 1):
        t1 = time.time()
        loss_list = []
        D_net.train()
        G_net.train()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            y = y - 1
            with torch.no_grad():
                x_ED = G_net(x)
            rand = torch.nn.init.uniform_(torch.empty(len(x), 1, 1, 1, device=device))
            x_ID = rand * x + (1 - rand) * x_ED

            x_tgt = G_net(x)
            p_SD, z_SD, z_env_SD = D_net(x, mode='train', return_env=True)
            p_ED, z_ED, z_env_ED = D_net(x_ED, mode='train', return_env=True)
            p_ID, z_ID = D_net(x_ID, mode='train', return_env=False)

            zsrc = torch.cat([z_SD.unsqueeze(1), z_ED.unsqueeze(1), z_ID.unsqueeze(1)], dim=1)
            src_cls_loss = cls_criterion(p_SD, y.long()) + cls_criterion(p_ED, y.long()) + cls_criterion(p_ID, y.long())
            p_tgt, z_tgt = D_net(x_tgt, mode='train', return_env=False)
            tgt_cls_loss = cls_criterion(p_tgt, y.long())

            zall = torch.cat([z_tgt.unsqueeze(1), zsrc], dim=1)
            con_loss = con_criterion(zall, y, adv=False)

            if args.use_orth_loss:
                orth_loss = 0.5 * (orth_criterion(z_SD, z_env_SD) + orth_criterion(z_ED, z_env_ED))
            else:
                orth_loss = torch.tensor(0.0, device=device)

            loss = src_cls_loss + args.lambda_1 * con_loss + tgt_cls_loss + args.lambda_orth * orth_loss
            D_opt.zero_grad()
            loss.backward(retain_graph=True)

            zsrc_con = torch.cat([z_tgt.unsqueeze(1), z_ED.unsqueeze(1).detach(), z_ID.unsqueeze(1).detach()], dim=1)
            con_loss_adv = torch.tensor(0.0, device=device)
            idx_1 = np.random.randint(0, zsrc.size(1))

            for idx_cls, cls_id in enumerate(y.unique()):
                mask = (y == cls_id)
                z_SD_i, zsrc_i = z_SD[mask], zsrc_con[mask]
                y_i = torch.cat([torch.zeros(z_SD_i.shape[0], device=device), torch.ones(z_SD_i.shape[0], device=device)])
                zall_adv = torch.cat([z_SD_i.unsqueeze(1).detach(), zsrc_i[:, idx_1:idx_1 + 1]], dim=0)
                if y_i.size(0) > 2:
                    con_loss_adv = con_loss_adv + con_criterion(zall_adv, y_i)
            con_loss_adv = con_loss_adv / max(1, y.unique().shape[0])

            loss_G = tgt_cls_loss + args.lambda_2 * con_loss_adv
            G_opt.zero_grad()
            loss_G.backward()
            D_opt.step()
            G_opt.step()

            loss_list.append([src_cls_loss.item(), tgt_cls_loss.item(), con_loss.item(), orth_loss.item()])
        src_cls_loss, tgt_cls_loss, con_loss, orth_loss_val = np.mean(loss_list, 0)

        D_net.eval()
        teacc = evaluate(D_net, val_loader, device)
        if best_acc < teacc:
            best_acc = teacc
            torch.save({'Discriminator': D_net.state_dict()}, os.path.join(log_dir, 'best.pkl'))
        t2 = time.time()

        print(
            f"epoch {epoch:3d} | time {t2 - t1:.2f}s | src_cls {src_cls_loss:.4f} | tgt_cls {tgt_cls_loss:.4f} "
            f"| con {con_loss:.4f} | orth {orth_loss_val:.4f} /// val_acc {teacc:2.2f}%"
        )

        if epoch % args.log_interval == 0:
            pklpath = os.path.join(log_dir, 'best.pkl')
            taracc = evaluate_tgt(D_net, device, test_loader, pklpath)
            taracc_list.append(round(taracc, 2))
            print(f"[Target Eval] Target samples={len(test_loader.dataset)} | Current Target OA={taracc:.2f}% | Max Target OA={max(taracc_list):.2f}%")


if __name__ == '__main__':
    experiment()
