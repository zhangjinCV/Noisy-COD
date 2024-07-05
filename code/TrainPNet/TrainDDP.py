import os
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from lib.Network import Network
from utils.data_val import get_train_loader, get_test_loader, PolypObjDataset
from utils.utils import clip_gradient, adjust_lr, get_coef, cal_ual
from tensorboardX import SummaryWriter
import logging
import tqdm
import random
from torch import nn
from torch.cuda import amp
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def seed_torch(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch()
scaler = amp.GradScaler(enabled=True)


class NCLoss(nn.Module):
    def __init__(self):
        super(NCLoss, self).__init__()

    def wbce_loss(self, preds, targets):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(targets, kernel_size=31, stride=1, padding=15) - targets)
        wbce = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
        return wbce.mean()

    def forward(self, preds, targets, q):
        wbce = self.wbce_loss(preds, targets)
        preds = torch.sigmoid(preds)
        preds_flat = preds.contiguous().view(preds.shape[0], -1)
        targets_flat = targets.contiguous().view(targets.shape[0], -1)
        numerator = torch.sum(torch.abs(preds_flat - targets_flat) ** q, dim=1)
        intersection = torch.sum(preds_flat * targets_flat, dim=1)
        denominator = torch.sum(preds_flat, dim=1) + torch.sum(targets_flat, dim=1) - intersection + 1e-6
        loss = numerator / denominator
        if q == 2:
            return loss.mean() + wbce
        else:
            return loss.mean() * 2

def cal_iou(pred, mask):
    inter = ((pred * mask)).sum(dim=(2, 3))
    union = ((pred + mask)).sum(dim=(2, 3))
    wiou = (inter + 1) / (union - inter + 1)
    return wiou.mean()

def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()

def train(rank, world_size, opt):
    seed_torch(2024 + rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    model = Network().to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print(f'Loaded model from {opt.load}')

    optimizer = torch.optim.Adam(model.parameters(), opt.init_lr)


    weak_train_dataset = PolypObjDataset(image_root=opt.weak_train_root + 'image/', gt_root=opt.weak_train_root + 'mask/', trainsize=opt.trainsize, istraining=True)
    weak_train_sampler = torch.utils.data.distributed.DistributedSampler(weak_train_dataset, num_replicas=world_size, rank=rank)
    weak_train_loader = torch.utils.data.DataLoader(dataset=weak_train_dataset, batch_size=opt.batchsize_weakly, num_workers=opt.num_workers, sampler=weak_train_sampler)

    fully_train_dataset = PolypObjDataset(image_root=opt.fully_train_root + 'image/', gt_root=opt.fully_train_root + 'mask/', trainsize=opt.trainsize, istraining=True)
    fully_train_sampler = torch.utils.data.distributed.DistributedSampler(fully_train_dataset, num_replicas=world_size, rank=rank)
    fully_train_loader = torch.utils.data.DataLoader(dataset=fully_train_dataset, batch_size=opt.batchsize_fully, num_workers=opt.num_workers, sampler=fully_train_sampler)

    val_loader = get_test_loader(image_root=opt.val_root + 'image/', gt_root=opt.val_root + 'mask/', batchsize=160, trainsize=opt.trainsize)

    print(len(weak_train_loader), len(fully_train_loader))
    total_step = min(len(weak_train_loader), len(fully_train_loader))
    save_path = opt.save_path
    if rank == 0:
        logging.basicConfig(filename=save_path + 'log.log',
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    step = 0
    if rank == 0:
        writer = SummaryWriter(save_path + 'summary')
    best_mae = 1
    best_epoch = 0

    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(epoch, opt.top_epoch, opt.epoch, opt.init_lr, opt.top_lr, opt.min_lr, optimizer)
        if rank == 0:
            logging.info(f'learning_rate: {cur_lr}')
            writer.add_scalar('learning_rate', cur_lr, global_step=epoch)

        model.train()
        loss_all = 0
        epoch_step = 0
        lr = optimizer.param_groups[0]['lr']
        nc_loss = NCLoss()
        q_value = 1 if epoch > opt.q_epoch else 2

        for i, (fully_train, weak_train) in enumerate(zip(fully_train_loader, weak_train_loader), start=1):
            optimizer.zero_grad()
            images_fully, gts_fully, edges_fully = fully_train
            images_weak, gts_weak, edges_weak = weak_train
            images = torch.cat([images_fully, images_weak], 0).to(device)
            gts = torch.cat([gts_fully, gts_weak], 0).to(device)
            edges = torch.cat([edges_fully, edges_weak], 0).to(device)

            with amp.autocast(enabled=True):
                preds = model(images)

                ual_coef = get_coef(iter_percentage=i / total_step, method='cos')
                ual_loss = cal_ual(seg_logits=preds[4], seg_gts=gts)
                ual_loss *= ual_coef

                loss_init = nc_loss(preds[0], gts, q_value) * 0.0625 + nc_loss(preds[1], gts, q_value) * 0.125 + nc_loss( preds[2], gts, q_value) * 0.25 + nc_loss(preds[3], gts, q_value) * 0.5
                # loss_init = structure_loss(preds[0], gts) * 0.0625 + structure_loss(preds[1], gts) * 0.125 + structure_loss( preds[2], gts) * 0.25 + structure_loss(preds[3], gts) * 0.5

                loss_final = nc_loss(preds[4], gts, q_value)
            # loss_final = structure_loss(preds[4], gts)

                loss_edge = dice_loss(preds[6], edges) * 0.125 + dice_loss(preds[7], edges) * 0.25 + dice_loss(preds[8], edges) * 0.5

            
                loss = loss_init + loss_final + loss_edge + 2 * ual_loss

            scaler.scale(loss).backward()
            # loss.backward()
            # clip_gradient(optimizer, opt.clip)
            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.item()

            if rank == 0 and (i % 20 == 0 or i == total_step or i == 1):
                print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Epoch [{epoch:03d}/{opt.epoch:03d}], Step [{i:04d}/{total_step:04d}], LR {lr:.8f} Total_loss: {loss.item():.4f} Loss1: {loss_init.item():.4f} Loss2: {loss_final.item():.4f} Loss3: {loss_edge.item():.4f} Q: {q_value:.4f} Size: {images.shape}')
                logging.info(f'[Train Info]:Epoch [{epoch}/{opt.epoch}], Step [{i}/{total_step}], Total_loss: {loss.item():.4f} Loss1: {loss_init.item():.4f} Loss2: {loss_final.item():.4f} Loss3: {loss_edge.item():.4f}')

        loss_all /= epoch_step
        if rank == 0:
            logging.info(f'[Train Info]: Epoch [{epoch:03d}/{opt.epoch:03d}], Loss_AVG: {loss_all:.4f}')
            writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
            if epoch > 96:
                torch.save(model.state_dict(), save_path + f'Net_epoch_{epoch}.pth')

        if rank == 0:
            val(val_loader, model, epoch, save_path, writer, opt)

    if rank == 0:
        writer.close()
    dist.destroy_process_group()

def val(test_loader, model, epoch, save_path, writer, opt):
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        mae_sum = []
        iou_sum = []
        for i, (image, gt, [H, W], name) in tqdm.tqdm(enumerate(test_loader, start=1)):
            gt = gt.cuda()
            image = image.cuda()
            results = model(image)
            res = results[4]
            res = res.sigmoid()
            iou_sum.append(cal_iou(res, gt).item())
            for i in range(len(res)):
                pre = F.interpolate(res[i].unsqueeze(0), size=(H[i].item(), W[i].item()), mode='bilinear')
                gt_single = F.interpolate(gt[i].unsqueeze(0), size=(H[i].item(), W[i].item()), mode='bilinear')
                mae_sum.append(torch.mean(torch.abs(gt_single - pre)).item())
        iou = np.mean(iou_sum)
        mae = np.mean(mae_sum)
        mae = "%.5f" % mae
        mae = float(mae)
        print(f'Epoch: {epoch}, MAE: {mae}, IoU: {iou}, bestMAE: {opt.best_mae}, bestEpoch: {opt.best_epoch}.')
        if mae < opt.best_mae:
            opt.best_mae = mae
            opt.best_epoch = epoch
            torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
            print(f'Save state_dict successfully! Best epoch: {epoch}.')
        logging.info(f'[Val Info]:Epoch: {epoch}, MAE: {mae}, IoU: {iou}, bestMAE: {opt.best_mae}, bestEpoch: {opt.best_epoch}.')

    torch.cuda.empty_cache()

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--top_epoch', type=int, default=5, help='epoch number')
    parser.add_argument('--top_lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--init_lr', type=float, default=1e-7, help='learning rate')
    parser.add_argument('--min_lr', type=float, default=14e-7, help='learning rate')
    parser.add_argument('--q_epoch', type=int, default=60, help='learning rate')
    parser.add_argument('--batchsize_fully', type=int, default=6, help='training batch size')
    parser.add_argument('--batchsize_weakly', type=int, default=24, help='training batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='train from checkpoints')
    parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--gpu_id', type=str, default='0,1,2,3', help='train use gpu')
    parser.add_argument('--ration', type=str, default='1', help='train ration')
    parser.add_argument('--fully_train_root', type=str, default='../data/LabelNoiseTrainDataset/CAMO_COD_train_', help='the training rgb images root')
    parser.add_argument('--weak_train_root', type=str, default='../pseduo label/ANet/CAMO_COD_generate_', help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='../data/CAMO_TestingDataset/', help='the test rgb images root')
    parser.add_argument('--best_mae', type=float, default=1.0, help='best mae')
    parser.add_argument('--best_iou', type=float, default=0.0, help='best mae')
    parser.add_argument('--best_epoch', type=int, default=1, help='best epoch')
    parser.add_argument('--save_path', type=str, default='../weight/PNet/', help='the path to save model and log')
    opt = parser.parse_args()

    opt.fully_train_root = opt.fully_train_root + opt.ration + '%/'
    opt.weak_train_root = opt.weak_train_root + str(100 - int(opt.ration)) + '%/'
    opt.save_path = opt.save_path + opt.ration + "%/"

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    world_size = len(opt.gpu_id.split(','))
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    mp.spawn(train, args=(world_size, opt), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
