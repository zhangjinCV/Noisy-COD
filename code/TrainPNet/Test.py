import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
from lib.Network import Network
from utils.data_val import get_test_loader
import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')  #
parser.add_argument('--ration', type=int, default=20, help='ration')
parser.add_argument('--pth_path', type=str, default='../weight/PNet/')  # this
parser.add_argument('--test_dataset_path', type=str, default='../data/')
parser.add_argument('--save_path', type=str, default="../results/")
opt = parser.parse_args()
opt.pth_path = opt.pth_path + str(opt.ration) + "%/Net_epoch_best.pth"
all_dataset_mae = []
datasets = ['COD10K_Test', "CAMO_TestingDataset", "CHAMELEON_TestingDataset", "NC4K"]
with torch.no_grad():
    for _data_name in datasets:
        mae = []
        data_path = opt.test_dataset_path + '/{}/'.format(_data_name)
        save_path = opt.save_path + "/" + str(opt.ration) + "%/" + _data_name  # this
        os.makedirs(save_path, exist_ok=True)

        model = Network()
        weights = torch.load(opt.pth_path)

        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v

        model.load_state_dict(weights_dict)
        model.cuda()
        model.eval()

        image_root = '{}/image/'.format(data_path)
        gt_root = '{}/mask/'.format(data_path)
        test_loader = get_test_loader(image_root, gt_root, 128, opt.testsize, False, 4)

        for i, (image, gt, [H, W], name) in tqdm.tqdm(enumerate(test_loader, start=1)):
            gt = gt.cuda()
            image = image.cuda()
            result = model(image)
            res = result[4]
            res = res.sigmoid()
            for num in range(len(image)):
                pre = res[num].squeeze().detach().cpu().numpy()
                gt_single = gt[num].squeeze().detach().cpu().numpy()
                pre = cv2.resize(pre, dsize=(H[num].item(), W[num].item()))
                gt_single = cv2.resize(gt_single, dsize=(H[num].item(), W[num].item()))
                mae.append(np.mean(np.abs(gt_single - pre)))
                cv2.imwrite(save_path + '/' + name[num].replace(".jpg", ".png"), pre * 255.)
        print(_data_name, ':', np.mean(mae))
