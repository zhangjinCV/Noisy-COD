import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
from lib.Network import Network
from utils.data_val import get_test_loader
import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')  #
parser.add_argument('--ration', type=str, default='1', help='testing size')
parser.add_argument('--pth_path', type=str, default="../weight/ANet/")
parser.add_argument('--test_dataset_path', type=str, default="../data/LabelNoiseTrainDataset/")
parser.add_argument('--save_path', type=str, default="../pseudo_label/ANet/")
opt = parser.parse_args()

opt.pth_path = opt.pth_path + opt.ration + '%/Net_epoch_best.pth'
datasets = ['CAMO_COD_generate_' + str(100-int(opt.ration)) + '%']
with torch.no_grad():
    for _data_name in datasets:
        mae = []
        data_path = opt.test_dataset_path+'/{}/'.format(_data_name)
        save_path = opt.save_path + '/' + _data_name
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path + '/mask', exist_ok=True)
        os.makedirs(save_path + '/edge', exist_ok=True)

        model = Network() 
        weights = torch.load(opt.pth_path)
        model.load_state_dict(weights)

        model.cuda()
        model.eval()

        image_root = '{}/image/'.format(data_path)
        gt_root = '{}/mask/'.format(data_path)
        test_loader = get_test_loader(image_root, gt_root, 12, opt.testsize)

        for  i, (image, bbox_image, gt, [H, W], name) in tqdm.tqdm(enumerate(test_loader, start=1)):
            gt = gt.cuda()
            bbox_image = bbox_image.cuda()
            image = image.cuda()
            result = model(image, bbox_image)
            res = result[4]
            edge = result[8]
            res = res.sigmoid()
            mae.append(torch.mean(torch.abs(gt - res)).data.cpu().numpy())
            edge = edge.squeeze().detach().cpu().numpy()
            res = res.squeeze().detach().cpu().numpy()
            for j in range(len(res)):
                pre = cv2.resize(res[j], dsize=(H[j].item(), W[j].item()))
                ed = cv2.resize(edge[j], dsize=(H[j].item(), W[j].item()))
                cv2.imwrite(save_path + '/mask' +'/'+name[j].replace(".jpg", ".png"), pre*255.)
                cv2.imwrite(save_path + '/edge' +'/'+name[j].replace(".jpg", ".png"), ed*255.)
        print(np.mean(mae))

os.system("cp -r ../data/LabelNoiseTrainDataset/CAMO_COD_generate_" + str(100 - int(opt.ration)) + '%' + "/image ../pseudo_label/ANet/CAMO_COD_generate_" + str(100 - int(opt.ration)) + '%/')
