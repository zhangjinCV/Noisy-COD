
import sys 
sys.path.append("/mnt/550aa7b7-3fbe-43b7-86bd-198efb9b4305/zj/works_in_phd/ECCV2024/Step2_Code")
from lib.Network import Network
from utils.data_val import get_test_loader
import glob
import torch
import tqdm
from torch.nn import functional as F
from torch import nn
import numpy as np


test_loader_fully = get_test_loader("/mnt/550aa7b7-3fbe-43b7-86bd-198efb9b4305/zj/works_in_phd/ECCV2024/LabelNoiseTrainDataset/CAMO_COD_train_10%/image/",
                                    "/mnt/550aa7b7-3fbe-43b7-86bd-198efb9b4305/zj/works_in_phd/ECCV2024/LabelNoiseTrainDataset/CAMO_COD_train_10%/mask/",
                                    128,
                                    384)
test_loader_weak = get_test_loader("/mnt/550aa7b7-3fbe-43b7-86bd-198efb9b4305/zj/works_in_phd/ECCV2024/LabelNoiseTrainDataset/CAMO_COD_generate_90%/image/",
                                   "/mnt/550aa7b7-3fbe-43b7-86bd-198efb9b4305/zj/works_in_phd/ECCV2024/LabelNoiseTrainDataset/CAMO_COD_generate_90%/mask/",
                                   128,
                                   384)

no_noise_train_path = ["/mnt/550aa7b7-3fbe-43b7-86bd-198efb9b4305/zj/works_in_phd/ECCV2024/step2/fix_q/10%/model_new/Net_epoch_" + str(i) + '.pth' for i in range(1, 200)]
noise_train_path = ["/mnt/550aa7b7-3fbe-43b7-86bd-198efb9b4305/zj/works_in_phd/ECCV2024/step2/noise/10%/model_new/Net_epoch_" + str(i) + '.pth' for i in range(1, 200)]

model = Network()
iou_weakly = []
iou_fully = []


def structure_loss(pred, mask):
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = (inter + 1) / (union - inter + 1)
    return iou.mean()


with torch.no_grad():
    for i_pth in tqdm.tqdm(range(len(no_noise_train_path))):
        weights = torch.load(no_noise_train_path[i_pth])
 
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v
        
        model.load_state_dict(weights_dict)

        model.eval()
        model.cuda()
        iou_f = []
        iou_w = []
        for i, (image, gt, [H, W], name) in enumerate(test_loader_fully, start=1):
            gt = gt.cuda()
            image = image.cuda()
            result = model(image)
            res = result[4]
            res = res.sigmoid()
            iou_f.append(structure_loss(res, gt).item())
        iou_fully.append(np.mean(iou_f))
        
        for i, (image, gt, [H, W], name) in enumerate(test_loader_weak, start=1):
            gt = gt.cuda()
            image = image.cuda()
            result = model(image)
            res = result[4]
            res = res.sigmoid()
            iou_w.append(structure_loss(res, gt).item())
        iou_weakly.append(np.mean(iou_w))


print(iou_fully)
print(iou_weakly)
        
        
