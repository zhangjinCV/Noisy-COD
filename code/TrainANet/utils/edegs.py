def mask2edge(path, save_path):
    import cv2
    import glob 
    import torch
    from torch.nn import functional as F
    import os
    import tqdm
    imgs = glob.glob(path + '/*')
    for img in tqdm.tqdm(imgs):
        name = img
        img = cv2.imread(img, 0) / 255.
        img = torch.Tensor(img).unsqueeze(0).unsqueeze(0)
        mask = img
        mask_boundary = F.max_pool2d(1 - mask, kernel_size=3, stride=1, padding=1)
        mask_boundary = mask_boundary - (1 - mask)
        mask_boundary = F.max_pool2d(mask_boundary, kernel_size=3, stride=1, padding=1) * mask
        mask_boundary = mask_boundary.numpy().squeeze() * 255.
        save = os.path.join(save_path, os.path.basename(name))
        cv2.imwrite(save, mask_boundary)


mask2edge("work/CAMO_COD_train/mask", "work/CAMO_COD_train/edge")