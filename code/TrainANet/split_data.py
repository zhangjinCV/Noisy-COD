def split_data_rate(ration=5):
    import torch 
    import os 
    import numpy as np 
    import random 
    import shutil
    
    def seed_torch(seed=2024):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    
    seed_torch()
    choose_samples = int(ration * 400 / 10)
    print("split nums:", choose_samples)
    images = ["../data/CAMO_COD_train/image/" + f for f in os.listdir("../data/CAMO_COD_train/image") if f.endswith('.jpg')]
    samples = np.random.choice(4040, choose_samples, replace=False)
    images = np.array(sorted(images))[samples]
    gts = [i.replace("image", "mask").replace(".jpg", ".png") for i in images]
    boxs = [i.replace("image", "box").replace(".jpg", ".png") for i in images]
    edges = [i.replace("image", "edge").replace(".jpg", ".png") for i in images]
    
    for image in images:
        if not os.path.exists(f"../data/LabelNoiseTrainDataset/CAMO_COD_train_{ration}%/image"):
            os.makedirs(f"../data/LabelNoiseTrainDataset/CAMO_COD_train_{ration}%/image")
        shutil.copy(image, f"../data/LabelNoiseTrainDataset/CAMO_COD_train_{ration}%/image/")
    
    for gt in gts:
        if not os.path.exists(f"../data/LabelNoiseTrainDataset/CAMO_COD_train_{ration}%/mask"):
            os.makedirs(f"../data/LabelNoiseTrainDataset/CAMO_COD_train_{ration}%/mask")
        shutil.copy(gt, f"../data/LabelNoiseTrainDataset/CAMO_COD_train_{ration}%/mask/")
    
    for box in boxs:
        if not os.path.exists(f"../data/LabelNoiseTrainDataset/CAMO_COD_train_{ration}%/box"):
            os.makedirs(f"../data/LabelNoiseTrainDataset/CAMO_COD_train_{ration}%/box")
        shutil.copy(box, f"../data/LabelNoiseTrainDataset/CAMO_COD_train_{ration}%/box/")
    
    for edge in edges:
        if not os.path.exists(f"../data/LabelNoiseTrainDataset/CAMO_COD_train_{ration}%/edge"):
            os.makedirs(f"../data/LabelNoiseTrainDataset/CAMO_COD_train_{ration}%/edge")
        shutil.copy(edge, f"../data/LabelNoiseTrainDataset/CAMO_COD_train_{ration}%/edge/")
    
    alls = ["../data/CAMO_COD_train/image/" + f for f in os.listdir("../data/CAMO_COD_train/image") if f.endswith('.jpg')]
    next_ration = 100 - ration
    images_next_ration = [i for i in alls if i not in images]
    gts_next_ration = [i.replace("image", "mask").replace(".jpg", ".png") for i in images_next_ration]
    boxs_next_ration = [i.replace("image", "box").replace(".jpg", ".png") for i in images_next_ration]
    edges_next_ration = [i.replace("image", "edge").replace(".jpg", ".png") for i in images_next_ration]
    
    for image in images_next_ration:
        if not os.path.exists(f"../data/LabelNoiseTrainDataset/CAMO_COD_generate_{next_ration}%/image"):
            os.makedirs(f"../data/LabelNoiseTrainDataset/CAMO_COD_generate_{next_ration}%/image")
        shutil.copy(image, f"../data/LabelNoiseTrainDataset/CAMO_COD_generate_{next_ration}%/image/")
    
    for gt in gts_next_ration:
        if not os.path.exists(f"../data/LabelNoiseTrainDataset/CAMO_COD_generate_{next_ration}%/mask"):
            os.makedirs(f"../data/LabelNoiseTrainDataset/CAMO_COD_generate_{next_ration}%/mask")
        shutil.copy(gt, f"../data/LabelNoiseTrainDataset/CAMO_COD_generate_{next_ration}%/mask/")
    
    for box in boxs_next_ration:
        if not os.path.exists(f"../data/LabelNoiseTrainDataset/CAMO_COD_generate_{next_ration}%/box"):
            os.makedirs(f"../data/LabelNoiseTrainDataset/CAMO_COD_generate_{next_ration}%/box")
        shutil.copy(box, f"../data/LabelNoiseTrainDataset/CAMO_COD_generate_{next_ration}%/box/")
    
    for edge in edges_next_ration:
        if not os.path.exists(f"../data/LabelNoiseTrainDataset/CAMO_COD_generate_{next_ration}%/edge"):
            os.makedirs(f"../data/LabelNoiseTrainDataset/CAMO_COD_generate_{next_ration}%/edge")
        shutil.copy(edge, f"../data/LabelNoiseTrainDataset/CAMO_COD_generate_{next_ration}%/edge/")
        
    print(len(os.listdir(f"../data/LabelNoiseTrainDataset/CAMO_COD_train_{ration}%/image")), len(os.listdir(f"../data/LabelNoiseTrainDataset/CAMO_COD_generate_{next_ration}%/image/")))


if __name__ == "__main__":
    split_data_rate(1)
    split_data_rate(5)
    split_data_rate(10)
    split_data_rate(20)