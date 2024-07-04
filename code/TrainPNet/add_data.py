import os
import shutil

def copy_images(original_folder, target_folder, total_copies=520):
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 获取原始文件夹中的所有文件
    files = [f for f in os.listdir(original_folder) if os.path.isfile(os.path.join(original_folder, f))]
    
    # 计算每个文件需要复制的次数
    copies_per_file = total_copies // len(files)

    # 对每个文件进行复制
    for file in files:
        file_path = os.path.join(original_folder, file)
        for i in range(copies_per_file):
            # 创建新文件名
            new_file_name = f"{os.path.splitext(file)[0]}_copy{i}{os.path.splitext(file)[1]}"
            new_file_path = os.path.join(target_folder, new_file_name)
            # 复制文件
            shutil.copy(file_path, new_file_path)

# 调用函数
original_folder = 'work/LabelNoiseTrainDataset/CAMO_COD_train_5%/image'  # 替换为原始文件夹的路径
target_folder = 'work/LabelNoiseTrainDataset/CAMO_COD_train_5%/image'      # 替换为目标文件夹的路径
copy_images(original_folder, target_folder)

original_folder = 'work/LabelNoiseTrainDataset/CAMO_COD_train_5%/mask'  # 替换为原始文件夹的路径
target_folder = 'work/LabelNoiseTrainDataset/CAMO_COD_train_5%/mask'      # 替换为目标文件夹的路径
copy_images(original_folder, target_folder)

original_folder = 'work/LabelNoiseTrainDataset/CAMO_COD_train_5%/edge'  # 替换为原始文件夹的路径
target_folder = 'work/LabelNoiseTrainDataset/CAMO_COD_train_5%/edge'      # 替换为目标文件夹的路径
copy_images(original_folder, target_folder)