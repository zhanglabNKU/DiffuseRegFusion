import os
import torch
import random
import numpy as np


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


# 判断给定的文件名是否是支持的图像文件类型
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# 递归遍历指定目录及其子目录，收集所有支持的图像文件路径
def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


# 对图像列表进行随机数据增强。只有在 split 为'train'时才会应用增强。（水平、垂直反转，90°旋转）
def augment(img_list, hflip=True, vflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    aug = (split == 'train' and random.random() < 1.0)
    hflip = hflip and aug and random.random() < 0.5
    vflip = vflip and aug and random.random() < 0.5
    rot90 = rot and aug and random.random() < 0.5

    def _augment(img):
        # print(type(img))
        if hflip:
            # print("hflip")
            img = img[::-1, :]
        if vflip:
            # print("vflip")
            img = img[:, ::-1]
        if rot90:
            # print("rot90")
            img = img.transpose(1, 0)
        return img

    return [_augment(img) for img in img_list]


# 将输入图像转换为NumPy数组，并确保其数据类型为float32
def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img


# 将NumPy数组转换为PyTorch张量，并调整其值范围。
def transform2tensor(img, min_max=(0, 1), split='val'):
    img = torch.from_numpy(img).float().unsqueeze(0)

    # to range min_max
    img = img * (min_max[1] - min_max[0]) + min_max[0]
    return img


# 对图像列表进行数据增强和格式转换的综合处理。
def transform_augment(img_list, split='val', min_max=(0, 1)):
    ret_img = []
    img_list = augment(img_list, split=split)
    for img in img_list[:2]:
        img = transform2numpy(img)
        img = transform2tensor(img, min_max, split=split)
        ret_img.append(img)
        # print(img.shape)
    for img in img_list[2:]:
        img = transform2numpy(img)
        img = torch.from_numpy(img).float().unsqueeze(0)
        ret_img.append(img)
    return ret_img