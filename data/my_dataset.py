from torch.utils.data import Dataset
import os
import numpy as np
import torch
from PIL import Image


class Dataset2D(Dataset):

    def __init__(self, dataroot, fineSize, split='train'):
        """
        初始化2D数据集
        
        参数:
            dataroot: 数据根目录, 应该是包含CT和MRI两个子文件夹的目录
            fineSize: 目标图像尺寸，格式为(height, width)
            split: 数据集分割类型，默认为'train'
        """
        self.split = split
        self.dataroot = dataroot
        self.fineSize = fineSize
        
        # 获取CT和MRI文件夹路径
        self.ct_folder = os.path.join(dataroot, 'CT')
        self.mri_folder = os.path.join(dataroot, 'MRI')
        
        # 确保文件夹存在
        assert os.path.exists(self.ct_folder), f"CT文件夹不存在: {self.ct_folder}"
        assert os.path.exists(self.mri_folder), f"MRI文件夹不存在: {self.mri_folder}"
        
        # 获取所有CT图像文件名
        self.image_files = [f for f in os.listdir(self.ct_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 确保CT和MRI文件夹中有相同名称的文件
        for img_file in self.image_files:
            mri_path = os.path.join(self.mri_folder, img_file)
            assert os.path.exists(mri_path), f"MRI图像不存在: {mri_path}"
        
        self.data_len = len(self.image_files)
        print(f"找到{self.data_len}对CT-MRI图像")

    def __len__(self):
        """返回数据集长度"""
        return self.data_len

    def __getitem__(self, index):
        """
        获取指定索引的数据项
        
        参数:
            index: 数据索引
            
        返回:
            包含CT图像、MRI图像、标签和索引的字典
        """
        # 获取图像文件名
        img_file = self.image_files[index]
        
        # 构建CT和MRI图像路径
        ct_path = os.path.join(self.ct_folder, img_file)
        mri_path = os.path.join(self.mri_folder, img_file)
        
        # 读取CT图像
        ct_img = Image.open(ct_path).convert('L')  # 转换为灰度图
        ct_img = np.array(ct_img).astype(np.float32)
        
        # 读取MRI图像
        mri_img = Image.open(mri_path).convert('L')  # 转换为灰度图
        mri_img = np.array(mri_img).astype(np.float32)
        

        # 数据归一化，范围[0, 1]
        ct_img = self._normalize_image(ct_img)
        mri_img = self._normalize_image(mri_img)

        ct_img = torch.from_numpy(ct_img).float()
        mri_img = torch.from_numpy(mri_img).float()

        moving = ct_img.unsqueeze(0).unsqueeze(0)
        fixed = mri_img.unsqueeze(0).unsqueeze(0)

        moving = moving.squeeze(0)
        fixed = fixed.squeeze(0)
        ori_moving = moving


        fixedM = torch.tensor([0], dtype=torch.float32)
        movingM = torch.tensor([1], dtype=torch.float32)

        
        return {'M': moving, 'F': fixed, 'OM': ori_moving, 'MS': movingM, 'FS': fixedM, 'Index': index}

    
    def _normalize_image(self, img):
        """
        归一化图像到[0, 1]范围
        
        参数:
            img: 输入图像
            
        返回:
            归一化后的图像
        """
        img -= img.min()
        img /= img.std()
        img -= img.min()
        img /= img.max()
        return img
    
    def _resize_images(self, ct_img, mri_img):
        """
        调整图像尺寸到目标大小
        
        参数:
            ct_img: CT图像
            mri_img: MRI图像
            
        返回:
            调整尺寸后的CT和MRI图像
        """
        h, w = ct_img.shape
        target_h, target_w = self.fineSize
        
        # 如果图像尺寸大于目标尺寸，进行中心裁剪
        if h >= target_h and w >= target_w:
            sh = int((h - target_h) / 2)
            sw = int((w - target_w) / 2)
            ct_img = ct_img[sh:sh + target_h, sw:sw + target_w]
            mri_img = mri_img[sh:sh + target_h, sw:sw + target_w]
        else:
            # 如果图像尺寸小于目标尺寸，进行零填充
            ct_img_new = np.zeros((target_h, target_w), dtype=np.float32)
            mri_img_new = np.zeros((target_h, target_w), dtype=np.float32)
            
            sh = max(0, int((target_h - h) / 2))
            sw = max(0, int((target_w - w) / 2))
            
            h_to_copy = min(h, target_h)
            w_to_copy = min(w, target_w)
            
            ct_img_new[sh:sh + h_to_copy, sw:sw + w_to_copy] = ct_img[:h_to_copy, :w_to_copy]
            mri_img_new[sh:sh + h_to_copy, sw:sw + w_to_copy] = mri_img[:h_to_copy, :w_to_copy]
            
            ct_img, mri_img = ct_img_new, mri_img_new
            
        return ct_img, mri_img


class Dataset2D_test(Dataset):
    def __init__(self, dataroot, fineSize, split='train'):
        """
        初始化2D数据集
        
        参数:
            dataroot: 数据根目录, 应该是包含CT和MRI两个子文件夹的目录
            fineSize: 目标图像尺寸，格式为(height, width)
            split: 数据集分割类型，默认为'train'
        """
        self.split = split
        self.dataroot = dataroot
        self.fineSize = fineSize
        
        # 获取CT和MRI文件夹路径
        self.ct_folder = os.path.join(dataroot, 'CT')
        self.mri_folder = os.path.join(dataroot, 'MRI')
        
        # 确保文件夹存在
        assert os.path.exists(self.ct_folder), f"CT文件夹不存在: {self.ct_folder}"
        assert os.path.exists(self.mri_folder), f"MRI文件夹不存在: {self.mri_folder}"
        
        # 获取所有CT图像文件名
        self.image_files = [f for f in os.listdir(self.ct_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 确保CT和MRI文件夹中有相同名称的文件
        for img_file in self.image_files:
            mri_path = os.path.join(self.mri_folder, img_file)
            assert os.path.exists(mri_path), f"MRI图像不存在: {mri_path}"
        
        self.data_len = len(self.image_files)
        print(f"找到{self.data_len}对CT-MRI图像")

    def __len__(self):
        """返回数据集长度"""
        return self.data_len

    def __getitem__(self, index):
        """
        获取指定索引的数据项
        
        参数:
            index: 数据索引
            
        返回:
            包含CT图像、MRI图像、标签和索引的字典
        """
        # 获取图像文件名
        img_file = self.image_files[index]
        
        # 构建CT和MRI图像路径
        ct_path = os.path.join(self.ct_folder, img_file)
        mri_path = os.path.join(self.mri_folder, img_file)
        
        # 读取CT图像
        ct_img = Image.open(ct_path).convert('L')  # 转换为灰度图
        ct_img = np.array(ct_img).astype(np.float32)
        
        # 读取MRI图像
        mri_img = Image.open(mri_path).convert('L')  # 转换为灰度图
        mri_img = np.array(mri_img).astype(np.float32)
        

        # 数据归一化，范围[0, 1]
        ct_img = self._normalize_image(ct_img)
        mri_img = self._normalize_image(mri_img)

        fixed = mri_img
        moving = ct_img

        fixed = fixed.reshape(1, fixed.shape[0], fixed.shape[1], 1)
        moving = moving.reshape(1, moving.shape[0], moving.shape[1], 1)
        
        # 将 NumPy 数组转换为 PyTorch 张量
        fixed = torch.from_numpy(fixed)
        moving = torch.from_numpy(moving)
        ori_moving = moving


        if fixed.shape[-1] == 1:  # 检查最后一个维度是否为通道维度
            fixed = fixed.permute(0, 3, 1, 2)  # 从 [B, H, W, C] 转换为 [B, C, H, W]
            moving = moving.permute(0, 3, 1, 2)  # 从 [B, H, W, C] 转换为 [B, C, H, W]
            ori_moving = moving

            fixed = fixed.squeeze(0)  # 移除批量维度
            moving = moving.squeeze(0)  # 移除批量维度
            ori_moving = ori_moving.squeeze(0)  # 移除批量维度

        # CT图像标签为0，MRI图像标签为1
        fixedM = torch.tensor([0], dtype=torch.float32)  # CT图像的标签
        movingM = torch.tensor([1], dtype=torch.float32)  # MRI图像的标签
        
        return {'M': moving, 'F': fixed, 'OM': ori_moving, 'MS': movingM, 'FS': fixedM, 'Index': index}
    
    def _normalize_image(self, img):
        """
        归一化图像到[0, 1]范围
        
        参数:
            img: 输入图像
            
        返回:
            归一化后的图像
        """
        # 简单的最小-最大归一化，将图像归一化到[0, 1]范围
        min_val = img.min()
        max_val = img.max()
        if max_val > min_val:  # 避免除以零
            img = (img - min_val) / (max_val - min_val)
        return img
