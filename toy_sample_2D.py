import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# 定义文件路径
base_path = r"F:\\DiffuseReg-master-2D\\results_nii\\fusion_sample_2D_15"
# 定义保存图片的文件夹路径
save_path = r"F:\\DiffuseReg-master-2D\\results\\fusion_16"

# 确保保存路径存在
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"创建保存目录: {save_path}")

# 定义文件名模板
file_templates = [
    "fixed{}.nii.gz",
    "moving{}.nii.gz",
    "ori_moving{}.nii.gz",
    "flow{}.nii.gz",
    "regist{}.nii.gz",
    "fusion_img{}.nii.gz"
]

# 定义数字范围
number_range = range(24)  # 从 0 到 23

# 遍历每个编号
for num in number_range:
    # 使用当前编号生成文件名列表
    file_names = [template.format(num) for template in file_templates]

    # 加载所有文件并提取数据
    images = []
    titles = []

    for file_name in file_names:
        file_path = os.path.join(base_path, file_name)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue
            
        nifti_file = nib.load(file_path)
        image_data = nifti_file.get_fdata()

        # 获取图像的维度
        ndims = len(image_data.shape)
        print(f"文件: {file_name}, 维度: {image_data.shape}")

        # 对于2D图像，直接使用
        if ndims == 2:
            slice_image = image_data
        # 对于3D图像
        elif ndims == 3:
            # 如果是流场图像，特殊处理
            if "flow" in file_name:
                # 检查流场图像的形状
                if image_data.shape[0] == 2 and image_data.shape[1] == 256 and image_data.shape[2] == 256:
                    # 如果形状为(2,256,256)，需要转置为(256,256,2)
                    image_data = np.transpose(image_data, (1, 2, 0))
                
                # 计算流场的幅度（两个方向的平方和的平方根）
                if image_data.shape[2] == 2:
                    # 计算流场幅度
                    slice_image = np.sqrt(np.sum(image_data**2, axis=2))
                else:
                    # 如果不是二通道，取第一个通道
                    slice_image = image_data[:, :, 0]
            # 如果通道数小于等于3，可能是RGB图像，取平均值
            elif image_data.shape[2] <= 3:
                slice_image = np.mean(image_data, axis=2)
            # 如果是多通道图像，取中间切片
            else:
                middle_slice = image_data.shape[2] // 2
                slice_image = image_data[:, :, middle_slice]
        else:
            print(f"不支持的维度数: {ndims}，文件: {file_name}")
            continue

        images.append(slice_image)
        titles.append(file_name)

    # 如果没有成功加载任何图像，跳过当前编号
    if not images:
        print(f"编号 {num} 没有找到有效图像，跳过")
        continue

    # 创建画布
    n_images = len(images)
    n_cols = min(3, n_images)  # 最多3列
    n_rows = (n_images + n_cols - 1) // n_cols  # 计算需要的行数
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))  # 动态调整画布大小
    
    # 确保axes是二维数组
    if n_images == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes = axes.flatten()  # 将二维数组展平为一维数组

    # 绘制每个图像
    for i, (image, title) in enumerate(zip(images, titles)):
        ax = axes[i]
        im = ax.imshow(image, cmap='gray')
        ax.set_title(title, fontsize=10)
        ax.axis('off')  # 关闭坐标轴
        
        # 为流场图像添加颜色条
        if "flow" in title:
            plt.colorbar(im, ax=ax)

    # 隐藏未使用的子图
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
        axes[i].set_visible(False)

    # 调整子图间距
    plt.tight_layout()
    
    # 使用新的保存路径
    save_file_path = os.path.join(save_path, f"visualization_{num}.png")
    plt.savefig(save_file_path)  # 保存图像到新路径
    print(f"保存图像: {save_file_path}")
    plt.close()  # 关闭图形以释放内存

print("可视化完成！所有图像已保存到:", save_path)