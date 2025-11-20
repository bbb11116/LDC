from utils.mydateset import YSTestDataset
import matplotlib.pyplot as plt
import numpy as np

data_root = r'D:\Workspace\projects\work\yingsu\train\data\data'

dataset = YSTestDataset(data_root, 1200, 1600)

data  = dataset[10]
image, gt_aug = data['images'], data["labels"]

img_to_show = image.numpy()   # 将PyTorch Tensor转换为numpy数组
img_to_show = img_to_show.transpose(1, 2, 0) # 从 (C, H, W) 转置为 (H, W, C)
# 处理gt_aug：移除通道维度并转换为numpy数组
gt_to_show = gt_aug.numpy()[0]  # 从 (1, H, W) 变为 (H, W)


# 创建一个包含两个子图的图像窗口，并排显示原图和标签
fig, axes = plt.subplots(1, 2, figsize=(10, 5)) # 1行2列，宽度10英寸，高度5英寸

# 显示增强后的图像
# 如果图像值范围是[0,1]，但matplotlib期望[0,1]对于float，则可以省略
# 如果值可能超出[0,1]，可以裁剪
img_to_show = np.clip(img_to_show, 0, 1)
axes[0].imshow(img_to_show)
axes[0].set_title('Augmented Image')
axes[0].axis('off')  # 不显示坐标轴

# 显示增强后的标签（灰度图）
axes[1].imshow(gt_to_show, cmap='gray') # 使用灰度颜色映射
axes[1].set_title('Augmented Label (GT)')
axes[1].axis('off')

plt.tight_layout() # 自动调整子图参数，使之填充整个图像区域，避免重叠
plt.show()
