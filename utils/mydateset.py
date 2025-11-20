from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import torch
import albumentations as A

class YSDataset(Dataset):
    def __init__(self,
                 data_root,
                 img_height,
                 img_width,
                 mean_bgr=[103.939, 116.779, 123.68],
                 ):
        self.data_root = data_root
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = mean_bgr

        self.data_index = self._build_index()

    def _build_index(self):
        sample_indices = []

        data_train_folder = os.path.join(self.data_root, 'imgs', 'train','real')
        gt_train_folder = os.path.join(self.data_root, 'edges', 'train','real')
        train_list = os.listdir(data_train_folder)
        for item in train_list:
            sample_indices.append(
                (os.path.join(data_train_folder, item),
                 os.path.join(gt_train_folder, item)))

        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path = self.data_index[idx]

        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        image, label = self.transform(img=image, gt=label)
        return dict(images=image, labels=label)

    def transform(self, img, gt):
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]
        gt /= 255.

        img = np.array(img, dtype=np.float32)/255.0

        # 同时应用于图像和gt
        double_transform_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),  # 50%概率水平翻转[1,11](@ref)
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=(-90,90), p=0.5),
            #A.RandomSizedCrop(
                #min_max_height=(self.img_height*0.8, self.img_height-10),
                #size=(self.img_height, self.img_width),
                #w2h_ratio=self.img_width/float(self.img_height),
                #p=0.5
                #),

            # shift_limit: 平移范围 (比例)，scale_limit: 缩放范围，rotate_limit: 旋转角度范围[6,14](@ref)
            # A.ElasticTransform(alpha=100,sigma=10, p=0.1)
        ], additional_targets={'gt': 'mask'})  # 指定对 gt 也应用相同的变换（视为mask）

        # 仅用于图像
        single_transform_pipeline = A.Compose([
            A.RandomShadow(shadow_roi=(0,0,1,1), num_shadows_limit=(1,2), shadow_dimension=5, shadow_intensity_range=(0.1,0.4), p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),  # 随机调整亮度和对比度[4,7](@ref)
            A.Illumination(mode='corner',p=0.3),
            A.Defocus(radius=(1,3), alias_blur=0.1, p=0.1),
            A.Blur(blur_limit=(3, 5), p=0.1)  # 模糊[4](@ref)
        ])

        augmented = double_transform_pipeline(image=img, gt=gt)
        img_aug, gt_aug = augmented['image'], augmented['gt']

        augmented = single_transform_pipeline(image=img_aug)
        img_aug = augmented['image']

        # img -= self.mean_bgr
        img_aug = img_aug.transpose((2, 0, 1))
        img_aug = torch.from_numpy(img_aug.copy()).float()
        gt_aug = torch.from_numpy(np.array([gt_aug])).float()

        return img_aug, gt_aug

class YSTestDataset(Dataset):
    def __init__(self,
                 data_root,
                 img_height,
                 img_width,
                 mean_bgr=[103.939, 116.779, 123.68],
                 ):
        self.data_root = data_root
        self.mean_bgr = mean_bgr
        self.img_height = img_height
        self.img_width = img_width
        self.data_index = self._build_index()

    def _build_index(self):
        sample_indices = []

        data_train_folder = os.path.join(self.data_root, 'imgs', 'test')
        gt_train_folder = os.path.join(self.data_root, 'edges', 'test')
        train_list = os.listdir(data_train_folder)
        for item in train_list:
            sample_indices.append(
                (os.path.join(data_train_folder, item),
                 os.path.join(gt_train_folder, item)))

        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path = self.data_index[idx]
        img_name = os.path.basename(image_path)
        file_name = os.path.splitext(img_name)[0] + ".png"

        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if  os.path.exists(label_path):
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        else:
            label = None

        # 记录原始尺寸
        im_shape = [image.shape[0], image.shape[1]]

        #transformer
        image, label = self.transform(img=image, gt=label)

        return dict(
            images=image,
            labels=label,
            file_names=file_name,
            image_shape=im_shape
        )

    def transform(self, img, gt):
        #image
        img = np.array(img, dtype=np.float32)/255.0
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        #gt
        if gt is not None:
            gt = np.array(gt, dtype=np.float32)
            if len(gt.shape) == 3:
                gt = gt[:, :, 0]  # 取单通道
            gt /= 255.0  # 归一化
            gt = torch.from_numpy(np.array([gt])).float()
        else:
            # 创建空标签
            gt = torch.zeros(1, img.shape[1], img.shape[2])

        return img, gt
