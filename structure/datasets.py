# coding:utf-8
import torch.utils.data as Data
import torchvision as tv
import glob
from PIL import Image


# 自定义数据集读取（懒加载 + 内存缓存）
class MyDataset(Data.Dataset):
    def __init__(self, PARAMS, isTrain=True):
        super().__init__()
        transform_list = [
            tv.transforms.Resize([PARAMS.img_size, PARAMS.img_size]),  # 统一尺寸
            tv.transforms.ToTensor(),  # 转张量
        ]
        if isTrain:
            transform_list.append(tv.transforms.RandomHorizontalFlip())  # 随机水平翻转
            transform_list.append(tv.transforms.RandomRotation(10))  # 随机旋转 ±10°
        self.Trans = tv.transforms.Compose(transform_list)

        path = PARAMS.train_dir if isTrain else PARAMS.test_dir  # 选择训练或测试目录

        files = []
        labels = []
        self.cache = []  # 缓存标志，False 表示未加载
        for class_label, class_dir in enumerate(sorted(glob.glob(path + '/*'))):  # 遍历类别目录
            for img_file in glob.glob(class_dir + '/*'):
                files.append(img_file)  # 存放图片路径
                labels.append(class_label)  # 目录索引即为类别标签
                self.cache.append(False)  # 缓存占位
        self.files = files
        self.labels = labels
        self.size = len(self.files)

    def __getitem__(self, index):
        if self.cache[index] is False:  # 首次访问
            img = Image.open(self.files[index]).convert('RGB')  # 打开图像
            img = self.Trans(img)  # transform 处理
            self.cache[index] = [img, self.labels[index]]  # 存入缓存
        return self.cache[index]

    def __len__(self):
        return self.size
