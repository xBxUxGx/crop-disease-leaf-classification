# coding:utf-8
from skimage.feature import hog
import numpy as np
from PIL import Image
import glob
from concurrent.futures import ProcessPoolExecutor


def extract_hog(img_array):
    """单张图像 HOG 特征提取"""
    return hog(img_array, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), channel_axis=-1)


def _process_one_image(args):
    """单张图片全流程（供多进程调用）"""
    img_file, img_size, class_label = args
    img = Image.open(img_file).convert('RGB').resize((img_size, img_size))  # 读取并缩放
    feat = extract_hog(np.array(img))  # HOG 提取
    return feat, class_label


def load_data_hog(data_dir, img_size, n_jobs=4):
    """多进程并行加载目录下所有图片并提取 HOG 特征"""
    tasks = []
    for class_label, class_dir in enumerate(sorted(glob.glob(data_dir + '/*'))):  # 遍历类别
        for img_file in glob.glob(class_dir + '/*'):
            tasks.append((img_file, img_size, class_label))  # 打包任务参数

    X, y = [], []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:  # 多进程并行
        for feat, label in executor.map(_process_one_image, tasks):
            X.append(feat)  # 特征向量
            y.append(label)  # 标签

    return np.array(X), np.array(y)  # 汇总为 numpy 数组
