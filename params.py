# coding:utf-8
class Params(object):
    lr = 0.01           # 学习率
    batch_size = 256     # 批量大小（增大以提高 GPU 利用率）
    train_dir = "./data/train"  # 训练集路径
    test_dir = "./data/test"    # 测试集路径
    img_size = 64        # 输入图像尺寸（缩放到 img_size × img_size）
    use_gpu = True       # 是否使用 GPU（False 则 CPU）
    num_workers = 5      # 数据加载子进程数（Windows 下多进程可能出错，可改为 0）
    num_classes = 10     # 类别数
    epoch = 50           # 训练轮数（仅 CNN 有效，ML 无此参数）
    output_dir = "./output"     # 输出目录（模型、图表）
    print_step = 5       # 每 N 轮打印一次训练/测试准确率
    save_step = 10       # 每 N 轮保存一次模型
