# 农作物病害叶片分类

《人工智能导论》课程设计 —— 基于 PlantVillage 数据集的 10 类病害叶片分类，对比三种技术路线。

## 运行

```bash
source .venv/Scripts/activate
python main.py --model ml       # HOG + SVM / 随机森林
python main.py --model simple   # 自定义 CNN（SimpleNet）
python main.py --model resnet   # 迁移学习（ResNet18）
python main.py --model all      # 全部对比 + 图表输出（默认）
```

## 方法

| 方法 | 说明 |
|------|------|
| HOG + SVM | 方向梯度直方图（9 方向，8×8 cell，2×2 block），SVM（RBF C=10） |
| HOG + 随机森林 | 相同 HOG 特征，随机森林（200 棵树，最大深度 20） |
| SimpleNet | 3 层 Conv-BN-ReLU-MaxPool，参数量约 120 万 |
| ResNet18 | ImageNet 预训练权重迁移，替换 FC 层为 10 分类 |

## 结果

| 方法 | 测试准确率 | Macro F1 | 训练耗时 |
|------|-----------|----------|----------|
| ResNet18 | 99.33% | 0.9929 | ~22 min |
| SimpleNet | 98.66% | 0.9863 | ~22 min |
| SVM | 85.43% | 0.8428 | ~85 s |
| 随机森林 | 71.44% | 0.6308 | ~24 s |

## 数据集

PlantVillage 公开数据集（Mendeley Data: 10.17632/tywbtsjrjv.1），选取 10 类约 8:2 划分训练/测试集。

| 类别 | 病害 | 训练/测试 |
|------|------|-----------|
| Apple___Apple_scab | 苹果黑星病 | 504/126 |
| Apple___Cedar_apple_rust | 苹果雪松锈病 | 220/55 |
| Corn___Cercospora_leaf_spot | 玉米灰斑病 | 410/103 |
| Corn___Common_rust | 玉米普通锈病 | 953/239 |
| Grape___Black_rot | 葡萄黑腐病 | 944/236 |
| Potato___Early_blight | 马铃薯早疫病 | 800/200 |
| Potato___Late_blight | 马铃薯晚疫病 | 800/200 |
| Strawberry___Leaf_scorch | 草莓叶焦病 | 887/222 |
| Tomato___Early_blight | 番茄早疫病 | 800/200 |
| Tomato___Septoria_leaf_spot | 番茄斑枯病 | 1416/355 |

## 输出

运行后 `output/` 目录生成：训练曲线 `curve_*.png`、混淆矩阵 `cm_*.png`、对比柱状图 `comparison.png`、模型权重 `*.pth`。

## 环境

Python 3.12，PyTorch 2.6.0+cu124，NVIDIA 驱动 595.79，RTX 4060 Laptop 8GB。

```bash
pip install torch torchvision numpy matplotlib scikit-learn scikit-image Pillow
```
