# coding:utf-8
import torch, os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from params import Params
from structure.model import SimpleNet, ResNetTransfer
from structure.datasets import MyDataset
from structure.feature_extract import load_data_hog
from torch.utils.data import DataLoader

PARAMS = Params()
if PARAMS.use_gpu and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
if not os.path.exists(PARAMS.output_dir):
    os.makedirs(PARAMS.output_dir)


def run_ml():
    print("===== 传统机器学习（HOG + SVM / 随机森林）=====")

    # ---提取 HOG 特征---#
    t0 = time.time()
    X_train, y_train = load_data_hog(PARAMS.train_dir, PARAMS.img_size, PARAMS.num_workers)  # 训练集特征
    X_test, y_test = load_data_hog(PARAMS.test_dir, PARAMS.img_size, PARAMS.num_workers)  # 测试集特征
    print("HOG 特征提取完成，维度: %s，耗时: %.1fs" % (str(X_train.shape), time.time() - t0))

    # ---训练并评估分类器---#
    results = {}
    for name, clf in [("SVM", SVC(kernel='rbf', C=10, gamma='scale')),
                       ("RandomForest", RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=PARAMS.num_workers))]:
        t0 = time.time()
        clf.fit(X_train, y_train)  # 训练
        train_pred = clf.predict(X_train)  # 训练集预测
        test_pred = clf.predict(X_test)  # 测试集预测
        train_acc = accuracy_score(y_train, train_pred) * 100
        test_acc = accuracy_score(y_test, test_pred) * 100
        _, _, f1, _ = precision_recall_fscore_support(y_test, test_pred, average='macro', zero_division=0)
        elapsed = time.time() - t0
        print("%s: 训练准确率=%.2f%%, 测试准确率=%.2f%%, F1=%.4f, 耗时=%.1fs" % (name, train_acc, test_acc, f1, elapsed))
        cm = confusion_matrix(y_test, test_pred)  # 混淆矩阵
        results[name] = {"train_acc": train_acc, "test_acc": test_acc, "f1": f1, "time": elapsed, "cm": cm}
    return results


def train_one_epoch(net, loader, optimizer, loss_fn):
    net.train()
    correct, total_loss, total = 0, 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()
        outputs = net(imgs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        correct += (outputs.argmax(1) == labels).sum().item()
        total_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
    return 100.0 * correct / total, total_loss / total


def evaluate(net, loader):
    net.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            outputs = net(imgs)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds) * 100
    _, _, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, f1, cm


def run_cnn(model_name):
    print("===== %s =====" % model_name)

    # ---获得数据集---#
    train_data = MyDataset(PARAMS, True)  # 训练集
    test_data = MyDataset(PARAMS, False)  # 测试集
    train_loader = DataLoader(train_data, batch_size=PARAMS.batch_size, num_workers=PARAMS.num_workers, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=PARAMS.batch_size, num_workers=PARAMS.num_workers, shuffle=False, pin_memory=True)
    print("训练样本: %d, 测试样本: %d" % (len(train_data), len(test_data)))

    # ---构建模型、优化器、损失函数---#
    if model_name == "SimpleNet":
        net = SimpleNet(PARAMS.num_classes).to(DEVICE)  # 自定义 CNN
    else:
        net = ResNetTransfer(PARAMS.num_classes).to(DEVICE)  # 迁移学习
    optimizer = torch.optim.SGD(net.parameters(), lr=PARAMS.lr, momentum=0.9)  # SGD 优化器
    loss_fn = torch.nn.CrossEntropyLoss().to(DEVICE)  # 交叉熵损失

    # ---训练与评估---#
    train_accs, test_accs = [], []  # 记录每轮准确率
    t0 = time.time()
    for epoch in range(PARAMS.epoch):
        train_acc, train_loss = train_one_epoch(net, train_loader, optimizer, loss_fn)  # 训练一轮
        if (epoch + 1) % PARAMS.print_step == 0 or epoch == 0 or epoch == PARAMS.epoch - 1:
            test_acc, f1, cm = evaluate(net, test_loader)  # 全量评估
        else:
            test_acc = test_accs[-1] if test_accs else 0  # 复用上一轮测试准确率占位
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        if (epoch + 1) % PARAMS.print_step == 0 or epoch == 0 or epoch == PARAMS.epoch - 1:
            print("【%d/%d轮】loss=%.4f, 训练准确率=%.2f%%, 测试准确率=%.2f%%" %
                  (epoch + 1, PARAMS.epoch, train_loss, train_acc, test_acc))
        if (epoch + 1) % PARAMS.save_step == 0:
            torch.save(net.state_dict(), PARAMS.output_dir + "/%s_%d.pth" % (model_name, epoch + 1))  # 保存模型

    elapsed = time.time() - t0
    final_test_acc, final_f1, final_cm = evaluate(net, test_loader)  # 最终评估
    print("%s 最终测试准确率=%.2f%%, F1=%.4f, 总耗时=%.1fs" % (model_name, final_test_acc, final_f1, elapsed))

    # ---绘制训练曲线---#
    plt.figure()
    plt.plot(range(1, PARAMS.epoch + 1), train_accs, label='Train')
    plt.plot(range(1, PARAMS.epoch + 1), test_accs, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(model_name + ' Training Curve')
    plt.legend()
    plt.savefig(PARAMS.output_dir + '/curve_%s.png' % model_name, dpi=100)

    return {"train_acc": train_accs[-1], "test_acc": final_test_acc, "f1": final_f1, "time": elapsed, "cm": final_cm}


CLASS_NAMES = ["Apple___Apple_scab", "Apple___Cedar_apple_rust", "Corn___Cercospora_leaf_spot",
               "Corn___Common_rust", "Grape___Black_rot", "Potato___Early_blight",
               "Potato___Late_blight", "Strawberry___Leaf_scorch", "Tomato___Early_blight",
               "Tomato___Septoria_leaf_spot"]


def plot_comparison(all_results):
    """绘制多方法对比图（准确率 / F1 / 耗时三子图）和混淆矩阵"""
    names = list(all_results.keys())
    test_accs = [all_results[n]["test_acc"] for n in names]
    f1s = [all_results[n]["f1"] for n in names]
    times = [all_results[n]["time"] for n in names]

    # ---对比柱状图---#
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    axes[0].bar(names, test_accs, color=colors[:len(names)])
    axes[0].set_title('Test Accuracy (%)')
    axes[0].set_ylim(0, 105)
    for i, v in enumerate(test_accs):
        axes[0].text(i, v + 1, "%.2f" % v, ha='center')

    axes[1].bar(names, f1s, color=colors[:len(names)])
    axes[1].set_title('Macro F1 Score')
    for i, v in enumerate(f1s):
        axes[1].text(i, v + 0.01, "%.4f" % v, ha='center')

    axes[2].bar(names, times, color=colors[:len(names)])
    axes[2].set_title('Training Time (s)')
    for i, v in enumerate(times):
        axes[2].text(i, v + 1, "%.1f" % v, ha='center')

    plt.tight_layout()
    plt.savefig(PARAMS.output_dir + '/comparison.png', dpi=150)
    plt.close()

    # ---混淆矩阵---#
    for name, res in all_results.items():
        plt.figure(figsize=(8, 6))
        plt.imshow(res["cm"], cmap='Blues')  # 蓝色渐变
        plt.colorbar()
        plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha='right', fontsize=8)
        plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES, fontsize=8)
        plt.title(name + ' Confusion Matrix')
        plt.tight_layout()
        plt.savefig(PARAMS.output_dir + '/cm_%s.png' % name, dpi=150)
        plt.close()


def run(model_type="all"):
    """入口函数：根据 --model 参数调度各方法"""
    all_results = {}
    if model_type in ("ml", "all"):
        all_results.update(run_ml())  # 传统 ML
    if model_type in ("simple", "all"):
        all_results["SimpleNet"] = run_cnn("SimpleNet")  # 自定义 CNN
    if model_type in ("resnet", "all"):
        all_results["ResNet18"] = run_cnn("ResNet18")  # 迁移学习
    if len(all_results) >= 2:
        print("\n===== 模型结果对比 =====")
        for name, res in all_results.items():
            print("%-15s  测试准确率=%.2f%%  F1=%.4f  耗时=%.1fs" % (name, res["test_acc"], res["f1"], res["time"]))
        plot_comparison(all_results)  # 绘制对比图和混淆矩阵
        print("对比图已保存至 %s" % PARAMS.output_dir)
