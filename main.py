# coding:utf-8
"""
课程：《人工智能导论》
课程设计：农作物病害叶片分类
作者：林野
学校：四川农业大学，信息工程学院
日期：2025.02.01
"""

import argparse
from structure.run import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="农作物病害叶片分类")
    parser.add_argument("--model", type=str, default="all",
                        choices=["ml", "simple", "resnet", "all"],
                        help="模型选择: ml/simple/resnet/all")
    args = parser.parse_args()
    run(args.model)
