---
title: 使用概率方法处理得分数据
date: 2019/7/28 11:01:00
body: [article]
description: "&emsp;&emsp;本文简述了笔者使用概率方法处理同学间的打分数据。最终建立了一个能够输出权衡后的分数的模型。"
category:
- 技术文档
tags:
- 机器学习
- 概率论
---

## &sect;写在前面

&emsp;&emsp;在教学的过程中，经常会有学员之间互相评价的环节。在这个环节中，每个人都会对自己和别人的表现进行打分，并最后产生大量的数据。由于每个人的评分标准的差异等因素，最终得到的数据不能通过求均值的方法计算每个人最终的得分。

## &sect;数据预处理

### 预处理策略

1. 剔除所有缺失数据
2. 补全少量缺失数据
3. 补全所有数据

### Choice 1

* 获得17*23的矩阵
* 分析每人自评分与其他评分的关系

### Choice 2

* 保留仅缺失部分被评分数据的用户（列）

### Choice 3

* 补全策略？

## &sect;问题抽象

* 目标：获得真实分数函数：`f(**S**)=**s**`
* 其中：
  * `**S**`：
    * 原始评分矩阵，为m*n的矩阵；
    * m为评分人数，n为被评分人数；
    * 矩阵中没有缺失数据，即数据已经经过预处理；
  * `**s**`：最终分数向量，为n维向量，代表每个人最终的分数。
* 影响准确性的因素：
  1. 每个人的评分标准不同
  2. 每个人对自己和别人的评分标准不同

## &sect;处理算法

1. 消除个人评分标准导致的误差：对每个评分人的评分进行标准化，得到每个被评分人的偏差值，再对每个被评分人的偏差值进行平均。
2. 调整整体分数：将每个人的偏差值映射至与原数据具有相同的均值和离散程度的空间中，得到最后的分数。

