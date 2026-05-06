# 表格语义关系提取 (Table Semantic Relationship Extraction) - SCNU AI Competition 2026

本项目旨在实现对表格列对（Subject, Object）之间语义关系的精准提取。我们基于 PaddlePaddle 框架，设计了一套针对**长尾分布**和**少样本（Few-Shot）**场景优化的稳健模型架构，在竞赛中表现出色。

## 项目核心：PaddlePaddle 优化版

我们在 `src/` 目录下实现了完整且经过验证的优化方案，相比基础 Baseline 具有显著的性能提升。

### 核心优化点 (Core Highlights)
- **更强的底座 (ERNIE 3.0)**：选用百度自研的 ERNIE 3.0 base 预训练模型，在中文及跨语言理解上优于传统的 BERT。
- **Mean-Pooling 池化策略**：放弃单一的 `[CLS]` 向量，采用整句 Embedding 的平均池化，更平衡地捕捉表格单元格的短文本语义。
- **对抗训练 (FGM)**：在词向量层引入微小扰动（ε=1.0），强迫模型学习更本质的特征，提升在测试集上的泛化能力。
- **R-Drop 一致性正则**：通过双向 KL 散度约束，确保模型对同一输入的不同 Dropout 预测保持一致，显著抑制小样本过拟合。
- **评价指标对齐 (Weighted Loss)**：直接将赛题官方的 $m_{weights}$ 计算公式注入 Loss 函数，使训练过程完全对齐 $Score_{final}$ 评分标准。

---

## 快速开始

### 1. 环境准备
确保您的环境中已安装 `paddlepaddle-gpu`、`paddlenlp` 以及 `pandas`、`scikit-learn` 等基础库。

### 2. 训练模型
运行 `src/` 目录下的优化脚本：
```bash
python src/train_paddle_optm.py --device gpu --epoch 10 --batch_size 32
```
*提示：默认已开启混合精度训练 (AMP) 以加速运行。*

### 3. 模型推理
使用训练产出的最佳权重进行预测：
```bash
python src/infer_paddle_optm.py --model_path ./cpa_output/cpa_TIMESTAMP/best_model.pdparams --labels_path ./cpa_output/cpa_TIMESTAMP/label_classes.txt
```
生成的预测结果将保存至 `result/submission.csv`。

---

## 目录结构说明
```text
SCNU_AI_Competition_2026/
│  question.md              # 官方评价指标定义
│  README.md                # 本项目说明文档
│
├─src/                      # 核心优化代码 (主要修改区域)
│  ├─train_paddle_optm.py   # 优化版训练脚本 (FGM + R-Drop + Weighted Loss)
│  └─infer_paddle_optm.py   # 优化版推理脚本
│
├─baseline/                 # 原始 Baseline 代码 (保持不变)
│  ├─train.py
│  └─infer.py
│
├─dataset/                  # 数据集存放目录
│  ├─Train_Set/             # 训练集 CSV 文件
│  └─test.csv               # 待预测测试集
│
├─cpa_output/               # 训练产出目录 (包含权重、日志、标签表)
└─result/                   # 推理结果存放目录
```

## 评价指标回顾
本项目严格遵循以下 $Score_{final}$ 计算规则：

$$Score_{final} = \frac{\sum_{m} (m_{weights} \times m_{score})}{\sum_{m} m_{weights}}$$

其中 $m_{weights}$ 根据样本数量反向加权，确保少样本类别在最终得分中占有更高权重。我们的模型通过在 Loss 层引入相同权重，实现了“所练即所得”。

---
*祝您在 SCNU AI Competition 2026 中取得优异成绩！*
