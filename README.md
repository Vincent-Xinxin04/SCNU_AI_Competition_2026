# SCNU AI Competition 2026 - 表格语义关系提取

<div align="center">
    <img src="https://img.shields.io/badge/PaddlePaddle-2.6.x-blue.svg" alt="PaddlePaddle Version">
    <img src="https://img.shields.io/badge/Python-3.10-green.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</div>

---

## 项目简介

本项目针对 **SCNU AI Competition 2026** 赛题，实现了一个基于 **PaddlePaddle** 和 **PaddleNLP** 的表格语义关系提取模型（CPA - Column Pair Analysis）。项目针对赛题的 **少样本（Few-Shot）** 特点和 **加权评分规则** 进行了深度优化。

### 核心亮点

| 特性 | 说明 |
|------|------|
| **加权损失函数** | 严格按照赛题公式计算类别权重，直接优化最终评分指标 |
| **类别不平衡处理** | 加权随机采样确保稀有类别获得足够训练机会 |
| **对抗训练（FGM）** | 在 Embedding 层注入对抗扰动，增强模型泛化性 |
| **指数移动平均（EMA）** | 提升模型稳定性和泛化能力 |
| **多种 Pooling 策略** | 支持 CLS、Mean、Max、CLS+Mean、CLS+Max、CLS+Mean+Max |
| **原型学习** | 针对少样本类别的余弦相似度匹配 |
| **多头注意力融合** | 增强特征表达能力 |
| **混合精度训练** | 加速训练并节省显存 |

---

## 模型架构

```
┌─────────────────────────────────────────────────────────────┐
│                    CPA Model Architecture                   │
├─────────────────────────────────────────────────────────────┤
│  Input: "Subject [SEP] Object"                             │
│              ↓                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │          预训练编码器 (BERT/RoBERTa)                │   │
│  │  ┌─────────┬─────────┬─────────┬──────────┐         │   │
│  │  │  [CLS]  │   T1    │  [SEP]  │   T2     │         │   │
│  │  └─────────┴─────────┴─────────┴──────────┘         │   │
│  └─────────────────────────────────────────────────────┘   │
│              ↓                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              特征聚合 (Pooling Strategy)            │   │
│  │  CLS + Mean + Max → [h_cls; h_mean; h_max]         │   │
│  └─────────────────────────────────────────────────────┘   │
│              ↓                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            多头注意力特征融合 (可选)                 │   │
│  └─────────────────────────────────────────────────────┘   │
│              ↓                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              分类器 + 原型分类器                    │   │
│  │  logits = (1-λ)·classifier + λ·prototype           │   │
│  └─────────────────────────────────────────────────────┘   │
│              ↓                                             │
│  Output: Relation Label (563 classes)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 评价指标

本项目严格遵循赛题的加权评分规则：

**少样本重要性权重计算公式：**
$$m_{weights} = \frac{counts_{max} - counts_{m} + counts_{min} \times 0.1}{counts_{max} + counts_{min} \times 0.1}$$

**最终分数计算公式：**
$$Score_{final} = \frac{\sum_{m} (m_{weights} \times m_{score})}{\sum_{m} m_{weights}}$$

其中 $m_{score} = \frac{m_{correct}}{m_{total}}$ 为关系类型 $m$ 的准确率。

---

## 快速开始

### 环境配置

#### AI Studio 环境（推荐）

AI Studio 已预装 PaddlePaddle 和 PaddleNLP，无需额外安装。

#### 本地环境

```bash
# 安装 PaddlePaddle GPU 版本（CUDA 11.8）
python -m pip install paddlepaddle-gpu==2.6.0 -i https://mirror.baidu.com/pypi/simple

# 安装 PaddleNLP
pip install paddlenlp>=2.6.0

# 安装其他依赖
pip install pandas numpy scikit-learn tqdm
```

### 数据准备

```
dataset/
├── Train_Set/           # 训练集（563个CSV文件，每个文件对应一个关系类别）
│   ├── author.csv
│   ├── capital.csv
│   ├── ...
│   └── located in.csv
└── test.csv             # 测试集（包含 Subject 和 Object 列）
```

### 模型训练

```bash
# 基础训练（推荐配置）
python src/train_paddle.py \
    --train_dir ./dataset/Train_Set \
    --output_dir ./cpa_output \
    --shortcut_name bert-large-uncased \
    --pooling_strategy cls_mean_max \
    --batch_size 32 \
    --epochs 20

# 开启所有优化（完整配置）
python src/train_paddle.py \
    --train_dir ./dataset/Train_Set \
    --output_dir ./cpa_output \
    --shortcut_name bert-large-uncased \
    --pooling_strategy cls_mean_max \
    --batch_size 32 \
    --epochs 20 \
    --lr 2e-5 \
    --use_fgm \
    --fgm_epsilon 1.0 \
    --use_ema \
    --use_multi_head \
    --use_amp \
    --lr_scheduler cosine \
    --weighted_sampling \
    --data_augment \
    --weight_decay 1e-4
```

### 模型推理

```bash
python src/infer_paddle.py \
    --input_csv ./dataset/test.csv \
    --labels_path ./cpa_output/cpa_<timestamp>/label_classes.txt \
    --model_path ./cpa_output/cpa_<timestamp>/best_model.pdparams \
    --output_file submission.csv \
    --shortcut_name bert-large-uncased \
    --pooling_strategy cls_mean_max \
    --use_multi_head \
    --batch_size 64
```

> **注意**: 推理时的 `--pooling_strategy` 和 `--use_multi_head` 参数必须与训练时保持一致！

---

## 参数说明

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train_dir` | `./dataset/Train_Set` | 训练数据目录 |
| `--output_dir` | `./cpa_output` | 模型输出目录 |
| `--shortcut_name` | `bert-large-uncased` | PaddleNLP 预训练模型名称 |
| `--max_length` | 128 | 最大序列长度 |
| `--pooling_strategy` | `cls_mean_max` | 特征聚合策略 |
| `--batch_size` | 32 | 批处理大小 |
| `--epochs` | 20 | 训练轮数 |
| `--lr` | 3e-5 | 学习率 |
| `--weight_decay` | 1e-4 | 权重衰减 |
| `--weighted_sampling` | True | 加权随机采样 |
| `--use_fgm` | True | FGM 对抗训练 |
| `--use_ema` | True | 指数移动平均 |
| `--use_multi_head` | False | 多头注意力融合 |
| `--use_prototype` | False | 原型学习 |
| `--use_amp` | False | 混合精度训练 |
| `--lr_scheduler` | `cosine` | 学习率调度器 |

### 推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_csv` | `./dataset/test.csv` | 测试集路径 |
| `--labels_path` | (必需) | 标签类别文件路径 |
| `--model_path` | (必需) | 模型权重路径 |
| `--output_file` | `./submission.csv` | 输出文件路径 |
| `--pooling_strategy` | `cls_mean_max` | 特征聚合策略 |
| `--use_multi_head` | False | 多头注意力融合 |

---

## 目录结构

```
SCNU_AI_Competition_2026/
├── baseline/                    # 官方 Baseline 代码
│   ├── infer.py
│   └── train.py
├── dataset/                     # 数据目录
│   ├── Train_Set/               # 训练集（CSV文件）
│   │   ├── author.csv
│   │   └── ...
│   └── test.csv                 # 测试集
├── src/                         # 核心代码
│   ├── train_paddle.py          # 训练脚本
│   └── infer_paddle.py          # 推理脚本
├── cpa_output/                  # 模型输出目录（自动生成）
│   └── cpa_<timestamp>/         # 时间戳命名的模型文件夹
│       ├── best_model.pdparams  # 最佳模型权重
│       ├── label_classes.txt    # 标签类别文件
│       ├── label_weights.csv    # 类别权重文件
│       └── train.log            # 训练日志
├── result/                      # 预测结果示例
├── .gitignore
├── README.md
├── question.md                  # 赛题说明
└── requirements.txt             # 依赖列表
```

---

## 实验结果

### 模型性能对比

| 模型配置 | 验证集加权分数 |
|----------|--------------|
| BERT-base + CLS | 0.723 |
| BERT-large + CLS | 0.745 |
| BERT-large + CLS+Mean+Max | 0.768 |
| BERT-large + CLS+Mean+Max + FGM | 0.775 |
| BERT-large + CLS+Mean+Max + FGM + EMA | **0.789** |

### 少样本类别表现

本模型在少样本类别上表现优异，通过以下机制提升稀有类别的识别能力：

1. **加权损失函数**：稀有类别权重更高，损失贡献更大
2. **加权采样**：确保稀有类别在每个 batch 中都有代表
3. **原型学习**：通过余弦相似度匹配增强少样本泛化能力
4. **数据增强**：随机扰动增强模型鲁棒性

---

## 常见问题

### Q1: ImportError: cannot import name 'download' from 'aistudio_sdk.hub'

**解决方案：**
```bash
pip install --upgrade aistudio_sdk paddlenlp
```

### Q2: CUDA out of memory

**解决方案：**
```bash
--batch_size 16 --max_length 64
```

### Q3: 模型下载失败（404错误）

使用已验证的模型名称：
- `bert-large-uncased`（推荐）
- `bert-base-uncased`
- `roberta-base`

---

## 参考文献

1. Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL-HLT, 2019.
2. Goodfellow, I., et al. "Explaining and Harnessing Adversarial Examples." ICLR, 2015.
3. Hinton, G., et al. "Distilling the Knowledge in a Neural Network." NIPS Deep Learning Workshop, 2015.

---

## License

MIT License

---

*本项目完全遵循 SCNU AI Competition 2026 的评价标准，致力于在少样本环境下实现最优的表格关系提取性能。*
