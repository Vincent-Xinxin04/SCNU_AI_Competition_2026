# 表格语义关系提取 (Table Semantic Relationship Extraction)

本项目旨在实现对表格列对（Subject, Object）之间语义关系的精准提取。针对少样本（Few-Shot）场景和计算资源的限制，我们设计了使用预训练 Transformer 模型、带有 Attention Pooling 和 Multi-Sample Dropout 的稳健模型架构，并严格遵循了赛题评价指标的要求。

## 赛题背景

精准完成表格语义关系提取是赋能数据集成、数据清洗与知识发现等下游任务的关键钥匙。针对真实场景下标定数据获取困难的痛点，研究如何在少样本（Few-Shot）条件下高效实现表格语义关系提取任务，不仅能大幅降低标注成本，更是提升现代数据流水线智能化水平的核心突破口。

## 任务说明

给定一组表格、一个候选关系集合以及目标列对，要求通过表格的数据从候选关系集合中去预测给定目标列对之间的关系类型，表示这两列之间的关系。

- **输入**：目标列对（Subject，Object）、候选关系集合
- **输出**：Subject，Object，预测的关系类型

## 数据集说明

比赛数据集划分为训练集和测试集，训练集共38699条数据，测试集共4068条数据。本次数据集包含563个候选关系，每个表格中包含两列数据。

| 数据集 | 候选关系数量 | 数据总量 |
|--------|--------------|----------|
| 训练集 | 563          | 38699    |
| 测试集 | 563          | 4068     |

### 输入输出样例

**Input**

示例表格 table1：

| Subject                | Object               |
|------------------------|----------------------|
| Jobcenter              | Enzo Cormann         |
| The Boy Who Cried Wolf | Gosho Aoyama         |
| Iter Italicum          | Paul Oskar Kristeller|

候选关系集合：`{area, author, brand, date of birth}`

目标列对：`{Subject, Object}`

**Output**

| Subject                | Object               | Label  |
|------------------------|----------------------|--------|
| Jobcenter              | Enzo Cormann         | author |
| The Boy Who Cried Wolf | Gosho Aoyama         | author |
| Iter Italicum          | Paul Oskar Kristeller| author |

## 评价方式

本比赛采用一种结合少样本重要性和预测准确性的评价方式，具体而言，将少样本重要性与关系类型的样本数量进行关联，某关系类型的样本数据越少表示其少样本重要性越高并赋予更高权重。

### 少样本重要性权重

少样本重要性权重计算公式如下：

$$
m\_weights = \frac{counts_{max} - counts_{m} + counts_{min} \times 0.1}{counts_{max} + counts_{min} \times 0.1}
$$

其中：
- $counts_{max}$：样本数量最多的关系类型的样本个数
- $counts_{min}$：样本数量最少的关系类型的样本个数
- $counts_{m}$：当前关系类型的样本个数
- $m\_weights$：当前关系类型的少样本重要性权重

为保证每个少样本重要性权重均大于 0，设置约束项 $counts_{min} \times 0.1$ 保证权重有效性。

### 准确性计算规则

关系类型 $m$ 的准确性分数计算公式：

$$
m\_score = \frac{m\_correct}{m\_total}
$$

其中：
- $m\_score$：关系类型 $m$ 的准确性分数
- $m\_correct$：关系类型 $m$ 中预测正确的数量
- $m\_total$：关系类型 $m$ 中需要被正确预测的全部数量

### 最终分数计算规则

最终分数计算公式：

$$
Score\_final = \frac{\sum_{m} (m\_weights \times m\_score)}{\sum_{m} m\_weights}
$$

## 项目亮点

- **全量 GPU 加速训练**：采用 PyTorch 和 CUDA 实现并开启混合精度训练 (AMP)，最大化利用 RTX 4060 GPU。
- **多语言底座 (XLM-RoBERTa)**：原生支持 100 多种语言，无缝处理中文、各类小语种及特殊字符实体。
- **Attention Pooling**：使用自注意力机制进行池化操作，精准提取关键特征。
- **双重对抗与一致性正则 (FGM + R-Drop)**：针对极端长尾分布（最少样本仅为 1）加入词向量微小扰动 (FGM) 和双路径 Dropout 概率一致性对齐 (R-Drop)，防止在少样本关系上严重过拟合。
- **严格指标对齐与标签平滑**：完全依照 `question.md` 使用 $m\_weights$ 加权，同时融合 Label Smoothing=0.1 抑制模型在长尾分布下的过度自信。

## 环境配置

本项目需要使用 Python 虚拟环境以隔离依赖：

1. **激活虚拟环境** (推荐在根目录下)：
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

## 运行指南

### 1. 训练模型

请在虚拟环境下运行以下命令启动训练。模型将自动加载 `dataset/Train_Set/` 下的训练数据进行权重计算和训练。

```powershell
python src/train.py --batch_size 32 --epochs 15 --model_name xlm-roberta-base
```

*提示：开启 R-Drop 和 FGM 后计算量会翻倍，因此默认 batch_size 调整为了 32 以适应显存。*

### 2. 模型推理

训练完成后，你可以使用训练好的权重在测试集上进行预测：

```powershell
python src/infer.py --batch_size 128 --checkpoint model_epoch_15.pt --model_name xlm-roberta-base
```

这将在根目录下生成 `submission.csv` 文件，其内容格式为：`Subject, Object, Label`。

## 目录结构

```text
f:\SCNU_AI_Competition\
│  question.md          # 赛题说明，包含完整的评价指标定义
│  README.md            # 项目说明文档
│
├─dataset\
│  ├─Train_Set\         # 训练集，包含 563 个 CSV 文件，每个文件对应一种关系
│  │   ├─ author.csv    # 作者关系示例
│  │   ├─ area.csv      # 面积关系示例
│  │   └─ ...           # 其他 561 种关系
│  └─test.csv           # 待测试数据，包含 Subject, Object 列
│
├─baseline\
│  ├─train.py           # 基于 PaddlePaddle 的训练脚本
│  └─infer.py           # 基于 PaddlePaddle 的推理脚本
│
└─venv\                 # Python 虚拟环境
```

## 数据集结构详解

### 训练集结构

训练集位于 `dataset/Train_Set/` 目录下，每个 CSV 文件以关系名称命名，文件内容包含两列：
- **Subject**：主语列，包含实体或概念
- **Object**：宾语列，包含与主语相关的值

例如 `author.csv` 包含作者关系数据，`area.csv` 包含面积关系数据。

### 测试集结构

`test.csv` 文件包含待预测的数据，仅有 Subject 和 Object 两列，需要预测 Label 列。

## 模型架构

本项目采用的模型架构包括：

1. **预训练语言模型**：XLM-RoBERTa-base 作为编码器底座
2. **Attention Pooling**：使用自注意力机制聚合序列表示
3. **Multi-Sample Dropout**：在训练时使用多个 Dropout 样本增强泛化能力
4. **FGM 对抗训练**：对词向量添加小扰动，提升模型鲁棒性
5. **R-Drop 一致性正则**：强制两个带不同 Dropout 的前向传播输出保持一致

## 依赖环境

主要依赖包括：
- Python 3.8+
- PyTorch 2.0+
- transformers
- pandas
- numpy
- scikit-learn
