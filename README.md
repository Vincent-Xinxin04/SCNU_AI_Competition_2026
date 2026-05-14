# SCNU AI Competition 2026 - 表格语义关系提取

本项目旨在实现对表格列对（Subject, Object）之间语义关系的精准提取。基于 **PaddlePaddle** 和 **PaddleNLP** 实现，并针对赛题的少样本（Few-Shot）特点和加权评分规则进行了深度优化。

## 项目特点

本项目针对赛题的 **Few-Shot** 和 **加权评分 ($Score_{final}$)** 特点进行了核心优化：

- **多种 Pooling 策略**: 支持 CLS、Mean、Max、CLS+Mean、CLS+Max、CLS+Mean+Max 等多种特征聚合方式
- **加权损失函数 (Weighted CE)**: 严格按照赛题公式计算类别权重，直接优化最终评分指标
- **加权采样 (Weighted Sampling)**: 使用加权随机采样处理类别不平衡问题
- **对抗训练 (FGM)**: 在 Embedding 层注入对抗扰动，增强模型泛化性
- **EMA (指数移动平均)**: 提升模型稳定性和泛化能力
- **学习率调度器**: 支持 Linear Decay 和 Cosine Annealing 两种调度策略
- **数据增强**: 随机插入空格和标点，增强模型鲁棒性
- **混合精度训练**: 支持自动混合精度训练，加速训练并节省显存

## 百度 AI Studio 环境配置

### 推荐框架版本

| 配置项 | 推荐选择 |
|--------|----------|
| **框架** | PaddlePaddle 2.6.x |
| **Python** | 3.10 |
| **GPU** | V100 32GB / A100 40GB（推荐） |
| **镜像** | AI Studio 默认镜像（已预装 PaddlePaddle） |

> **注意**: AI Studio 环境已预装 PaddlePaddle 和 PaddleNLP，无需手动安装。如遇 `aistudio_sdk` 报错，请参考下方常见问题。

### 可用预训练模型（PaddleNLP）

| 模型名称 | 推荐指数 | 说明 |
|----------|----------|------|
| `bert-large-uncased` | ⭐⭐⭐⭐⭐ | **推荐**，稳定可靠，英文数据效果好 |
| `roberta-base` | ⭐⭐⭐⭐ | RoBERTa 基础模型 |
| `bert-base-uncased` | ⭐⭐⭐⭐ | 基础模型，显存占用低 |


> **注意**: 本项目处理的是英文数据，请使用英文预训练模型。

## 运行指南

### 1. 环境配置

#### AI Studio 环境（推荐）
AI Studio 已预装 PaddlePaddle，无需额外安装。如需手动安装依赖：

```bash
pip install -r requirements.txt
```

#### 本地环境
```bash
# 安装 PaddlePaddle GPU 版本（CUDA 11.8）
python -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple

# 安装 PaddleNLP
pip install paddlenlp>=2.6.0

# 安装其他依赖
pip install pandas numpy scikit-learn tqdm
```

### 2. 模型训练

使用 PaddlePaddle 训练脚本进行训练：

```bash
# 基础训练（推荐配置，使用 BERT Large）
python src/train_paddle.py \
    --train_dir ./dataset/Train_Set \
    --output_dir ./cpa_output \
    --shortcut_name bert-large-uncased \
    --pooling_strategy cls_mean_max \
    --batch_size 32 \
    --epochs 20

# 开启 FGM 对抗训练
python src/train_paddle.py \
    --train_dir ./dataset/Train_Set \
    --output_dir ./cpa_output \
    --pooling_strategy cls_mean \
    --use_fgm \
    --epochs 15

# 使用 Cosine 学习率调度
python src/train_paddle.py \
    --train_dir ./dataset/Train_Set \
    --output_dir ./cpa_output \
    --pooling_strategy cls_mean \
    --lr_scheduler cosine \
    --epochs 20

# 使用 EMA
python src/train_paddle.py \
    --train_dir ./dataset/Train_Set \
    --output_dir ./cpa_output \
    --pooling_strategy cls_mean \
    --use_ema \
    --epochs 15

# 使用 Iluvatar GPU 训练（完整优化配置，使用 BERT Large）
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
    --weight_decay 1e-4 \
    --device iluvatar_gpu

# 使用 XPU 训练
python src/train_paddle.py \
    --train_dir ./dataset/Train_Set \
    --output_dir ./cpa_output \
    --shortcut_name bert-large-uncased \
    --pooling_strategy cls_mean_max \
    --batch_size 32 \
    --use_fgm \
    --use_multi_head \
    --use_amp \
    --device xpu
```

#### 主要参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train_dir` | `./dataset/Train_Set` | 训练数据目录 |
| `--output_dir` | `./cpa_output` | 模型输出目录 |
| `--deduplicate` | True | 是否去除重复样本 |
| `--shortcut_name` | `bert-large-uncased` | PaddleNLP 预训练模型名称 |
| `--max_length` | 128 | 最大序列长度 |
| `--dropout_rate` | 0.1 | Dropout 率 |
| `--pooling_strategy` | `cls_mean_max` | Pooling 策略（cls/mean/max/cls_mean/cls_max/cls_mean_max） |
| `--batch_size` | 32 | 批处理大小 |
| `--epochs` | 20 | 训练轮数 |
| `--lr` | 3e-5 | 学习率 |
| `--weight_decay` | 1e-4 | 权重衰减 |
| `--warmup_ratio` | 0.1 | 学习率预热比例 |
| `--patience` | 5 | 早停耐心值 |
| `--val_ratio` | 0.1 | 验证集比例 |
| `--random_seed` | 42 | 随机种子 |
| `--num_workers` | 0 | 数据加载线程数 |
| `--rare_threshold` | 5 | 稀有类别阈值（样本数少于此值的类别全部放入训练集） |
| `--weighted_sampling` | True | 使用加权随机采样（处理类别不平衡） |
| `--data_augment` | True | 启用数据增强 |
| `--use_fgm` | True | 使用 FGM 对抗训练 |
| `--fgm_epsilon` | 1.0 | FGM 攻击的 epsilon 值 |
| `--use_ema` | True | 使用指数移动平均 |
| `--ema_decay` | 0.999 | EMA 衰减率 |
| `--use_multi_head` | False | 使用多头注意力进行特征融合 |
| `--use_amp` | False | 使用混合精度训练 |
| `--lr_scheduler` | `cosine` | 学习率调度器（linear/cosine） |
| `--device` | auto | 设备类型 (auto/cpu/gpu/xpu/iluvatar_gpu) |

### 3. 模型推理

使用推理脚本进行预测：

```bash
python src/infer_paddle.py \
    --input_csv ./dataset/test.csv \
    --labels_path ./cpa_output/cpa_<timestamp>/label_classes.txt \
    --model_path ./cpa_output/cpa_<timestamp>/best_model.pdparams \
    --output_file submission.csv \
    --shortcut_name bert-large-uncased \
    --pooling_strategy cls_mean_max \
    --use_multi_head \
    --batch_size 64 \
    --device gpu
```

> **注意**: 推理时的 `--pooling_strategy` 参数必须与训练时保持一致！

#### 推理参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_csv` | `./dataset/test.csv` | 测试集 CSV 文件路径 |
| `--labels_path` | (必需) | 标签类别文件路径 |
| `--model_path` | (必需) | 训练好的模型权重路径 |
| `--output_file` | `./submission.csv` | 预测结果输出路径 |
| `--shortcut_name` | `bert-large-uncased` | 预训练模型名称（需与训练一致） |
| `--pooling_strategy` | `cls_mean_max` | Pooling 策略（需与训练一致） |
| `--use_multi_head` | False | 使用多头注意力（需与训练一致） |
| `--batch_size` | 64 | 批处理大小 |
| `--max_length` | 128 | 最大序列长度（需与训练一致） |
| `--num_workers` | 0 | 数据加载线程数 |
| `--use_amp` | False | 使用混合精度推理 |
| `--device` | auto | 设备类型 |

## 评价指标

本项目严格遵循赛题的加权评分规则：

**少样本重要性权重计算公式：**
$$m_{weights} = \frac{counts_{max} - counts_{m} + counts_{min} \times 0.1}{counts_{max} + counts_{min} \times 0.1}$$

**最终分数计算公式：**
$$Score_{final} = \frac{\sum_{m} (m_{weights} \times m_{score})}{\sum_{m} m_{weights}}$$

其中 $m_{score} = \frac{m_{correct}}{m_{total}}$ 为关系类型 $m$ 的准确率。

## 目录结构

```text
SCNU_AI_Competition_2026/
├── baseline/               # 官方 PaddlePaddle Baseline
│   ├── infer.py
│   └── train.py
├── dataset/                # 数据存放目录
│   ├── Train_Set/          # 训练集（CSV文件）
│   └── test.csv            # 测试集
├── src/                    # 核心代码
│   ├── train_paddle.py     # PaddlePaddle 训练脚本（改进版）
│   └── infer_paddle.py     # PaddlePaddle 推理脚本（改进版）
├── cpa_output/             # 模型输出目录（自动生成）
│   └── cpa_<timestamp>/    # 时间戳命名的模型文件夹
│       ├── best_model.pdparams   # 最佳模型权重
│       └── label_classes.txt     # 标签类别文件
├── result/                 # 预测结果示例
├── .gitignore
├── README.md
├── question.md             # 赛题说明
└── requirements.txt        # 项目依赖
```

## 常见问题

### Q1: ImportError: cannot import name 'download' from 'aistudio_sdk.hub'

**解决方案：**
```bash
# 方法1：升级依赖（推荐）
pip install --upgrade aistudio_sdk paddlenlp

# 方法2：降级 paddlenlp
pip install paddlenlp==2.6.0

# 方法3：设置环境变量跳过
export PADDLE_NLP_DISABLE_AISTUDIO=True
```

### Q2: CUDA out of memory

**解决方案：**
```bash
--batch_size 8 --max_length 64
```

### Q3: 模型下载失败（404错误）

部分模型在 PaddleNLP 社区不可用，请使用已验证的模型名称：
- `bert-large-uncased`（推荐）
- `bert-base-uncased`
- `roberta-base`

### Q4: 训练集数据不平衡

本项目已内置加权损失函数和加权采样机制，自动处理类别不平衡问题。

---

*本项目完全遵循 SCNU AI Competition 2026 的评价标准，致力于在少样本环境下实现最优的表格关系提取性能。*
