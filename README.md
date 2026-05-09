# SCNU AI Competition 2026 - 表格语义关系提取

本项目旨在实现对表格列对（Subject, Object）之间语义关系的精准提取。基于 **PyTorch** 和 **Hugging Face Transformers** 实现，并针对赛题的少样本（Few-Shot）特点和加权评分规则进行了深度优化。

## 项目特点

本项目针对赛题的 **Few-Shot** 和 **加权评分 ($Score_{final}$)** 特点进行了核心优化：

- **中毒数据检测与剔除**: 自动检测并移除训练集中的中毒数据（高重复率样本），提升数据质量。
- **加权损失函数 (Weighted CE)**: 严格按照赛题公式计算类别权重，直接优化最终评分指标。
- **加权采样 (Weighted Sampling)**: 使用加权随机采样处理类别不平衡问题。
- **对抗训练 (FGM)**: 在Embedding层注入对抗扰动，增强模型泛化性。
- **混合精度训练**: 支持自动混合精度训练，加速训练并节省显存。

## 推荐预训练模型

以下模型在本任务上经过验证表现优异：

| 模型名称 | 推荐指数 | 说明 |
|----------|----------|------|
| `microsoft/deberta-v3-base` | ⭐⭐⭐⭐⭐ | **推荐**，在表格语义提取任务上表现优异 |
| `microsoft/deberta-v3-large` | ⭐⭐⭐⭐ | 更大模型，效果更好但需要更多显存 |
| `roberta-large` | ⭐⭐⭐⭐ | 稳定可靠，适合通用场景 |
| `xlm-roberta-large` | ⭐⭐⭐⭐ | 多语言模型，适合跨语言场景 |

## 运行指南

### 1. 环境配置
建议使用 Python 3.10+ 环境。

```bash
# 创建并激活虚拟环境
python -m venv venv
.\venv\Scripts\activate

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers pandas numpy scikit-learn tqdm
```

### 2. 数据预处理（可选但推荐）

在训练前进行数据预处理，检测并剔除中毒数据：

```bash
python src/data_preprocess.py \
    --train_dir ./dataset/Train_Set \
    --output_dir ./dataset \
    --poison_threshold 0.95
```

**预处理脚本功能：**
- 检测并移除中毒数据（高重复率的Subject/Object对）
- 检测并移除异常样本（基于文本长度的Z-score检测）
- 分析数据分布并输出统计信息
- 计算并保存类别权重文件

### 3. 模型训练

使用PyTorch训练脚本进行训练：

```bash
# 使用 DeBERTa-v3（推荐）
python src/train_torch.py \
    --train_dir ./dataset/Train_Set \
    --output_dir ./cpa_output \
    --model_name microsoft/deberta-v3-base \
    --batch_size 16 \
    --epochs 10 \
    --lr 2e-5 \
    --weighted_sampling \
    --use_fgm \
    --num_workers 4
```

#### 主要参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train_dir` | `./dataset/Train_Set` | 训练数据目录 |
| `--output_dir` | `./cpa_output` | 模型输出目录 |
| `--model_name` | `microsoft/deberta-v3-base` | 预训练模型名称 |
| `--batch_size` | 32 | 批处理大小 |
| `--epochs` | 10 | 训练轮数 |
| `--lr` | 2e-5 | 学习率 |
| `--max_length` | 128 | 最大序列长度 |
| `--val_ratio` | 0.1 | 验证集比例 |
| `--patience` | 3 | 早停耐心值 |
| `--weighted_sampling` | True | 使用加权随机采样 |
| `--use_fgm` | False | 使用FGM对抗训练 |
| `--num_workers` | 4 | DataLoader进程数 |

#### 不同模型的推荐配置

```bash
# DeBERTa-v3-base（平衡效果和速度）
python src/train_torch.py --model_name microsoft/deberta-v3-base --batch_size 16

# DeBERTa-v3-large（追求极致效果，需要更多显存）
python src/train_torch.py --model_name microsoft/deberta-v3-large --batch_size 8

# RoBERTa-large（稳定可靠）
python src/train_torch.py --model_name roberta-large --batch_size 16

# XLM-RoBERTa-large（多语言场景）
python src/train_torch.py --model_name xlm-roberta-large --batch_size 8
```

### 4. 模型推理

使用推理脚本进行预测：

```bash
python src/infer_torch.py \
    --input_csv ./dataset/test.csv \
    --labels_path ./cpa_output/cpa_<timestamp>/label_classes.txt \
    --model_path ./cpa_output/cpa_<timestamp>/best_model.pth \
    --output_file submission.csv \
    --model_name microsoft/deberta-v3-base \
    --batch_size 64
```

### 5. 性能优化建议

#### 加速训练速度
```bash
# 增加 DataLoader 进程数
--num_workers 8

# 调整学习率和 epoch 数
--lr 3e-5 --epochs 8
```

#### 显存优化
对于显存较小的 GPU（< 8GB）：
```bash
--batch_size 8 --max_length 64
```

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
├── baseline/               # 官方 PaddlePaddle Baseline（保留）
│   ├── infer.py
│   └── train.py
├── dataset/                # 数据存放目录
│   ├── Train_Set/          # 训练集（CSV文件）
│   ├── test.csv            # 测试集
│   ├── cleaned_train_data.csv  # 清理后的数据（可选）
│   └── label_weights.csv   # 类别权重文件（可选）
├── src/                    # PyTorch核心代码
│   ├── data_preprocess.py  # 数据预处理脚本
│   ├── train_torch.py      # PyTorch训练脚本
│   ├── infer_torch.py      # PyTorch推理脚本
│   ├── train.py            # 旧版PaddlePaddle训练脚本
│   ├── train_improved.py   # 旧版改进训练脚本
│   └── infer.py            # 旧版PaddlePaddle推理脚本
├── cpa_output/             # 模型输出目录（自动生成）
│   └── cpa_<timestamp>/    # 时间戳命名的模型文件夹
│       ├── best_model.pth      # 最佳模型权重
│       ├── label_classes.txt   # 标签类别文件
│       └── train.log           # 训练日志
├── result/                 # 预测结果
├── .gitignore
├── README.md
├── question.md             # 赛题说明
└── requirements.txt        # 项目依赖
```

## 常见问题

### Q1: CUDA out of memory

**解决方案：**
```bash
--batch_size 8 --max_length 64
```

### Q2: 模型下载速度慢

**解决方案：**
```bash
# 设置代理或使用国内镜像
export TRANSFORMERS_OFFLINE=1  # 如果已下载模型
```

### Q3: 训练集数据不平衡

本项目已内置加权损失函数和加权采样机制，自动处理类别不平衡问题。

---

*本项目完全遵循 SCNU AI Competition 2026 的评价标准，致力于在少样本环境下实现最优的表格关系提取性能。*