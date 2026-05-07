# SCNU AI Competition 2026 - 表格语义关系提取

本项目旨在实现对表格列对（Subject, Object）之间语义关系的精准提取。项目提供了基于 **PyTorch** 框架实现的训练与推理代码，且在逻辑上与官方提供的 PaddlePaddle Baseline 完全对齐。

## 项目特点

本项目在 `src/` 目录下复现了 Baseline 的核心逻辑，并将其迁移至 PyTorch 框架：

- **框架迁移**：使用 PyTorch 和 Hugging Face `transformers` 库，替代了原始的 PaddlePaddle 和 PaddleNLP 实现。
- **逻辑对齐**：
    - **Pooling 策略**：采用 `CLS token` 作为序列的全局表示，与 Baseline 保持一致。
    - **输入处理**：支持 (Subject, Object) 双输入拼接，自动处理 [SEP] 分隔符。
- **Few-Shot 专项优化**（针对赛题权重公式设计）：
    - **均衡采样 (WeightedRandomSampler)**：动态调整采样概率，大幅提升长尾分布中稀有类别的训练频率。
    - **Focal Loss**：引入焦点损失函数，降低易分类样本权重，强制模型关注难以识别的少样本关系。
    - **R-Drop 正则化**：通过双向 KL 散度约束，显著增强模型在低资源样本下的鲁棒性与泛化能力。
    - **加权损失 (Weighted Loss)**：直接将赛题评分公式中的重要性权重对齐到损失函数中。
    - **数据增强**：集成随机删除、词序扰动等文本增强技术，扩充稀有类别的语义空间。
- **训练优化**：
    - **混合精度训练 (AMP)**：支持自动混合精度训练，显著提升训练速度并降低显存占用。
    - **学习率调度**：集成 Linear Decay with Warmup 调度器，优化模型收敛过程。
    - **标签平滑 (Label Smoothing)**：防止少样本场景下的模型过拟合。
    - **早停机制**：内置 Early Stopping，防止模型过拟合。

## 数据集分析

训练集包含约 38699 个样本，563 个类别，分布极不均衡：
- **头部效应**：最多样本类别 "instance of"（4681 样本）。
- **长尾分布**：约 37% 的类别样本量不足 10 条，16% 的类别不足 3 条。
- **极端少样本**：部分类别仅有 1 个样本。
- **评分偏向**：赛题评分公式 $Score_{final}$ 赋予了这些少样本关系极高的权重，是提升排名的关键。

## 运行指南

### 1. 环境依赖
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- pandas, scikit-learn, tqdm

### 2. 环境配置
使用虚拟环境配置依赖：
```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境 (Windows)
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 模型训练

使用 `src/train_torch.py` 进行模型训练。建议开启针对 Few-shot 的优化参数：

```bash
# 推荐的最强优化配置
python src/train_torch.py `
    --use_balanced_sampler `
    --use_focal_loss `
    --use_rdrop `
    --use_augmentation `
    --use_weighted_loss `
    --use_amp `
    --batch_size 32 `
    --epoch 20
```

主要参数说明：
- `--use_balanced_sampler`: 开启均衡采样，处理长尾分布。
- `--use_focal_loss`: 开启 Focal Loss，关注困难/稀有样本。
- `--use_rdrop`: 开启 R-Drop，增强少样本泛化性。
- `--use_augmentation`: 开启在线文本增强。
- `--use_weighted_loss`: 使用赛题公式对齐的加权损失。
- `--label_smoothing`: 标签平滑系数 (默认 0.1)。
- `--train_dir`: 训练数据集目录。
- `--device`: 设备选择 (cuda/cpu)。
- `--use_amp`: 开启混合精度训练。
- `--patience`: 早停 patience (默认 3)。

### 4. 模型推理

使用 `src/infer_torch.py` 进行推理并生成提交文件：

```bash
python src/infer_torch.py --model_path ./cpa_output/cpa_TIMESTAMP/best_model.pth --labels_path ./cpa_output/cpa_TIMESTAMP/label_classes.txt --input_csv ./dataset/test.csv --output_file ./result/submission.csv
```

## 目录结构
```text
SCNU_AI_Competition_2026/
├── baseline/               # 官方原始 PaddlePaddle Baseline 代码
├── dataset/                # 数据存放 (Train_Set/ , test.csv)
├── src/                    # PyTorch 实现代码
│   ├── train_torch.py      # 训练脚本 (与 Baseline 逻辑对齐)
│   ├── infer_torch.py      # 推理脚本 (与 Baseline 逻辑对齐)
│   └── preprocess_data.py  # 数据预处理辅助脚本
├── cpa_output/             # 模型输出 (日志、权重、标签表)
├── result/                 # 预测结果 (submission.csv)
└── requirements.txt        # 项目依赖
```

---
*本项目提供的 PyTorch 实现版本完全遵循 SCNU AI Competition 2026 的评价标准，确保了与 Baseline 的一致性与可复现性。*
