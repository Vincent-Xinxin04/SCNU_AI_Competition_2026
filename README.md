# SCNU AI Competition 2026 - 表格语义关系提取

本项目旨在实现对表格列对（Subject, Object）之间语义关系的精准提取。项目提供了基于 **PyTorch** 框架实现的训练与推理代码，且在逻辑上与官方提供的 PaddlePaddle Baseline 完全对齐。

## 项目特点

本项目在 `src/` 目录下复现了 Baseline 的核心逻辑，并将其迁移至 PyTorch 框架：

- **框架迁移**：使用 PyTorch 和 Hugging Face `transformers` 库，替代了原始的 PaddlePaddle 和 PaddleNLP 实现。
- **逻辑对齐**：
    - **Pooling 策略**：采用 `CLS token` 作为序列的全局表示，与 Baseline 保持一致。
    - **损失函数**：使用标准交叉熵损失（`CrossEntropyLoss`）。
    - **评估指标**：使用标准准确率（Accuracy）进行模型评估与筛选。
    - **输入处理**：支持 (Subject, Object) 双输入拼接，自动处理 [SEP] 分隔符。
- **训练优化**：
    - **混合精度训练 (AMP)**：支持自动混合精度训练，显著提升训练速度并降低显存占用。
    - **学习率调度**：集成 Linear Decay with Warmup 调度器，优化模型收敛过程。
    - **早停机制**：内置 Early Stopping，防止模型过拟合。

## 数据集分析

训练集包含 39233 个样本，563 个类别，分布极不均衡：
- 最多样本类别："instance of"（4681 样本）
- 最少样本类别：仅 1 个样本
- 平均每类样本：约 69.7 个

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

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 模型训练

使用 `src/train_torch.py` 进行模型训练。该脚本已与 Baseline 对齐：

```bash
python src/train_torch.py --device cuda --batch_size 32 --epoch 20 --max_length 128 --use_amp --shortcut_name bert-base-uncased
```

训练产出的模型将保存在 `./cpa_output/cpa_YYYYMMDD_HHMMSS/` 目录下。

主要参数说明：
- `--train_dir`: 训练数据集目录
- `--device`: 设备选择 (cuda/cpu)
- `--batch_size`: 批次大小 (默认 32)
- `--epoch`: 训练轮数 (默认 20)
- `--lr`: 学习率 (默认 5e-5)
- `--max_length`: 序列最大长度 (默认 128)
- `--use_amp`: 开启混合精度训练
- `--patience`: 早停 patience (默认 3)

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
