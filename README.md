# SCNU AI Competition 2026 - 表格语义关系提取

本项目旨在实现对表格列对（Subject, Object）之间语义关系的精准提取。模型基于 **PyTorch** 框架开发，针对赛题的长尾分布和少样本（Few-Shot）评估标准进行了专项优化。

## 核心优化方案

我们通过在 `src/` 目录下复现的最优版本，实现了以下核心技术点：

- **Mean-Pooling (平均池化)**：通过对 Transformer 输出的序列向量进行按掩码平均，相比传统的 CLS 向量，能够更稳健地表征表格单元格的短文本语义。
- **FGM 对抗训练 (Adversarial Training)**：在 Embedding 层通过添加微小扰动进行对抗攻击（ε=1.0），显著提升了模型的泛化能力和对噪声的抵抗力。
- **动态加权损失函数 (Weighted CrossEntropy)**：严格遵循赛题 `question.md` 中的 $m_{weights}$ 公式，根据样本量自动调整 Loss 权重，使模型在评估高分值的少样本类别时具有更高的准确度。
- **Score_final 指标对齐**：验证环节直接采用赛题要求的加权评分标准进行模型筛选，确保训练产出的 `best_model` 即为评分最高版本。

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

### 3. 数据预处理（可选，推荐）
先运行预处理脚本生成缓存，加速后续训练：
```bash
python src/preprocess_data.py --max_length 64
```
预处理后的缓存保存在 `./dataset/cache/` 目录下。如需强制重建缓存：
```bash
python src/preprocess_data.py --max_length 64 --force_rebuild
```

### 4. 模型训练
使用 `src/train_torch.py` 启动训练。默认使用 GPU 运行，并开启 AMP 混合精度加速。
```bash
python src/train_torch.py --device cuda --batch_size 32 --epoch 10 --max_length 64 --use_amp
```
训练产出的模型将保存在 `./cpa_output/cpa_YYYYMMDD_HHMMSS/` 目录下。

主要参数说明：
- `--device`: 设备选择 (cuda/cpu)
- `--batch_size`: 批次大小
- `--epoch`: 训练轮数
- `--max_length`: 序列最大长度
- `--use_amp`: 开启混合精度训练（推荐）
- `--num_workers`: 数据加载进程数（默认 0，表示主进程加载）
- `--force_rebuild_cache`: 强制重建数据缓存

### 5. 模型推理
推理时需指定训练产出的路径和标签文件：
```bash
python src/infer_torch.py \
  --model_path ./cpa_output/cpa_TIMESTAMP/best_model.pth \
  --labels_path ./cpa_output/cpa_TIMESTAMP/label_classes.txt \
  --input_csv ./dataset/test.csv \
  --output_file ./result/submission.csv
```

## 目录结构
```text
SCNU_AI_Competition_2026/
├── baseline/               # 原始 Baseline 代码
├── dataset/                # 数据存放 (Train_Set/ , test.csv)
│   └── cache/              # 预处理数据缓存（自动生成）
├── src/                    # 核心优化代码
│   ├── preprocess_data.py  # 数据预处理脚本
│   ├── train_torch.py      # 训练脚本 (Mean-Pooling + FGM + Weighted Loss)
│   └── infer_torch.py      # 推理脚本
├── cpa_output/             # 模型输出 (日志、权重、标签表)
├── result/                 # 预测结果 (submission.csv)
└── requirements.txt        # 项目依赖
```

---
*本项目完全遵循 SCNU AI Competition 2026 的评价指标体系，旨在提供一个高性能且易于复现的解决方案。*
