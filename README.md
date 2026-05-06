# SCNU AI Competition 2026 - 表格语义关系提取

本项目旨在实现对表格列对（Subject, Object）之间语义关系的精准提取。模型基于 **PaddlePaddle** 框架开发，针对赛题的长尾分布和少样本（Few-Shot）评估标准进行了专项优化。

## 核心优化方案

我们通过在 `src/` 目录下复现的最优版本，实现了以下核心技术点：

- **Mean-Pooling (平均池化)**：通过对 Transformer 输出的序列向量进行按掩码平均，相比传统的 CLS 向量，能够更稳健地表征表格单元格的短文本语义。
- **FGM 对抗训练 (Adversarial Training)**：在 Embedding 层通过添加微小扰动进行对抗攻击（ε=1.0），显著提升了模型的泛化能力和对噪声的抵抗力。
- **动态加权损失函数 (Weighted CrossEntropy)**：严格遵循赛题 `question.md` 中的 $m_{weights}$ 公式，根据样本量自动调整 Loss 权重，使模型在评估高分值的少样本类别时具有更高的准确度。
- **Score_final 指标对齐**：验证环节直接采用赛题要求的加权评分标准进行模型筛选，确保训练产出的 `best_model` 即为评分最高版本。

## 运行指南

### 1. 环境依赖
- Python 3.8+
- PaddlePaddle 2.x
- PaddleNLP
- pandas, scikit-learn, tqdm

### 2. 模型训练
使用 `src/train_paddle_optm.py` 启动训练。默认使用 GPU 运行，并开启 AMP 混合精度加速。
```bash
python src/train_paddle_optm.py --device gpu --batch_size 32 --epoch 10 --max_length 64
```
训练产出的模型将保存在 `./cpa_output/cpa_YYYYMMDD_HHMMSS/` 目录下。

### 3. 模型推理
推理时需指定训练产出的路径和标签文件：
```bash
python src/infer_paddle_optm.py \
  --model_path ./cpa_output/cpa_TIMESTAMP/best_model.pdparams \
  --labels_path ./cpa_output/cpa_TIMESTAMP/label_classes.txt \
  --input_csv ./dataset/test.csv \
  --output_file ./result/submission.csv
```

## 目录结构
```text
SCNU_AI_Competition_2026/
├── baseline/               # 原始 Baseline 代码
├── dataset/                # 数据存放 (Train_Set/ , test.csv)
├── src/                    # 核心优化代码
│   ├── train_paddle_optm.py # 训练脚本 (Mean-Pooling + FGM + Weighted Loss)
│   └── infer_paddle_optm.py # 推理脚本
├── cpa_output/             # 模型输出 (日志、权重、标签表)
└── result/                 # 预测结果 (submission.csv)
```

---
*本项目完全遵循 SCNU AI Competition 2026 的评价指标体系，旨在提供一个高性能且易于复现的解决方案。*
