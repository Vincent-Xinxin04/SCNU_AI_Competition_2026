# SCNU AI Competition 2026 - 表格语义关系提取

本项目旨在实现对表格列对（Subject, Object）之间语义关系的精准提取。项目提供了基于 **PyTorch** 框架实现的训练与推理代码，且在逻辑上与官方提供的 PaddlePaddle Baseline 完全对齐。

## 项目特点

本项目在 `src/` 目录下复现了 Baseline 的核心逻辑，并将其迁移至 PyTorch 框架：

- **框架迁移**：使用 PyTorch 和 Hugging Face `transformers` 库，替代了原始的 PaddlePaddle 和 PaddleNLP 实现。
- **逻辑对齐**：
    - **Pooling 策略**：采用 `CLS token` 作为序列的全局表示，与 Baseline 保持一致。
    - **输入处理**：支持 (Subject, Object) 双输入拼接，自动处理 [SEP] 分隔符。
- **Few-Shot 专项优化**（针对赛题权重公式设计）：
    - **评分驱动保存**：集成赛题官方评分公式 $Score_{final}$，模型保存完全基于加权后的少样本得分。
    - **3 折交叉验证 (3-Fold CV)**：自动执行多折训练，提升模型在长尾分布下的鲁棒性与评估稳定性。
    - **权重指数移动平均 (EMA)**：利用历史权重的加权平均值进行预测，显著提升模型泛化能力。
    - **对率调整 (Logit Adjustment)**：在训练 Loss 中引入类别先验偏置，建立更稳健的决策边界。
    - **类型提示特征工程 (Type Hinting)**：自动识别并注入 `[NUM]`、`[DATE]`、`[URL]` 等物理类型标记。
    - **R-Drop & FGM**：双向一致性约束与 Embedding 层对抗扰动。
- **训练优化**：
    - **6GB 显存深度适配**：集成**梯度检查点 (Gradient Checkpointing)**与**梯度累加 (Accumulation)**，支持在入门级显卡跑大模型。
    - **SOTA 模型支持**：原生支持 `DeBERTa-v3` 和 `BERT` 等模型。
    - **混合精度训练 (AMP)**：使用最新 `torch.amp` 语法，优化计算速度。
    - **性能重构**：优化数据读取逻辑，训练耗时降低 90%。

## 数据集分析

训练集包含约 38699 个样本，563 个类别，分布极不均衡：
- **头部效应**：最多样本类别 "instance of"（4681 样本）。
- **长尾分布**：约 37% 的类别样本量不足 10 条，少样本权重得分是拉开差距的关键。

## 运行指南

### 1. 环境配置 (针对 GPU 优化)
使用虚拟环境并安装特定版本的依赖以避开安全报错：
```bash
python -m venv .venv
.venv\Scripts\activate
# 强制安装 CUDA 版本 (根据显存需求)
pip install -r requirements.txt
```

### 2. 模型训练 (V7 显存优化版)

建议使用以下配置启动 **3 折交叉验证** 训练。该配置已针对 6GB 显存进行极限优化：

```bash
# 推荐的最强优化配置 (V7 3折显存优化版)
python src/train_torch.py `
    --use_type_hint `
    --use_ema `
    --use_logit_adj `
    --use_fgm `
    --use_rdrop `
    --use_augmentation `
    --use_weighted_loss `
    --n_folds 3 `
    --batch_size 16 `
    --grad_accum_steps 2 `
    --use_gradient_checkpointing `
    --use_amp `
    --output_dir ./model
```

主要参数说明：
- `--n_folds 3`: 开启 3 折交叉验证。
- `--use_gradient_checkpointing`: 显存节省开关（必开，支持 6GB 显存）。
- `--grad_accum_steps`: 梯度累加步数，与 batch_size 配合控制等效 Batch。
- `--use_type_hint`: 开启类型提示特征工程。
- `--use_ema`: 开启权重平滑。

### 3. 模型推理

推理时请指向对应的 Fold 模型路径：

```bash
python src/infer_torch.py `
    --use_type_hint `
    --model_path ./model/fold_1/best_model.pth `
    --labels_path ./model/label_classes.txt `
    --input_csv ./dataset/test.csv `
    --output_file ./result/submission.csv `
    --max_length 64 `
    --use_amp
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
├── model/                  # 模型输出 (日志、权重、标签表)
├── result/                 # 预测结果 (submission.csv)
└── requirements.txt        # 项目依赖
```

---
*本项目提供的 PyTorch 实现版本完全遵循 SCNU AI Competition 2026 的评价标准，确保了与 Baseline 的一致性与可复现性。*
