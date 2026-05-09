import argparse
import os
import random
import logging
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from transformers import (
    AutoTokenizer, 
    AutoModel, 
    get_linear_schedule_with_warmup,
)


# ==================== Data Loading & Preprocessing ====================
def load_data_from_directory(dir_path):
    """加载目录下所有CSV文件"""
    all_data = []
    if not os.path.exists(dir_path):
        raise ValueError(f"Directory not found: {dir_path}")

    csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    logging.info(f"Loading data from {dir_path}...")

    for filename in tqdm(csv_files, desc=f"Loading {os.path.basename(dir_path)}"):
        file_path = os.path.join(dir_path, filename)
        label_name = filename[:-4]
        try:
            df = pd.read_csv(file_path, low_memory=False, encoding='utf-8-sig')
            if df.empty:
                continue
            df.columns = [str(col).strip() for col in df.columns]
            if 'Subject' in df.columns and 'Object' in df.columns:
                df = df[['Subject', 'Object']].dropna()
                df['label'] = label_name
                all_data.append(df)
        except Exception as e:
            logging.warning(f"{filename} load error: {e}")

    if not all_data:
        raise ValueError(f"No valid data found in {dir_path}")

    full_df = pd.concat(all_data, ignore_index=True)
    full_df['Subject'] = full_df['Subject'].astype(str)
    full_df['Object'] = full_df['Object'].astype(str)
    return full_df


def detect_and_remove_poisoned_data(df, threshold_ratio=0.95, min_samples=5):
    """检测并移除中毒数据"""
    label_counts = df['label'].value_counts()
    rows_to_remove = []

    for label in label_counts.index:
        label_df = df[df['label'] == label]
        n_samples = len(label_df)
        
        if n_samples < min_samples:
            continue
        
        # 检测Subject重复率
        subject_counts = label_df['Subject'].value_counts()
        if len(subject_counts) > 0 and subject_counts.iloc[0] / n_samples > threshold_ratio:
            logging.warning(f"Removing poisoned label '{label}'")
            rows_to_remove.extend(label_df.index.tolist())
            continue
        
        # 检测Object重复率
        object_counts = label_df['Object'].value_counts()
        if len(object_counts) > 0 and object_counts.iloc[0] / n_samples > threshold_ratio:
            logging.warning(f"Removing poisoned label '{label}'")
            rows_to_remove.extend(label_df.index.tolist())
            continue
        
        # 检测(S,O)对重复率
        pair_counts = label_df.groupby(['Subject', 'Object']).size()
        if len(pair_counts) > 0 and pair_counts.max() / n_samples > threshold_ratio:
            logging.warning(f"Removing poisoned label '{label}'")
            rows_to_remove.extend(label_df.index.tolist())
            continue

    cleaned_df = df.drop(rows_to_remove).reset_index(drop=True)
    logging.info(f"Removed {len(rows_to_remove)} poisoned samples")
    return cleaned_df


def calculate_label_weights(label_counts):
    """根据赛题公式计算少样本重要性权重"""
    counts_max = label_counts.max()
    counts_min = label_counts.min()
    
    weights = {}
    for label, count in label_counts.items():
        numerator = counts_max - count + counts_min * 0.1
        denominator = counts_max + counts_min * 0.1
        weights[label] = numerator / denominator
    
    return weights


# ==================== Dataset & DataLoader ====================
class RelationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, label_encoder, max_length=128):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text_input = f"{row['Subject']} [SEP] {row['Object']}"
        
        encoding = self.tokenizer(
            text_input,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        label_id = torch.tensor(self.label_encoder.transform([row['label']])[0], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label_id': label_id
        }


# ==================== Model ====================
class CPAModel(nn.Module):
    def __init__(self, model_name, num_labels, dropout_rate=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 获取隐藏层维度
        if hasattr(self.encoder.config, 'hidden_size'):
            hidden_size = self.encoder.config.hidden_size
        elif hasattr(self.encoder.config, 'dim'):
            hidden_size = self.encoder.config.dim
        else:
            hidden_size = 768  # 默认值
        
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        if hasattr(outputs, 'last_hidden_state'):
            sequence_output = outputs.last_hidden_state
        elif isinstance(outputs, tuple):
            sequence_output = outputs[0]
        else:
            sequence_output = outputs
        
        cls_embedding = sequence_output[:, 0, :]
        logits = self.classifier(self.dropout(cls_embedding))
        return logits


# ==================== Evaluation Metric ====================
def calculate_weighted_score(predictions, labels, label_encoder, label_weights):
    """根据赛题公式计算加权分数"""
    correct_counts = defaultdict(int)
    total_counts = defaultdict(int)
    
    for pred, label in zip(predictions, labels):
        total_counts[label] += 1
        if pred == label:
            correct_counts[label] += 1
    
    weighted_sum = 0.0
    weight_sum = 0.0
    
    for label in label_encoder.classes_:
        label_idx = label_encoder.transform([label])[0]
        m_total = total_counts.get(label_idx, 0)
        m_correct = correct_counts.get(label_idx, 0)
        
        if m_total == 0:
            continue
        
        m_score = m_correct / m_total
        m_weight = label_weights.get(label, 0.0)
        
        weighted_sum += m_weight * m_score
        weight_sum += m_weight
    
    if weight_sum == 0:
        return 0.0
    
    return weighted_sum / weight_sum


# ==================== FGM Adversarial Training ====================
class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# ==================== Training Helpers ====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(save_dir, 'train.log'), mode='w', encoding='utf-8'),
            logging.StreamHandler(),
        ],
    )


def save_label_classes(label_encoder, save_dir):
    path = os.path.join(save_dir, 'label_classes.txt')
    with open(path, 'w', encoding='utf-8') as f:
        for label in label_encoder.classes_:
            f.write(f'{label}\n')


# ==================== Main Training Loop ====================
def run_training(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.output_dir, f'cpa_{timestamp}')
    setup_logging(save_dir)
    set_seed(args.random_seed)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')
    logging.info(f'Device: {device}')

    # 1. 加载并预处理数据
    logging.info("Step 1: Loading raw training data...")
    raw_train_df = load_data_from_directory(args.train_dir)
    logging.info(f"Raw data size: {len(raw_train_df)}")
    
    # 2. 检测并移除中毒数据
    logging.info("Step 2: Detecting and removing poisoned data...")
    cleaned_train_df = detect_and_remove_poisoned_data(
        raw_train_df, 
        threshold_ratio=args.poison_threshold,
        min_samples=args.min_poison_samples
    )
    logging.info(f"Cleaned data size: {len(cleaned_train_df)}")

    # 3. 构建标签编码器和权重
    logging.info("Step 3: Building label encoder and calculating weights...")
    label_encoder = LabelEncoder()
    label_encoder.fit(cleaned_train_df['label'].unique())
    num_classes = len(label_encoder.classes_)
    logging.info(f'Number of labels: {num_classes}')
    
    # 计算类别权重
    label_counts = cleaned_train_df['label'].value_counts()
    label_weights = calculate_label_weights(label_counts)
    index_weights = torch.tensor([label_weights.get(label, 1.0) for label in label_encoder.classes_], dtype=torch.float32)
    
    save_label_classes(label_encoder, save_dir)
    
    # 4. 分割数据集
    logging.info("Step 4: Splitting dataset...")
    
    # 检查是否有类别样本数少于2，如果有则不能使用分层抽样
    label_counts = cleaned_train_df['label'].value_counts()
    min_count = label_counts.min()
    
    if min_count < 2:
        logging.warning(f"Found classes with less than 2 samples (min: {min_count}). Using random split instead of stratified split.")
        train_df, val_df = train_test_split(
            cleaned_train_df,
            test_size=args.val_ratio,
            random_state=args.random_seed,
        )
    else:
        train_df, val_df = train_test_split(
            cleaned_train_df,
            test_size=args.val_ratio,
            stratify=cleaned_train_df['label'],
            random_state=args.random_seed,
        )
    
    # 5. Tokenizer & DataLoader
    logging.info("Step 5: Initializing tokenizer and dataloaders...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 加权采样（针对不平衡数据）
    if args.weighted_sampling:
        train_labels = train_df['label'].values
        label_to_weight = {label: label_weights[label] for label in label_counts.index}
        sample_weights = np.array([label_to_weight.get(label, 1.0) for label in train_labels])
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_shuffle = False
    else:
        sampler = None
        train_shuffle = True
    
    train_dataset = RelationDataset(train_df, tokenizer, label_encoder, args.max_length)
    val_dataset = RelationDataset(val_df, tokenizer, label_encoder, args.max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 6. 模型初始化
    logging.info("Step 6: Initializing model...")
    model = CPAModel(args.model_name, num_classes, args.dropout_rate).to(device)
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps
    )
    
    # 加权交叉熵损失
    criterion = nn.CrossEntropyLoss(weight=index_weights.to(device))

    # 7. 训练
    logging.info("Step 7: Starting training...")
    best_score = 0.0
    patience_counter = 0
    fgm = FGM(model) if args.use_fgm else None

    for epoch in range(args.epochs):
        model.train()
        tr_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}')
        for batch in pbar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label_id'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            
            # FGM对抗训练
            if fgm is not None:
                fgm.attack()
                logits_adv = model(input_ids, attention_mask)
                loss_adv = criterion(logits_adv, labels)
                loss_adv.backward()
                fgm.restore()
            
            optimizer.step()
            scheduler.step()
            
            tr_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 验证
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label_id'].to(device)
                
                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy().tolist())

        # 计算指标
        avg_train_loss = tr_loss / len(train_loader)
        val_acc = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)
        weighted_score = calculate_weighted_score(all_preds, all_labels, label_encoder, label_weights)

        logging.info(f'Epoch {epoch + 1} | Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | Weighted Score: {weighted_score:.4f}')

        # 保存最佳模型
        if weighted_score > best_score:
            best_score = weighted_score
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            tokenizer.save_pretrained(save_dir)
            logging.info(f'Best model saved! (Weighted Score: {best_score:.4f})')
        else:
            patience_counter += 1
            logging.info(f'Early stop count: {patience_counter}/{args.patience}')
            if patience_counter == args.patience:
                logging.info(f'Early stopping after {args.patience} epochs without improvement')
                break

    logging.info(f'Training finished. Best weighted score: {best_score:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training for CPA Task')
    
    # Data parameters
    parser.add_argument('--train_dir', type=str, default="./dataset/Train_Set")
    parser.add_argument('--output_dir', type=str, default='./cpa_output')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-base',
                        help='Pre-trained model name from Hugging Face')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Data preprocessing parameters
    parser.add_argument('--poison_threshold', type=float, default=0.95)
    parser.add_argument('--min_poison_samples', type=int, default=5)
    
    # Training enhancements
    parser.add_argument('--weighted_sampling', action='store_true', default=True,
                        help='Use weighted random sampling for class imbalance')
    parser.add_argument('--use_fgm', action='store_true', default=False,
                        help='Use FGM adversarial training')
    parser.add_argument('--device', type=str, default='gpu',
                        help='Device type: cpu or gpu')
    
    args = parser.parse_args()
    run_training(args)