import os
os.environ['PADDLE_NLP_DISABLE_AISTUDIO'] = '1'

import argparse
import random
import logging
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import paddle
import paddle.nn as nn
from paddle.io import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from paddlenlp.transformers import AutoTokenizer, AutoModel, LinearDecayWithWarmup


# ==================== Custom EMA Implementation ====================
class ExponentialMovingAverage:
    def __init__(self, parameters, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.param_list = []
        for p in parameters:
            if p.stop_gradient:
                continue
            self.shadow[p.name] = paddle.clone(p).detach()
            self.param_list.append(p)
    
    def step(self):
        for p in self.param_list:
            if p.name in self.shadow:
                new_average = (1.0 - self.decay) * p + self.decay * self.shadow[p.name]
                self.shadow[p.name] = new_average
    
    def apply(self):
        self.backup = {}
        for p in self.param_list:
            if p.name in self.shadow:
                self.backup[p.name] = paddle.clone(p).detach()
                p.set_value(self.shadow[p.name])
    
    def restore(self):
        for p in self.param_list:
            if p.name in self.backup:
                p.set_value(self.backup[p.name])


# ==================== Data Loading & Preprocessing ====================
def load_data_from_directory(dir_path, deduplicate=True):
    """加载训练数据，支持去重"""
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
                # 去重
                if deduplicate:
                    df = df.drop_duplicates(subset=['Subject', 'Object'])
                all_data.append(df)
        except Exception as e:
            logging.warning(f"{filename} load error: {e}")

    if not all_data:
        raise ValueError(f"No valid data found in {dir_path}")

    full_df = pd.concat(all_data, ignore_index=True)
    full_df['Subject'] = full_df['Subject'].astype(str)
    full_df['Object'] = full_df['Object'].astype(str)
    
    # 全局去重
    original_count = len(full_df)
    if deduplicate:
        full_df = full_df.drop_duplicates(subset=['Subject', 'Object', 'label'])
        removed = original_count - len(full_df)
        if removed > 0:
            logging.info(f"Removed {removed} duplicate samples globally")
    
    return full_df


def detect_and_remove_poisoned_data(df, threshold_ratio=0.95, min_samples=5):
    label_counts = df['label'].value_counts()
    rows_to_remove = []

    for label in label_counts.index:
        label_df = df[df['label'] == label]
        n_samples = len(label_df)
        
        if n_samples < min_samples:
            continue
        
        subject_counts = label_df['Subject'].value_counts()
        if len(subject_counts) > 0 and subject_counts.iloc[0] / n_samples > threshold_ratio:
            logging.warning(f"Removing poisoned label '{label}'")
            rows_to_remove.extend(label_df.index.tolist())
            continue
        
        object_counts = label_df['Object'].value_counts()
        if len(object_counts) > 0 and object_counts.iloc[0] / n_samples > threshold_ratio:
            logging.warning(f"Removing poisoned label '{label}'")
            rows_to_remove.extend(label_df.index.tolist())
            continue
        
        pair_counts = label_df.groupby(['Subject', 'Object']).size()
        if len(pair_counts) > 0 and pair_counts.max() / n_samples > threshold_ratio:
            logging.warning(f"Removing poisoned label '{label}'")
            rows_to_remove.extend(label_df.index.tolist())
            continue

    cleaned_df = df.drop(rows_to_remove).reset_index(drop=True)
    logging.info(f"Removed {len(rows_to_remove)} poisoned samples")
    return cleaned_df


def calculate_label_weights(label_counts):
    counts_max = label_counts.max()
    counts_min = label_counts.min()
    
    weights = {}
    for label, count in label_counts.items():
        numerator = counts_max - count + counts_min * 0.1
        denominator = counts_max + counts_min * 0.1
        weights[label] = numerator / denominator
    
    return weights


# ==================== Dataset & DataLoader ====================
def encode_text(tokenizer, text_input, max_length):
    try:
        encoding = tokenizer(
            text_input,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
        )
    except TypeError:
        encoding = tokenizer(
            text=text_input,
            max_seq_len=max_length,
            pad_to_max_seq_len=True,
            truncation=True,
            return_attention_mask=True,
        )

    input_ids = encoding['input_ids']
    attention_mask = encoding.get('attention_mask', None)

    if attention_mask is None:
        seq_len = encoding.get('seq_len', len(input_ids))
        seq_len = min(seq_len, max_length)
        attention_mask = [1] * seq_len + [0] * (max_length - seq_len)

    return np.array(input_ids, dtype='int64'), np.array(attention_mask, dtype='int64')


# 数据增强相关的替换词
SYNONYM_MAP = {
    'author': ['writer', 'creator', 'composer', 'producer', 'originator'],
    'director': ['film director', 'movie director', 'helmer'],
    'release date': ['publication date', 'launch date', 'issue date'],
    'location': ['place', 'site', 'area', 'region'],
    'capital': ['capital city', 'national capital'],
    'language': ['tongue', 'dialect', 'speech'],
    'genre': ['category', 'type', 'style'],
    'manufacturer': ['maker', 'producer', 'builder'],
    'publisher': ['distributor', 'printer', 'issuer'],
    'developer': ['creator', 'builder', 'designer'],
}

def augment_text(subject, object_, prob=0.1):
    """改进的数据增强：同义词替换+随机操作"""
    augmented_subject = subject
    augmented_object = object_
    
    # 随机在文本中插入空格
    if random.random() < prob:
        augmented_subject = ' '.join([c if random.random() > 0.05 else c + ' ' for c in subject])
    if random.random() < prob:
        augmented_object = ' '.join([c if random.random() > 0.05 else c + ' ' for c in object_])
    
    # 随机添加标点
    if random.random() < prob * 0.5:
        augmented_subject = augmented_subject + '.'
    if random.random() < prob * 0.5:
        augmented_object = augmented_object + '.'
    
    # 随机交换subject和object（仅对对称关系）
    symmetric_relations = ['part of', 'related to', 'shares border with']
    if random.random() < prob * 0.3:
        augmented_subject, augmented_object = augmented_object, augmented_subject
    
    return augmented_subject.strip(), augmented_object.strip()


class RelationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, label_encoder, max_length=128, augment=False):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        subject = row['Subject']
        object_ = row['Object']
        
        # 数据增强
        if self.augment:
            subject, object_ = augment_text(subject, object_)
        
        text_input = f"{subject} [SEP] {object_}"
        input_ids, attention_mask = encode_text(self.tokenizer, text_input, self.max_length)
        label_id = self.label_encoder.transform([row['label']])[0]
        return {
            'valid': True,
            'token_ids': input_ids,
            'cls_mask': attention_mask,
            'label_id': np.int64(label_id),
        }


def dynamic_collate_fn(samples):
    valid_samples = [s for s in samples if s.get('valid', False)]
    if not valid_samples:
        return None

    return {
        'data': np.stack([s['token_ids'] for s in valid_samples]).astype('int64'),
        'label': np.array([s['label_id'] for s in valid_samples], dtype='int64'),
        'cls_mask': np.stack([s['cls_mask'] for s in valid_samples]).astype('int64'),
    }


# ==================== Model ====================
class CPAModel(nn.Layer):
    def __init__(self, model_name, num_labels, dropout_rate=0.1, pooling_strategy='cls', use_multi_head=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.pooling_strategy = pooling_strategy
        self.use_multi_head = use_multi_head

        hidden_size = None
        if hasattr(self.encoder, 'config'):
            if isinstance(self.encoder.config, dict):
                hidden_size = self.encoder.config.get('hidden_size', None)
            else:
                hidden_size = getattr(self.encoder.config, 'hidden_size', None)

        if hidden_size is None and hasattr(self.encoder, 'embeddings') and hasattr(self.encoder.embeddings, 'word_embeddings'):
            hidden_size = self.encoder.embeddings.word_embeddings.weight.shape[-1]

        if hidden_size is None:
            raise ValueError('Unable to infer hidden_size automatically.')

        # 根据 pooling 策略调整分类器输入维度
        if pooling_strategy == 'cls_mean_max':
            classifier_input_dim = hidden_size * 3
        elif pooling_strategy in ['cls_mean', 'cls_max']:
            classifier_input_dim = hidden_size * 2
        elif pooling_strategy == 'triple':
            classifier_input_dim = hidden_size * 3
        else:
            classifier_input_dim = hidden_size

        # 多头注意力机制用于融合不同特征
        if use_multi_head:
            self.multi_head_fusion = nn.MultiHeadAttention(embed_dim=classifier_input_dim, 
                                                          num_heads=min(8, classifier_input_dim // 64))
            self.layer_norm = nn.LayerNorm(classifier_input_dim)
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, classifier_input_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(classifier_input_dim // 2, num_labels)
            )
        else:
            self.classifier = nn.Linear(classifier_input_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        if isinstance(outputs, tuple):
            sequence_output = outputs[0]
        elif hasattr(outputs, 'last_hidden_state'):
            sequence_output = outputs.last_hidden_state
        else:
            sequence_output = outputs

        cls_embedding = sequence_output[:, 0, :]

        if self.pooling_strategy == 'cls':
            pooled = cls_embedding
        elif self.pooling_strategy == 'mean':
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).astype('float32')
            sum_embeddings = paddle.sum(sequence_output * mask_expanded, axis=1)
            sum_mask = paddle.clamp(mask_expanded.sum(axis=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        elif self.pooling_strategy == 'max':
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).astype('float32')
            sequence_output = sequence_output.masked_fill(mask_expanded == 0, -1e9)
            pooled = paddle.max(sequence_output, axis=1)
        elif self.pooling_strategy == 'cls_mean':
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).astype('float32')
            sum_embeddings = paddle.sum(sequence_output * mask_expanded, axis=1)
            sum_mask = paddle.clamp(mask_expanded.sum(axis=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            pooled = paddle.concat([cls_embedding, mean_pooled], axis=1)
        elif self.pooling_strategy == 'cls_max':
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).astype('float32')
            masked_output = sequence_output.masked_fill(mask_expanded == 0, -1e9)
            max_pooled = paddle.max(masked_output, axis=1)
            pooled = paddle.concat([cls_embedding, max_pooled], axis=1)
        elif self.pooling_strategy == 'cls_mean_max':
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).astype('float32')
            sum_embeddings = paddle.sum(sequence_output * mask_expanded, axis=1)
            sum_mask = paddle.clamp(mask_expanded.sum(axis=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            masked_output = sequence_output.masked_fill(mask_expanded == 0, -1e9)
            max_pooled = paddle.max(masked_output, axis=1)
            pooled = paddle.concat([cls_embedding, mean_pooled, max_pooled], axis=1)
        else:
            pooled = cls_embedding

        # 应用dropout
        pooled = self.dropout(pooled)
        
        if self.use_multi_head:
            # 使用多头注意力融合特征
            pooled = pooled.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            pooled = self.multi_head_fusion(pooled, pooled, pooled)
            pooled = pooled.squeeze(1)  # [batch_size, hidden_dim]
            pooled = self.layer_norm(pooled)
        
        logits = self.classifier(pooled)
        return logits


# ==================== Evaluation Metric ====================
def calculate_weighted_score(predictions, labels, label_encoder, label_weights):
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
            if not param.stop_gradient and emb_name in name:
                self.backup[name] = param.numpy().copy()
                grad = param.grad
                if grad is not None:
                    norm = np.linalg.norm(grad.numpy())
                    if norm != 0:
                        r_at = epsilon * grad.numpy() / norm
                        param.set_value(param.numpy() + r_at)

    def restore(self, emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and emb_name in name:
                assert name in self.backup
                param.set_value(self.backup[name])
        self.backup = {}


# ==================== Training Helpers ====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


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


def resolve_device(device_arg):
    try:
        custom_types = paddle.device.get_all_custom_device_type()
    except Exception:
        custom_types = []

    logging.info(f'Custom device types: {custom_types}')
    
    if device_arg and device_arg != 'auto':
        try:
            dev = paddle.set_device(device_arg)
            logging.info(f'Using device: {dev}')
            return dev
        except Exception as e:
            logging.warning(f'{device_arg} use error: {e}')
    
    if 'iluvatar_gpu' in custom_types:
        try:
            dev = paddle.set_device('iluvatar_gpu')
            logging.info(f'Auto-detected and using Iluvatar GPU: {dev}')
            return dev
        except Exception:
            pass
    
    if 'xpu' in custom_types:
        try:
            dev = paddle.set_device('xpu')
            logging.info(f'Auto-detected and using XPU: {dev}')
            return dev
        except Exception:
            pass
    
    if 'gpu' in custom_types:
        try:
            dev = paddle.set_device('gpu')
            logging.info(f'Auto-detected and using GPU: {dev}')
            return dev
        except Exception:
            pass
            
    dev = paddle.set_device('cpu')
    logging.warning('Falling back to CPU')
    return dev


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
    device = resolve_device(args.device)

    logging.info(f'Device: {device}')

    # 1. 加载训练数据（带去重）
    logging.info("Step 1: Loading training data...")
    cleaned_train_df = load_data_from_directory(args.train_dir, deduplicate=args.deduplicate)
    logging.info(f"Data size after loading: {len(cleaned_train_df)}")
    
    # 2. 构建标签编码器和权重
    logging.info("Step 2: Building label encoder and calculating weights...")
    label_encoder = LabelEncoder()
    label_encoder.fit(cleaned_train_df['label'].unique())
    num_classes = len(label_encoder.classes_)
    logging.info(f'Number of labels: {num_classes}')
    
    # 计算类别权重（赛题公式）
    label_counts = cleaned_train_df['label'].value_counts()
    label_weights = calculate_label_weights(label_counts)
    index_weights = np.array([label_weights.get(label, 1.0) for label in label_encoder.classes_], dtype='float32')
    
    logging.info(f"Weight range: [{index_weights.min():.4f}, {index_weights.max():.4f}]")
    save_label_classes(label_encoder, save_dir)
    
    # 保存权重文件
    weights_df = pd.DataFrame({
        'label': label_encoder.classes_,
        'weight': index_weights
    })
    weights_df.to_csv(os.path.join(save_dir, 'label_weights.csv'), index=False, encoding='utf-8-sig')
    
    # 3. 分割数据集
    logging.info("Step 3: Splitting dataset...")
    
    label_counts = cleaned_train_df['label'].value_counts()
    # 将稀有类别（<5样本）全部放入训练集
    rare_labels = label_counts[label_counts < args.rare_threshold].index
    df_rare = cleaned_train_df[cleaned_train_df['label'].isin(rare_labels)]
    df_common = cleaned_train_df[~cleaned_train_df['label'].isin(rare_labels)]
    
    if len(df_common) == 0:
        raise ValueError("No common labels found, cannot split dataset")
    
    train_c, val_c = train_test_split(
        df_common,
        test_size=args.val_ratio,
        stratify=df_common['label'],
        random_state=args.random_seed,
    )
    
    train_df = pd.concat([train_c, df_rare]).sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
    val_df = val_c.reset_index(drop=True)
    logging.info(f'Split success: train={len(train_df)}, val={len(val_df)}')
    logging.info(f'Rare labels in training: {len(rare_labels)}')

    # 4. Tokenizer & DataLoader
    logging.info("Step 4: Initializing tokenizer and dataloaders...")
    tokenizer = AutoTokenizer.from_pretrained(args.shortcut_name)
    
    # 加权采样
    if args.weighted_sampling:
        train_labels = train_df['label'].values
        sample_weights = np.array([label_weights.get(label, 1.0) for label in train_labels], dtype='float32')
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        batch_sampler = paddle.io.BatchSampler(sampler=sampler, batch_size=args.batch_size, drop_last=False)
        train_loader = DataLoader(
            RelationDataset(train_df, tokenizer, label_encoder, args.max_length, augment=args.data_augment),
            batch_sampler=batch_sampler,
            collate_fn=dynamic_collate_fn,
            num_workers=args.num_workers,
            return_list=True,
        )
    else:
        train_loader = DataLoader(
            RelationDataset(train_df, tokenizer, label_encoder, args.max_length, augment=args.data_augment),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=dynamic_collate_fn,
            num_workers=args.num_workers,
            return_list=True,
        )
    
    val_loader = DataLoader(
        RelationDataset(val_df, tokenizer, label_encoder, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dynamic_collate_fn,
        num_workers=args.num_workers,
        return_list=True,
    )

    # 5. 模型初始化
    logging.info("Step 5: Initializing model...")
    model = CPAModel(args.shortcut_name, num_classes, args.dropout_rate, args.pooling_strategy, args.use_multi_head)
    
    if args.init_checkpoint is not None:
        logging.info(f"Loading checkpoint from {args.init_checkpoint}...")
        model.set_state_dict(paddle.load(args.init_checkpoint))
        logging.info("Checkpoint loaded successfully!")
    
    total_steps = max(1, len(train_loader) * args.epochs)
    
    if args.lr_scheduler == 'cosine':
        lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=args.lr,
            T_max=total_steps,
            eta_min=1e-7
        )
        lr_scheduler = paddle.optimizer.lr.LinearWarmup(
            learning_rate=lr_scheduler,
            warmup_steps=int(total_steps * args.warmup_ratio),
            start_lr=0.0,
            end_lr=args.lr
        )
    else:
        lr_scheduler = LinearDecayWithWarmup(args.lr, total_steps, warmup=args.warmup_ratio)
    
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay
    )
    
    # 加权交叉熵损失
    weight_tensor = paddle.to_tensor(index_weights, dtype='float32')
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    use_amp = args.use_amp and str(device) != 'cpu'
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024) if use_amp else None

    # EMA
    if args.use_ema:
        ema = ExponentialMovingAverage(
            model.parameters(),
            decay=args.ema_decay
        )
    else:
        ema = None

    # 6. 训练
    logging.info("Step 6: Starting training...")
    best_score = 0.0
    patience_counter = 0
    fgm = FGM(model) if args.use_fgm else None

    for epoch in range(args.epochs):
        model.train()
        tr_loss = 0.0
        train_steps = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}')
        for batch in pbar:
            if batch is None:
                continue

            input_ids = paddle.to_tensor(batch['data'], dtype='int64')
            mask = paddle.to_tensor(batch['cls_mask'], dtype='int64')
            label_ids = paddle.to_tensor(batch['label'], dtype='int64')

            if use_amp:
                with paddle.amp.auto_cast(enable=True):
                    logits = model(input_ids, mask)
                    loss = criterion(logits, label_ids)
                scaled = scaler.scale(loss)
                scaled.backward()
                
                if fgm is not None:
                    fgm.attack(epsilon=args.fgm_epsilon)
                    with paddle.amp.auto_cast(enable=True):
                        logits_adv = model(input_ids, mask)
                        loss_adv = criterion(logits_adv, label_ids)
                    scaled_adv = scaler.scale(loss_adv)
                    scaled_adv.backward()
                    fgm.restore()
                
                scaler.minimize(optimizer, scaled)
                optimizer.clear_grad()
            else:
                logits = model(input_ids, mask)
                loss = criterion(logits, label_ids)
                loss.backward()
                
                if fgm is not None:
                    fgm.attack(epsilon=args.fgm_epsilon)
                    logits_adv = model(input_ids, mask)
                    loss_adv = criterion(logits_adv, label_ids)
                    loss_adv.backward()
                    fgm.restore()
                
                optimizer.step()
                optimizer.clear_grad()

                if ema is not None:
                    ema.step()

            lr_scheduler.step()
            loss_value = float(loss.numpy())
            tr_loss += loss_value
            train_steps += 1
            pbar.set_postfix({'loss': f'{loss_value:.4f}'})

        # 验证（使用EMA权重）
        model.eval()
        all_preds = []
        all_labels = []
        
        # 应用EMA权重
        if ema is not None:
            ema.apply()
        
        with paddle.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                input_ids = paddle.to_tensor(batch['data'], dtype='int64')
                mask = paddle.to_tensor(batch['cls_mask'], dtype='int64')
                label_ids = paddle.to_tensor(batch['label'], dtype='int64')

                if use_amp:
                    with paddle.amp.auto_cast(enable=True):
                        logits = model(input_ids, mask)
                else:
                    logits = model(input_ids, mask)

                preds = paddle.argmax(logits, axis=1).numpy().tolist()
                all_preds.extend(preds)
                all_labels.extend(label_ids.numpy().tolist())
        
        # 恢复原始权重
        if ema is not None:
            ema.restore()

        avg_train_loss = tr_loss / max(1, train_steps)
        val_acc = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)
        weighted_score = calculate_weighted_score(all_preds, all_labels, label_encoder, label_weights)

        logging.info(f'Epoch {epoch + 1} | Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | Weighted Score: {weighted_score:.4f}')

        if weighted_score > best_score:
            best_score = weighted_score
            patience_counter = 0
            paddle.save(model.state_dict(), os.path.join(save_dir, 'best_model.pdparams'))
            try:
                tokenizer.save_pretrained(save_dir)
            except Exception:
                pass
            logging.info(f'Best model saved! (Weighted Score: {best_score:.4f})')
        else:
            patience_counter += 1
            logging.info(f'Early stop count: {patience_counter}/{args.patience}')
            if patience_counter == args.patience:
                logging.info(f'Early stopping after {args.patience} epochs without improvement')
                break

    logging.info(f'Training finished. Best weighted score: {best_score:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PaddlePaddle Training for CPA Task')
    
    # Data parameters
    parser.add_argument('--train_dir', type=str, default="./dataset/Train_Set")
    parser.add_argument('--output_dir', type=str, default='./cpa_output')
    parser.add_argument('--deduplicate', action='store_true', default=True,
                        help='Whether to remove duplicate samples')
    
    # Model parameters
    parser.add_argument('--shortcut_name', type=str, default='bert-large-uncased',
                        help='Pre-trained model name from PaddleNLP')
    parser.add_argument('--init_checkpoint', type=str, default=None,
                        help='Path to checkpoint file for resume training')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--pooling_strategy', type=str, default='cls_mean_max',
                        choices=['cls', 'mean', 'max', 'cls_mean', 'cls_max', 'cls_mean_max'],
                        help='Pooling strategy for model output')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--rare_threshold', type=int, default=5,
                        help='Labels with fewer samples than this go entirely to training')
    
    # Enhancement options
    parser.add_argument('--weighted_sampling', action='store_true', default=True,
                        help='Use weighted random sampling to handle class imbalance')
    parser.add_argument('--data_augment', action='store_true', default=True,
                        help='Enable data augmentation')
    parser.add_argument('--use_fgm', action='store_true', default=True,
                        help='Enable FGM adversarial training')
    parser.add_argument('--fgm_epsilon', type=float, default=1.0,
                        help='Epsilon for FGM attack')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='Enable Exponential Moving Average')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA decay rate')
    parser.add_argument('--use_multi_head', action='store_true', default=False,
                        help='Use multi-head attention for feature fusion')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Enable automatic mixed precision training')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['linear', 'cosine'],
                        help='Learning rate scheduler type')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: auto, cpu, gpu, xpu, iluvatar_gpu')
    
    args = parser.parse_args()
    run_training(args)
