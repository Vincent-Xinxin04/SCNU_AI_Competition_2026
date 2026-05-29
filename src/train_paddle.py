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


# ==================== Exponential Moving Average (EMA) ====================
class ExponentialMovingAverage:
    """指数移动平均，提升模型稳定性和泛化能力"""
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
        """更新 EMA 参数"""
        for p in self.param_list:
            if p.name in self.shadow:
                new_average = (1.0 - self.decay) * p + self.decay * self.shadow[p.name]
                self.shadow[p.name] = new_average
    
    def apply(self):
        """应用 EMA 参数到模型"""
        self.backup = {}
        for p in self.param_list:
            if p.name in self.shadow:
                self.backup[p.name] = paddle.clone(p).detach()
                p.set_value(self.shadow[p.name])
    
    def restore(self):
        """恢复原始参数"""
        for p in self.param_list:
            if p.name in self.backup:
                p.set_value(self.backup[p.name])


# ==================== Data Loading & Preprocessing ====================
def load_data_from_directory(dir_path, deduplicate=True):
    """
    从目录加载训练数据
    :param dir_path: 训练数据目录路径
    :param deduplicate: 是否去重
    :return: 合并后的 DataFrame
    """
    all_data = []
    if not os.path.exists(dir_path):
        raise ValueError(f"训练数据目录不存在: {dir_path}")

    csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    logging.info(f"正在从 {dir_path} 加载数据...")

    for filename in tqdm(csv_files, desc=f"加载 {os.path.basename(dir_path)}"):
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
                if deduplicate:
                    df = df.drop_duplicates(subset=['Subject', 'Object'])
                all_data.append(df)
        except Exception as e:
            logging.warning(f"加载 {filename} 时出错: {e}")

    if not all_data:
        raise ValueError(f"{dir_path} 中未找到有效数据")

    full_df = pd.concat(all_data, ignore_index=True)
    full_df['Subject'] = full_df['Subject'].astype(str)
    full_df['Object'] = full_df['Object'].astype(str)
    
    original_count = len(full_df)
    if deduplicate:
        full_df = full_df.drop_duplicates(subset=['Subject', 'Object', 'label'])
        removed = original_count - len(full_df)
        if removed > 0:
            logging.info(f"全局去重: 移除了 {removed} 条重复样本")
    
    return full_df


def calculate_label_weights(label_counts):
    """
    按照赛题公式计算类别权重
    :param label_counts: 类别样本数统计
    :return: 类别权重字典
    """
    counts_max = label_counts.max()
    counts_min = label_counts.min()
    
    weights = {}
    for label, count in label_counts.items():
        numerator = counts_max - count + counts_min * 0.1
        denominator = counts_max + counts_min * 0.1
        weights[label] = numerator / denominator
    
    return weights


# ==================== Feature Encoding ====================
def encode_text(tokenizer, text_input, max_length):
    """
    将文本编码为模型输入格式
    :param tokenizer: 预训练 tokenizer
    :param text_input: 输入文本
    :param max_length: 最大序列长度
    :return: input_ids, attention_mask
    """
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


# ==================== Data Augmentation ====================
def augment_text(subject, object_, prob=0.1):
    """
    数据增强：随机插入空格、添加标点、交换主体对象
    :param subject: 主体文本
    :param object_: 对象文本
    :param prob: 增强概率
    :return: 增强后的主体和对象
    """
    augmented_subject = subject
    augmented_object = object_
    
    if random.random() < prob:
        augmented_subject = ' '.join([c if random.random() > 0.05 else c + ' ' for c in subject])
    if random.random() < prob:
        augmented_object = ' '.join([c if random.random() > 0.05 else c + ' ' for c in object_])
    
    if random.random() < prob * 0.5:
        augmented_subject = augmented_subject + '.'
    if random.random() < prob * 0.5:
        augmented_object = augmented_object + '.'
    
    if random.random() < prob * 0.3:
        augmented_subject, augmented_object = augmented_object, augmented_subject
    
    return augmented_subject.strip(), augmented_object.strip()


# ==================== Dataset Class ====================
class RelationDataset(Dataset):
    """关系分类数据集"""
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
    """动态批量处理函数"""
    valid_samples = [s for s in samples if s.get('valid', False)]
    if not valid_samples:
        return None

    return {
        'data': np.stack([s['token_ids'] for s in valid_samples]).astype('int64'),
        'label': np.array([s['label_id'] for s in valid_samples], dtype='int64'),
        'cls_mask': np.stack([s['cls_mask'] for s in valid_samples]).astype('int64'),
    }


# ==================== Prototype Learning ====================
class PrototypeClassifier(nn.Layer):
    """
    原型分类器：用于少样本学习场景
    通过计算样本特征与类别原型的余弦相似度进行分类
    """
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.proto_vectors = nn.Parameter(
            paddle.randn([num_labels, hidden_size]) * 0.01
        )
    
    def forward(self, features):
        norm_features = features / paddle.norm(features, axis=1, keepdim=True) + 1e-8
        norm_protos = self.proto_vectors / paddle.norm(self.proto_vectors, axis=1, keepdim=True) + 1e-8
        similarities = paddle.matmul(norm_features, norm_protos.t())
        return similarities


# ==================== Main Model ====================
class CPAModel(nn.Layer):
    """
    表格语义关系提取模型 (CPA - Column Pair Analysis)
    
    架构设计：
    1. 预训练编码器：BERT/RoBERTa
    2. 特征聚合：支持多种 Pooling 策略
    3. 多头注意力融合：增强特征表达能力
    4. 原型分类器：提升少样本学习性能
    """
    def __init__(self, model_name, num_labels, dropout_rate=0.1, 
                 pooling_strategy='cls_mean_max', use_multi_head=False, 
                 use_prototype=False, prototype_weight=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.pooling_strategy = pooling_strategy
        self.use_multi_head = use_multi_head
        self.use_prototype = use_prototype
        self.prototype_weight = prototype_weight

        # 自动推断隐藏层维度
        hidden_size = self._infer_hidden_size()
        
        # 根据 pooling 策略确定分类器输入维度
        classifier_input_dim = self._calculate_input_dim(hidden_size)

        # 多头注意力特征融合
        if use_multi_head:
            self.multi_head_fusion = nn.MultiHeadAttention(
                embed_dim=classifier_input_dim, 
                num_heads=min(8, classifier_input_dim // 64)
            )
            self.layer_norm = nn.LayerNorm(classifier_input_dim)
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, classifier_input_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(classifier_input_dim // 2, num_labels)
            )
        else:
            self.classifier = nn.Linear(classifier_input_dim, num_labels)
        
        # 原型分类器（少样本学习）
        if use_prototype:
            self.proto_classifier = PrototypeClassifier(classifier_input_dim, num_labels)

    def _infer_hidden_size(self):
        """自动推断预训练模型的隐藏层维度"""
        hidden_size = None
        if hasattr(self.encoder, 'config'):
            if isinstance(self.encoder.config, dict):
                hidden_size = self.encoder.config.get('hidden_size', None)
            else:
                hidden_size = getattr(self.encoder.config, 'hidden_size', None)

        if hidden_size is None and hasattr(self.encoder, 'embeddings'):
            if hasattr(self.encoder.embeddings, 'word_embeddings'):
                hidden_size = self.encoder.embeddings.word_embeddings.weight.shape[-1]

        if hidden_size is None:
            raise ValueError('无法自动推断 hidden_size，请检查模型配置')
        
        return hidden_size

    def _calculate_input_dim(self, hidden_size):
        """根据 pooling 策略计算分类器输入维度"""
        if self.pooling_strategy == 'cls_mean_max':
            return hidden_size * 3
        elif self.pooling_strategy in ['cls_mean', 'cls_max']:
            return hidden_size * 2
        elif self.pooling_strategy == 'triple':
            return hidden_size * 3
        else:
            return hidden_size

    def _pooling(self, sequence_output, attention_mask, cls_embedding):
        """根据策略进行特征聚合"""
        if self.pooling_strategy == 'cls':
            return cls_embedding
        elif self.pooling_strategy == 'mean':
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).astype('float32')
            sum_embeddings = paddle.sum(sequence_output * mask_expanded, axis=1)
            sum_mask = paddle.clamp(mask_expanded.sum(axis=1), min=1e-9)
            return sum_embeddings / sum_mask
        elif self.pooling_strategy == 'max':
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).astype('float32')
            sequence_output = sequence_output.masked_fill(mask_expanded == 0, -1e9)
            return paddle.max(sequence_output, axis=1)
        elif self.pooling_strategy == 'cls_mean':
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).astype('float32')
            sum_embeddings = paddle.sum(sequence_output * mask_expanded, axis=1)
            sum_mask = paddle.clamp(mask_expanded.sum(axis=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            return paddle.concat([cls_embedding, mean_pooled], axis=1)
        elif self.pooling_strategy == 'cls_max':
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).astype('float32')
            masked_output = sequence_output.masked_fill(mask_expanded == 0, -1e9)
            max_pooled = paddle.max(masked_output, axis=1)
            return paddle.concat([cls_embedding, max_pooled], axis=1)
        elif self.pooling_strategy == 'cls_mean_max':
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).astype('float32')
            sum_embeddings = paddle.sum(sequence_output * mask_expanded, axis=1)
            sum_mask = paddle.clamp(mask_expanded.sum(axis=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            masked_output = sequence_output.masked_fill(mask_expanded == 0, -1e9)
            max_pooled = paddle.max(masked_output, axis=1)
            return paddle.concat([cls_embedding, mean_pooled, max_pooled], axis=1)
        else:
            return cls_embedding

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        if isinstance(outputs, tuple):
            sequence_output = outputs[0]
        elif hasattr(outputs, 'last_hidden_state'):
            sequence_output = outputs.last_hidden_state
        else:
            sequence_output = outputs

        cls_embedding = sequence_output[:, 0, :]
        pooled = self._pooling(sequence_output, attention_mask, cls_embedding)
        pooled = self.dropout(pooled)
        
        if self.use_multi_head:
            pooled = pooled.unsqueeze(1)
            pooled = self.multi_head_fusion(pooled, pooled, pooled)
            pooled = pooled.squeeze(1)
            pooled = self.layer_norm(pooled)
        
        logits = self.classifier(pooled)
        
        if self.use_prototype:
            proto_logits = self.proto_classifier(pooled)
            logits = (1 - self.prototype_weight) * logits + self.prototype_weight * proto_logits
        
        return logits


# ==================== Evaluation Metric ====================
def calculate_weighted_score(predictions, labels, label_encoder, label_weights):
    """
    计算加权分数（按照赛题公式）
    :param predictions: 预测标签索引列表
    :param labels: 真实标签索引列表
    :param label_encoder: 标签编码器
    :param label_weights: 类别权重字典
    :return: 加权分数
    """
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
    
    return weighted_sum / weight_sum if weight_sum > 0 else 0.0


# ==================== FGM Adversarial Training ====================
class FGM:
    """
    Fast Gradient Method (FGM) 对抗训练
    通过在 Embedding 层注入对抗扰动增强模型鲁棒性
    """
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name='embeddings'):
        """执行对抗攻击"""
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
        """恢复参数"""
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and emb_name in name:
                assert name in self.backup
                param.set_value(self.backup[name])
        self.backup = {}


# ==================== Utility Functions ====================
def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def setup_logging(save_dir):
    """配置日志记录"""
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
    """解析并设置计算设备"""
    try:
        custom_types = paddle.device.get_all_custom_device_type()
    except Exception:
        custom_types = []

    logging.info(f'可用自定义设备类型: {custom_types}')
    
    if device_arg and device_arg != 'auto':
        try:
            dev = paddle.set_device(device_arg)
            logging.info(f'使用指定设备: {dev}')
            return dev
        except Exception as e:
            logging.warning(f'{device_arg} 设备使用失败: {e}')
    
    if 'iluvatar_gpu' in custom_types:
        try:
            dev = paddle.set_device('iluvatar_gpu')
            logging.info(f'自动检测并使用 Iluvatar GPU: {dev}')
            return dev
        except Exception:
            pass
    
    if 'xpu' in custom_types:
        try:
            dev = paddle.set_device('xpu')
            logging.info(f'自动检测并使用 XPU: {dev}')
            return dev
        except Exception:
            pass
    
    if 'gpu' in custom_types:
        try:
            dev = paddle.set_device('gpu')
            logging.info(f'自动检测并使用 GPU: {dev}')
            return dev
        except Exception:
            pass
            
    dev = paddle.set_device('cpu')
    logging.warning('回退到 CPU')
    return dev


def save_label_classes(label_encoder, save_dir):
    """保存标签类别文件"""
    path = os.path.join(save_dir, 'label_classes.txt')
    with open(path, 'w', encoding='utf-8') as f:
        for label in label_encoder.classes_:
            f.write(f'{label}\n')


def save_checkpoint(model, optimizer, lr_scheduler, epoch, best_score, save_dir):
    """保存训练断点"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'epoch': epoch,
        'best_score': best_score,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch{epoch+1}.pdparams')
    paddle.save(checkpoint, checkpoint_path)
    logging.info(f'检查点已保存: {checkpoint_path}')
    return checkpoint_path


# ==================== Main Training Loop ====================
def run_training(args):
    """主训练函数"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.output_dir, f'cpa_{timestamp}')
    setup_logging(save_dir)
    set_seed(args.random_seed)
    device = resolve_device(args.device)

    logging.info(f'训练设备: {device}')

    # 1. 加载训练数据（带去重）
    logging.info("步骤 1: 加载训练数据...")
    train_df = load_data_from_directory(args.train_dir, deduplicate=args.deduplicate)
    logging.info(f"数据加载完成，样本数: {len(train_df)}")
    
    # 2. 构建标签编码器和权重
    logging.info("步骤 2: 构建标签编码器和计算类别权重...")
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df['label'].unique())
    num_classes = len(label_encoder.classes_)
    logging.info(f'类别数量: {num_classes}')
    
    # 计算类别权重（赛题公式）
    label_counts = train_df['label'].value_counts()
    label_weights = calculate_label_weights(label_counts)
    index_weights = np.array([label_weights.get(label, 1.0) for label in label_encoder.classes_], dtype='float32')
    
    logging.info(f"类别权重范围: [{index_weights.min():.4f}, {index_weights.max():.4f}]")
    save_label_classes(label_encoder, save_dir)
    
    # 保存权重文件（便于分析）
    weights_df = pd.DataFrame({
        'label': label_encoder.classes_,
        'weight': index_weights
    })
    weights_df.to_csv(os.path.join(save_dir, 'label_weights.csv'), index=False, encoding='utf-8-sig')
    
    # 3. 数据集分割（稀有类别全部放入训练集）
    logging.info("步骤 3: 分割数据集...")
    
    rare_labels = label_counts[label_counts < args.rare_threshold].index
    df_rare = train_df[train_df['label'].isin(rare_labels)]
    df_common = train_df[~train_df['label'].isin(rare_labels)]
    
    if len(df_common) == 0:
        raise ValueError("未找到常见类别，无法分割数据集")
    
    train_c, val_c = train_test_split(
        df_common,
        test_size=args.val_ratio,
        stratify=df_common['label'],
        random_state=args.random_seed,
    )
    
    train_df = pd.concat([train_c, df_rare]).sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
    val_df = val_c.reset_index(drop=True)
    logging.info(f'数据集分割完成: 训练集={len(train_df)}, 验证集={len(val_df)}')
    logging.info(f'稀有类别数量: {len(rare_labels)}')

    # 4. 初始化 Tokenizer 和 DataLoader
    logging.info("步骤 4: 初始化 Tokenizer 和 DataLoader...")
    tokenizer = AutoTokenizer.from_pretrained(args.shortcut_name)
    
    # 加权采样处理类别不平衡
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
    logging.info("步骤 5: 初始化模型...")
    model = CPAModel(
        args.shortcut_name, 
        num_classes, 
        args.dropout_rate, 
        args.pooling_strategy, 
        args.use_multi_head, 
        args.use_prototype, 
        args.prototype_weight
    )
    
    if args.init_checkpoint is not None:
        logging.info(f"从检查点加载模型: {args.init_checkpoint}")
        model.set_state_dict(paddle.load(args.init_checkpoint))
    
    # 6. 优化器和学习率调度器
    logging.info("步骤 6: 配置优化器和学习率调度器...")
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
    
    # 加权交叉熵损失（按照赛题权重公式）
    weight_tensor = paddle.to_tensor(index_weights, dtype='float32')
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    use_amp = args.use_amp and str(device) != 'cpu'
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024) if use_amp else None

    # EMA（指数移动平均）
    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay) if args.use_ema else None

    def train_one_epoch(model, train_loader, optimizer, lr_scheduler, criterion, fgm, ema, scaler, use_amp, epoch, total_epochs):
        """训练单个 epoch"""
        model.train()
        tr_loss = 0.0
        train_steps = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{total_epochs}')
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
        
        return tr_loss / max(1, train_steps)
    
    def evaluate(model, val_loader, use_amp, ema):
        """评估模型性能"""
        model.eval()
        all_preds = []
        all_labels = []
        
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
        
        if ema is not None:
            ema.restore()
        
        val_acc = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)
        weighted_score = calculate_weighted_score(all_preds, all_labels, label_encoder, label_weights)
        
        return val_acc, weighted_score
    
    # 7. 开始训练
    logging.info("步骤 7: 开始训练...")
    best_score = 0.0
    patience_counter = 0
    fgm = FGM(model) if args.use_fgm else None

    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, lr_scheduler,
            criterion, fgm, ema, scaler, use_amp, epoch, args.epochs
        )
        val_acc, weighted_score = evaluate(model, val_loader, use_amp, ema)
        logging.info(f'Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Weighted Score: {weighted_score:.4f}')
        
        if args.save_checkpoint_epochs > 0 and (epoch + 1) % args.save_checkpoint_epochs == 0:
            save_checkpoint(model, optimizer, lr_scheduler, epoch, best_score, save_dir)

        if weighted_score > best_score:
            best_score = weighted_score
            patience_counter = 0
            paddle.save(model.state_dict(), os.path.join(save_dir, 'best_model.pdparams'))
            try:
                tokenizer.save_pretrained(save_dir)
            except Exception:
                pass
            logging.info(f'最佳模型已保存! (加权分数: {best_score:.4f})')
        else:
            patience_counter += 1
            logging.info(f'早停计数器: {patience_counter}/{args.patience}')
            if patience_counter == args.patience:
                logging.info(f'连续 {args.patience} 个 epoch 无提升，触发早停')
                break

    logging.info(f'训练完成. 最佳加权分数: {best_score:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='表格语义关系提取模型训练 (CPA - Column Pair Analysis)')
    
    # 数据参数
    parser.add_argument('--train_dir', type=str, default="./dataset/Train_Set",
                        help='训练数据目录')
    parser.add_argument('--output_dir', type=str, default='./cpa_output',
                        help='模型输出目录')
    parser.add_argument('--deduplicate', action='store_true', default=True,
                        help='是否去除重复样本')
    
    # 模型参数
    parser.add_argument('--shortcut_name', type=str, default='bert-large-uncased',
                        help='PaddleNLP 预训练模型名称')
    parser.add_argument('--init_checkpoint', type=str, default=None,
                        help='初始化检查点路径')
    parser.add_argument('--max_length', type=int, default=128,
                        help='最大序列长度')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='Dropout 率')
    parser.add_argument('--pooling_strategy', type=str, default='cls_mean_max',
                        choices=['cls', 'mean', 'max', 'cls_mean', 'cls_max', 'cls_mean_max'],
                        help='特征聚合策略')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批处理大小')
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='学习率预热比例')
    parser.add_argument('--patience', type=int, default=5,
                        help='早停耐心值')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载线程数')
    parser.add_argument('--rare_threshold', type=int, default=5,
                        help='稀有类别阈值（样本数少于此值的类别全部放入训练集）')
    
    # 增强选项
    parser.add_argument('--weighted_sampling', action='store_true', default=True,
                        help='使用加权随机采样处理类别不平衡')
    parser.add_argument('--data_augment', action='store_true', default=True,
                        help='启用数据增强')
    parser.add_argument('--use_fgm', action='store_true', default=True,
                        help='启用 FGM 对抗训练')
    parser.add_argument('--fgm_epsilon', type=float, default=1.0,
                        help='FGM 攻击的 epsilon 值')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='启用指数移动平均')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA 衰减率')
    parser.add_argument('--use_multi_head', action='store_true', default=False,
                        help='使用多头注意力进行特征融合')
    parser.add_argument('--use_prototype', action='store_true', default=False,
                        help='启用原型学习（用于少样本分类）')
    parser.add_argument('--prototype_weight', type=float, default=0.3,
                        help='原型分类器权重（0-1）')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='启用自动混合精度训练')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['linear', 'cosine'],
                        help='学习率调度器类型')
    
    # 检查点参数
    parser.add_argument('--save_checkpoint_epochs', type=int, default=2,
                        help='每 N 个 epoch 保存一次检查点')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='auto',
                        help='设备类型: auto, cpu, gpu, xpu, iluvatar_gpu')
    
    args = parser.parse_args()
    run_training(args)
