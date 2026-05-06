import argparse
import os
import random
import logging
import warnings
import re
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import paddle
import paddle.nn as nn
from paddle.io import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from tqdm import tqdm

from paddlenlp.transformers import AutoTokenizer, AutoModel, LinearDecayWithWarmup
# Chinese synonym dictionary for data augmentation
CHINESE_SYNONYMS = {
    '是': ['为', '乃', '即', '便是'],
    '有': ['存在', '具备', '拥有', '具有'],
    '在': ['于', '处在', '位于', '居于'],
    '和': ['与', '及', '以及', '同'],
    '的': ['之', '其', '所'],
    '了': ['已', '已经', '已然'],
    '不': ['非', '并非', '绝非'],
    '人': ['人们', '人类', '民众'],
    '说': ['讲', '谈论', '表述'],
    '看': ['瞧', '观察', '注视'],
    '想': ['思考', '思索', '考虑'],
    '做': ['干', '从事', '进行'],
    '好': ['优秀', '出色', '良好'],
    '大': ['巨大', '庞大', '宏大'],
    '小': ['微小', '细小', '细微'],
    '多': ['许多', '众多', '大量'],
    '少': ['稀少', '稀缺', '少量'],
    '上': ['之上', '上方', '上面'],
    '下': ['之下', '下方', '下面'],
    '前': ['之前', '前方', '前面'],
    '后': ['之后', '后方', '后面'],
    '中': ['之中', '中间', '当中'],
    '高': ['高大', '高大', '崇高'],
    '低': ['低下', '低矮', '卑微'],
    '长': ['长久', '漫长', '悠长'],
    '短': ['短暂', '短促', '简短'],
    '快': ['迅速', '快速', '飞快'],
    '慢': ['缓慢', '迟缓', '慢条斯理'],
    '新': ['崭新', '新鲜', '新颖'],
    '旧': ['陈旧', '古老', '破旧'],
    '来': ['来到', '到来', '来临'],
    '去': ['离去', '离开', '前往'],
    '出': ['出现', '产生', '发出'],
    '进': ['进入', '走进', '迈入'],
    '过': ['经过', '通过', '经历'],
    '到': ['到达', '抵达', '至'],
    '会': ['将会', '将要', '可能'],
    '能': ['可以', '能够', '得以'],
    '要': ['需要', '想要', '将要'],
    '应': ['应该', '应当', '理应'],
    '得': ['必须', '需要', '不得不'],
    '就': ['便是', '就是', '即刻'],
    '都': ['全部', '所有', '皆'],
    '也': ['亦', '同样', '亦是'],
    '很': ['非常', '十分', '特别'],
    '更': ['更加', '越发', '愈加'],
    '最': ['最为', '至极', '极度'],
    '又': ['再次', '再度', '另外'],
    '再': ['再次', '重新', '再度'],
    '还': ['仍然', '依然', '照旧'],
    '只': ['仅仅', '只是', '唯有'],
    '才': ['方才', '刚刚', '始才'],
    '却': ['然而', '但是', '反倒'],
    '可': ['可以', '能够', '却'],
    '而': ['并且', '而且', '然而'],
    '或': ['或者', '或许', '要么'],
    '若': ['如果', '假如', '倘若'],
    '因': ['因为', '由于', '鉴于'],
    '故': ['所以', '因此', '因而'],
    '使': ['让', '令', '致使'],
    '被': ['遭', '受', '为...所'],
    '把': ['将', '拿', '用'],
    '对': ['对于', '关于', '针对'],
    '给': ['给予', '予以', '为'],
    '向': ['朝着', '对着', '往'],
    '从': ['自', '由', '由...起'],
    '比': ['比较', '相比', '较之'],
    '跟': ['和', '与', '同'],
    '同': ['和', '与', '跟'],
    '与': ['和', '跟', '同'],
}

# data load
def load_data_from_directory(dir_path):
    all_data = []
    if not os.path.exists(dir_path):
        raise ValueError(f"can't find: {dir_path}")

    csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    logging.info(f"load data from {dir_path} ...")

    for filename in tqdm(csv_files, desc=f"loading {os.path.basename(dir_path)}"):
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
        raise ValueError(f"{dir_path} not valid data")

    full_df = pd.concat(all_data, ignore_index=True)
    full_df['Subject'] = full_df['Subject'].astype(str)
    full_df['Object'] = full_df['Object'].astype(str)
    return full_df

# Data Augmentation Utils
class TextAugmenter:
    def __init__(self):
        self.synonyms = CHINESE_SYNONYMS
    
    def synonym_replace(self, text, p=0.2):
        chars = list(text)
        for i in range(len(chars)):
            if random.random() < p and chars[i] in self.synonyms:
                chars[i] = random.choice(self.synonyms[chars[i]])
        return ''.join(chars)
    
    def random_delete(self, text, p=0.1):
        chars = list(text)
        result = []
        for char in chars:
            if random.random() >= p:
                result.append(char)
        return ''.join(result) if result else text
    
    def random_insert(self, text, p=0.1):
        chars = list(text)
        result = []
        for char in chars:
            result.append(char)
            if random.random() < p and self.synonyms:
                random_char = random.choice(list(self.synonyms.keys()))
                result.append(random_char)
        return ''.join(result)
    
    def random_swap(self, text):
        chars = list(text)
        if len(chars) < 2:
            return text
        idx1, idx2 = random.sample(range(len(chars)), 2)
        chars[idx1], chars[idx2] = chars[idx2], chars[idx1]
        return ''.join(chars)
    
    def augment(self, text, aug_type='mix'):
        if aug_type == 'replace':
            return self.synonym_replace(text)
        elif aug_type == 'delete':
            return self.random_delete(text)
        elif aug_type == 'insert':
            return self.random_insert(text)
        elif aug_type == 'swap':
            return self.random_swap(text)
        else:
            aug_methods = [self.synonym_replace, self.random_delete, self.random_insert, self.random_swap]
            method = random.choice(aug_methods)
            return method(text)

# Few-shot Augmentation
def augment_few_shot_samples(df, augmenter, min_samples=5, aug_multiplier=3):
    augmented_data = []
    label_counts = df['label'].value_counts()
    
    for label, count in label_counts.items():
        label_data = df[df['label'] == label]
        
        if count < min_samples:
            for idx, row in label_data.iterrows():
                augmented_data.append(row)
                for _ in range(aug_multiplier):
                    new_subject = augmenter.augment(row['Subject'])
                    new_object = augmenter.augment(row['Object'])
                    augmented_data.append({
                        'Subject': new_subject,
                        'Object': new_object,
                        'label': label
                    })
        else:
            augmented_data.extend(label_data.to_dict('records'))
    
    return pd.DataFrame(augmented_data).sample(frac=1, random_state=42).reset_index(drop=True)

# encode data
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

# dataset
class RelationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, label_encoder, max_length=128, augmenter=None, aug_prob=0.0):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.le = label_encoder
        self.max_length = max_length
        self.augmenter = augmenter
        self.aug_prob = aug_prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        subject = row['Subject']
        object_ = row['Object']
        
        if self.augmenter and random.random() < self.aug_prob:
            subject = self.augmenter.augment(subject)
            object_ = self.augmenter.augment(object_)
        
        text_input = f"{subject} [SEP] {object_}"
        input_ids, attention_mask = encode_text(self.tokenizer, text_input, self.max_length)
        label_id = self.le.transform([row['label']])[0]
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

# Focal Loss for Few-shot
class FocalLoss(nn.Layer):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, logits, labels):
        ce_loss = nn.functional.cross_entropy(logits, labels, weight=self.alpha, reduction='none')
        pt = paddle.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return paddle.mean(focal_loss)
        elif self.reduction == 'sum':
            return paddle.sum(focal_loss)
        else:
            return focal_loss

# Contrastive Loss
class ContrastiveLoss(nn.Layer):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        batch_size = embeddings.shape[0]
        embeddings = nn.functional.normalize(embeddings, axis=1)
        
        similarity_matrix = paddle.matmul(embeddings, embeddings.t()) / self.temperature
        
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).astype('float32')
        mask = mask - paddle.eye(batch_size, dtype='float32')
        
        exp_sim = paddle.exp(similarity_matrix) * (1 - paddle.eye(batch_size, dtype='float32'))
        log_prob = similarity_matrix - paddle.log(paddle.sum(exp_sim, axis=1, keepdim=True))
        
        mean_log_prob_pos = paddle.sum(mask * log_prob, axis=1) / paddle.clamp(paddle.sum(mask, axis=1), min=1)
        
        loss = -mean_log_prob_pos.mean()
        return loss

# model with enhanced features
class CPAModel(nn.Layer):
    def __init__(self, shortcut_name, num_classes, hidden_size=None, use_contrastive=False):
        super(CPAModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(shortcut_name)
        if hidden_size is None:
            self.hidden_size = self.encoder.config['hidden_size']
        else:
            self.hidden_size = hidden_size
        
        self.use_contrastive = use_contrastive
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, num_classes)
        )
        
        # Attention layer for better representation
        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1)
        )
        
        # Projection head for contrastive learning
        if self.use_contrastive:
            self.projection_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size)
            )

    def forward(self, input_ids, attention_mask, return_embedding=False):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        seq_output = outputs[0]
        
        # Enhanced attention pooling
        mask = paddle.cast(attention_mask, dtype='float32').unsqueeze(-1)
        attn_weights = self.attention_layer(seq_output)
        attn_weights = nn.functional.softmax(attn_weights, axis=1) * mask
        attn_weights = attn_weights / paddle.clamp(paddle.sum(attn_weights, axis=1, keepdim=True), min=1e-9)
        
        pooled_output = paddle.sum(seq_output * attn_weights, axis=1)
        
        if return_embedding or self.use_contrastive:
            if self.use_contrastive:
                proj_output = self.projection_head(pooled_output)
                logits = self.classifier(pooled_output)
                return logits, proj_output
            return pooled_output
        
        logits = self.classifier(pooled_output)
        return logits

# FGM for adversarial training
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and emb_name in name:
                self.backup[name] = param.numpy().copy()
                grad = param.grad.numpy()
                norm = np.linalg.norm(grad)
                if norm != 0 and not np.isnan(norm):
                    r_at = epsilon * grad / norm
                    param.set_value(param.numpy() + r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and emb_name in name:
                assert name in self.backup
                param.set_value(self.backup[name])
        self.backup = {}

# EMA (Exponential Moving Average)
class EMA():
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if not param.stop_gradient:
                self.shadow[name] = param.numpy().copy()

    def update(self):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient:
                assert name in self.shadow
                new_val = (1.0 - self.decay) * param.numpy() + self.decay * self.shadow[name]
                self.shadow[name] = new_val

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient:
                assert name in self.shadow
                self.backup[name] = param.numpy().copy()
                param.set_value(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if not param.stop_gradient:
                assert name in self.backup
                param.set_value(self.backup[name])
        self.backup = {}

# seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

# log
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

# device
def resolve_device(device_arg):
    try:
        custom_types = paddle.device.get_all_custom_device_type()
    except Exception:
        custom_types = []

    logging.info(f'custom device types: {custom_types}')
    
    if device_arg:
        try:
            dev = paddle.set_device(device_arg)
            logging.info(f'use: {dev}')
            return dev
        except Exception as e:
            logging.warning(f'{device_arg} use error: {e}')
            
    dev = paddle.set_device('cpu')
    logging.warning('set device to CPU')
    return dev

# labels
def save_label_classes(label_encoder, save_dir):
    path = os.path.join(save_dir, 'label_classes.txt')
    with open(path, 'w', encoding='utf-8') as f:
        for label in label_encoder.classes_:
            f.write(f'{label}\n')

# few-shot weight calculation based on question.md
def calculate_few_shot_weights(counts_dict, label_encoder):
    counts = np.array([counts_dict.get(label, 1) for label in label_encoder.classes_], dtype=np.float32)
    c_max = np.max(counts)
    c_min = np.min(counts)
    
    # Formula from question.md with enhanced weighting
    weights = (c_max - counts + c_min * 0.1) / (c_max + c_min * 0.1)
    
    # Add smoothing to avoid extreme weights
    weights = weights * 0.8 + 0.2 * np.ones_like(weights)
    
    return paddle.to_tensor(weights, dtype='float32')

# train
def run_training(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.output_dir, f'cpa_{timestamp}')
    setup_logging(save_dir)
    set_seed(args.random_seed)
    device = resolve_device(args.device)

    logging.info(f'device: {device}')

    # 1. load train data
    raw_train_df = load_data_from_directory(args.train_dir)

    # 2. build label
    label_encoder = LabelEncoder()
    label_encoder.fit(raw_train_df['label'].unique())
    num_classes = len(label_encoder.classes_)
    logging.info(f'label_num: {num_classes}')
    save_label_classes(label_encoder, save_dir)
    
    # 2.1 Calculate weights for loss function based on few-shot importance
    label_counts = raw_train_df['label'].value_counts().to_dict()
    class_weights = calculate_few_shot_weights(label_counts, label_encoder)
    logging.info("few-shot weights calculated and applied to Loss function.")
    
    # 2.2 Augment few-shot samples
    augmenter = TextAugmenter()
    augmented_df = augment_few_shot_samples(raw_train_df, augmenter, min_samples=args.min_samples, aug_multiplier=args.aug_multiplier)
    logging.info(f"Data augmented: {len(raw_train_df)} -> {len(augmented_df)}")

    # 3. split dataset with stratified strategy
    counts = augmented_df['label'].value_counts()
    rare_labels = counts[counts < args.rare_threshold].index
    df_rare = augmented_df[augmented_df['label'].isin(rare_labels)]
    df_common = augmented_df[~augmented_df['label'].isin(rare_labels)]

    if len(df_common) == 0:
        raise ValueError("data num < 2, can't split dataset")

    train_c, val_c = train_test_split(
        df_common,
        test_size=args.val_ratio,
        stratify=df_common['label'],
        random_state=args.random_seed,
    )
    train_df = pd.concat([train_c, df_rare]).sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
    val_df = val_c.reset_index(drop=True)
    logging.info(f'split success: train={len(train_df)}, val={len(val_df)}')

    # 4. Tokenizer & DataLoader
    tokenizer = AutoTokenizer.from_pretrained(args.shortcut_name)
    
    train_dataset = RelationDataset(
        train_df, 
        tokenizer, 
        label_encoder, 
        args.max_length,
        augmenter=augmenter if args.use_augmentation else None,
        aug_prob=args.aug_prob
    )
    
    train_loader = DataLoader(
        train_dataset,
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

    # 5. model init
    model = CPAModel(args.shortcut_name, num_classes, use_contrastive=args.use_contrastive)
    total_steps = max(1, len(train_loader) * args.epoch)
    lr_scheduler = LinearDecayWithWarmup(args.lr, total_steps, warmup=args.warmup_ratio)
    optimizer = paddle.optimizer.AdamW(learning_rate=lr_scheduler, parameters=model.parameters(), weight_decay=args.weight_decay)
    
    # Use Focal Loss for better few-shot performance
    if args.use_focal_loss:
        loss_fn = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    # Contrastive loss
    contrastive_loss_fn = ContrastiveLoss(temperature=args.contrastive_temperature)
    
    # Initialize FGM and EMA
    fgm = FGM(model)
    ema = EMA(model, decay=args.ema_decay)

    use_amp = args.use_amp and str(device) != 'cpu'
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024) if use_amp else None

    # 6. training
    best_val_acc = 0.0
    best_f1 = 0.0
    best_final_score = 0.0
    patience_counter = 0
    patience_limit = args.patience

    logging.info('start training...')
    for epoch in range(args.epoch):
        model.train()
        tr_loss = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epoch}')
        for batch in pbar:
            if batch is None:
                continue

            input_ids = paddle.to_tensor(batch['data'], dtype='int64') if not isinstance(batch['data'], paddle.Tensor) else batch['data']
            mask = paddle.to_tensor(batch['cls_mask'], dtype='int64') if not isinstance(batch['cls_mask'], paddle.Tensor) else batch['cls_mask']
            label_ids = paddle.to_tensor(batch['label'], dtype='int64') if not isinstance(batch['label'], paddle.Tensor) else batch['label']

            if use_amp:
                with paddle.amp.auto_cast(enable=True):
                    if args.use_contrastive:
                        logits, embeddings = model(input_ids, mask)
                        ce_loss = loss_fn(logits, label_ids)
                        contrastive_loss = contrastive_loss_fn(embeddings, label_ids)
                        loss = ce_loss + args.contrastive_weight * contrastive_loss
                    else:
                        logits = model(input_ids, mask)
                        loss = loss_fn(logits, label_ids)
                scaled = scaler.scale(loss)
                scaled.backward()
                
                # FGM Attack
                fgm.attack(epsilon=args.fgm_epsilon)
                with paddle.amp.auto_cast(enable=True):
                    if args.use_contrastive:
                        logits_adv, embeddings_adv = model(input_ids, mask)
                        ce_loss_adv = loss_fn(logits_adv, label_ids)
                        contrastive_loss_adv = contrastive_loss_fn(embeddings_adv, label_ids)
                        loss_adv = ce_loss_adv + args.contrastive_weight * contrastive_loss_adv
                    else:
                        logits_adv = model(input_ids, mask)
                        loss_adv = loss_fn(logits_adv, label_ids)
                scaled_adv = scaler.scale(loss_adv)
                scaled_adv.backward()
                fgm.restore()
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.clear_grad()
            else:
                if args.use_contrastive:
                    logits, embeddings = model(input_ids, mask)
                    ce_loss = loss_fn(logits, label_ids)
                    contrastive_loss = contrastive_loss_fn(embeddings, label_ids)
                    loss = ce_loss + args.contrastive_weight * contrastive_loss
                else:
                    logits = model(input_ids, mask)
                    loss = loss_fn(logits, label_ids)
                loss.backward()
                
                # FGM Attack
                fgm.attack(epsilon=args.fgm_epsilon)
                if args.use_contrastive:
                    logits_adv, embeddings_adv = model(input_ids, mask)
                    ce_loss_adv = loss_fn(logits_adv, label_ids)
                    contrastive_loss_adv = contrastive_loss_fn(embeddings_adv, label_ids)
                    loss_adv = ce_loss_adv + args.contrastive_weight * contrastive_loss_adv
                else:
                    logits_adv = model(input_ids, mask)
                    loss_adv = loss_fn(logits_adv, label_ids)
                loss_adv.backward()
                fgm.restore()
                
                optimizer.step()
                optimizer.clear_grad()

            # Update EMA
            ema.update()
            
            lr_scheduler.step()
            loss_value = float(loss.numpy())
            tr_loss += loss_value
            train_steps += 1
            pbar.set_postfix({'loss': f'{loss_value:.4f}'})

        # Validation with EMA
        ema.apply_shadow()
        
        m_weights = class_weights.numpy()
        m_correct = np.zeros(num_classes)
        m_total = np.zeros(num_classes)
        all_preds = []
        all_labels = []
        
        model.eval()
        with paddle.no_grad():
            for batch in val_loader:
                if batch is None: continue
                input_ids = paddle.to_tensor(batch['data'], dtype='int64') if not isinstance(batch['data'], paddle.Tensor) else batch['data']
                mask = paddle.to_tensor(batch['cls_mask'], dtype='int64') if not isinstance(batch['cls_mask'], paddle.Tensor) else batch['cls_mask']
                label_ids = batch['label']
                
                logits = model(input_ids, mask) if not args.use_contrastive else model(input_ids, mask)[0]
                preds = paddle.argmax(logits, axis=1).numpy()
                labels = label_ids
                
                all_preds.extend(preds)
                all_labels.extend(labels)
                
                for p, l in zip(preds, labels):
                    m_total[l] += 1
                    if p == l:
                        m_correct[l] += 1
        
        ema.restore()
        
        m_scores = np.divide(m_correct, m_total, out=np.zeros_like(m_correct), where=m_total!=0)
        valid_mask = m_total > 0
        final_score = np.sum(m_weights[valid_mask] * m_scores[valid_mask]) / np.sum(m_weights[valid_mask])
        
        val_acc = np.sum(m_correct) / np.sum(m_total) if np.sum(m_total) > 0 else 0.0
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        logging.info(f"Epoch {epoch+1} | Loss: {tr_loss/max(1, train_steps):.4f} | Val Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Score_final: {final_score:.4f}")

        if final_score > best_final_score:
            best_final_score = final_score
            best_val_acc = val_acc
            best_f1 = val_f1
            patience_counter = 0
            logging.info(f"best model! (Score_final: {final_score:.4f})")
            paddle.save(model.state_dict(), os.path.join(save_dir, 'best_model.pdparams'))
            try:
                tokenizer.save_pretrained(save_dir)
            except Exception:
                pass
        else:
            patience_counter += 1
            logging.info(f'early stop count: {patience_counter}/{patience_limit}')
            if patience_counter == patience_limit:
                logging.info(f'{patience_limit} epoch not up, early stop!!!')
                break

    logging.info(f'train finish, best score: {best_final_score:.4f}, best acc: {best_val_acc:.4f}, best f1: {best_f1:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default="./dataset/Train_Set")
    parser.add_argument('--output_dir', type=str, default='./cpa_output')
    parser.add_argument('--shortcut_name', type=str, default='bert-base-chinese')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--use_flash_attention', action='store_true')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # Few-shot specific arguments
    parser.add_argument('--min_samples', type=int, default=5, help='Minimum samples for augmentation')
    parser.add_argument('--aug_multiplier', type=int, default=3, help='Augmentation multiplier for few-shot samples')
    parser.add_argument('--rare_threshold', type=int, default=10, help='Threshold for rare labels')
    parser.add_argument('--use_augmentation', action='store_true', default=True, help='Enable data augmentation')
    parser.add_argument('--aug_prob', type=float, default=0.3, help='Probability of augmentation during training')
    
    # Focal Loss arguments
    parser.add_argument('--use_focal_loss', action='store_true', default=True, help='Use Focal Loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
    
    # Contrastive learning arguments
    parser.add_argument('--use_contrastive', action='store_true', default=True, help='Use contrastive learning')
    parser.add_argument('--contrastive_weight', type=float, default=0.5, help='Contrastive loss weight')
    parser.add_argument('--contrastive_temperature', type=float, default=0.5, help='Contrastive temperature')
    
    # FGM arguments
    parser.add_argument('--fgm_epsilon', type=float, default=1.0, help='FGM epsilon')
    
    # EMA arguments
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate')
    
    args = parser.parse_args()
    run_training(args)