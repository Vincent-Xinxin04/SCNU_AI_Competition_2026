import argparse
import os
import random
import logging
import warnings
import re
from datetime import datetime

# Set Hugging Face Mirror (Allow override via environment variable)
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# EMA
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, 
                                 reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


# Supervised Contrastive Loss
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # Compute mean of log-likelihood over positive
        # Modify mask to avoid division by zero
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        loss = -mean_log_prob_pos.mean()
        return loss

# Adversarial Training (FGM)
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


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
            
            # Locate Subject and Object columns in a case-insensitive way.
            subject_col = None
            object_col = None
            for col in df.columns:
                if col.lower() == 'subject':
                    subject_col = col
                elif col.lower() == 'object':
                    object_col = col
            
            if subject_col is not None and object_col is not None:
                df = df[[subject_col, object_col]].dropna()
                df['label'] = label_name
                all_data.append(df)
        except Exception as e:
            logging.warning(f"{filename} load error: {e}")

    if not all_data:
        raise ValueError(f"{dir_path} not valid data")

    full_df = pd.concat(all_data, ignore_index=True)
    
    # Re-find columns after concat
    subject_col = None
    object_col = None
    for col in full_df.columns:
        if col.lower() == 'subject':
            subject_col = col
        elif col.lower() == 'object':
            object_col = col
            
    full_df[subject_col] = full_df[subject_col].astype(str)
    full_df[object_col] = full_df[object_col].astype(str)
    return full_df


# encode data
def get_type_hint(text):
    text = str(text)
    if re.match(r'^-?\d+(\.\d+)?$', text):
        return "[NUM]"
    if re.search(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}\s+[A-Za-z]+\s+\d{4}', text):
        return "[DATE]"
    if re.search(r'https?://\S+', text):
        return "[URL]"
    return "[TXT]"

def encode_pair(tokenizer, text_a, text_b, max_length, use_type_hint=False):
    if use_type_hint:
        text_a = f"{get_type_hint(text_a)} {text_a}"
        text_b = f"{get_type_hint(text_b)} {text_b}"
        
    encoding = tokenizer(
        text=text_a,
        text_pair=text_b,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors=None
    )
    
    input_ids = encoding['input_ids']
    attention_mask = encoding.get('attention_mask', None)

    if attention_mask is None:
        seq_len = len(input_ids)
        seq_len = min(seq_len, max_length)
        attention_mask = [1] * seq_len + [0] * (max_length - seq_len)

    return np.array(input_ids, dtype='int64'), np.array(attention_mask, dtype='int64')


# dataset
class RelationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, label_encoder, max_length=64, augment=False, use_type_hint=False):
        self.tokenizer = tokenizer
        self.augment = augment
        self.max_length = max_length
        self.use_type_hint = use_type_hint
        
        # Pre-process Subject and Object columns
        subject_col, object_col = None, None
        for col in dataframe.columns:
            if col.lower() == 'subject': subject_col = col
            elif col.lower() == 'object': object_col = col
        
        self.subjects = dataframe[subject_col].astype(str).tolist()
        self.objects = dataframe[object_col].astype(str).tolist()
        self.labels = label_encoder.transform(dataframe['label']).astype('int64').tolist()

    def __len__(self):
        return len(self.labels)

    def _apply_augmentation(self, text):
        if not text: return text
        words = text.split()
        if len(words) < 2: return text
        # Random word deletion
        if random.random() < 0.1:
            idx = random.randint(0, len(words) - 1)
            words.pop(idx)
        # Random word swap
        if len(words) >= 2 and random.random() < 0.1:
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return " ".join(words)

    def __getitem__(self, idx):
        subject_text = self.subjects[idx]
        object_text = self.objects[idx]
        
        if self.augment:
            subject_text = self._apply_augmentation(subject_text)
            object_text = self._apply_augmentation(object_text)
            
        input_ids, attention_mask = encode_pair(self.tokenizer, subject_text, object_text, self.max_length, use_type_hint=self.use_type_hint)
            
        return {
            'valid': True,
            'token_ids': input_ids,
            'cls_mask': attention_mask,
            'label_id': self.labels[idx],
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


# model
class CPAModel(nn.Module):
    def __init__(self, shortcut_name, num_classes, use_gradient_checkpointing=False):
        super(CPAModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(shortcut_name)
        if use_gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
            logging.info("Gradient checkpointing enabled for VRAM saving.")
            
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token pooling
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_embedding))
        return logits


def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask can be used to ignore certain samples if needed
    if pad_mask is not None:
        p_loss.masked_fill_(~pad_mask, 0.)
        q_loss.masked_fill_(~pad_mask, 0.)

    # Normalized by batch size to be consistent with mean CE loss
    p_loss = p_loss.sum() / p.size(0)
    q_loss = q_loss.sum() / q.size(0)

    loss = (p_loss + q_loss) / 2
    return loss

# seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    if device_arg and 'cuda' in device_arg and torch.cuda.is_available():
        device = torch.device(device_arg)
        logging.info(f'use: {device}')
        return device
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f'use: {device}')
        return device
    else:
        device = torch.device('cpu')
        logging.warning('set device to CPU')
        return device


# labels
def save_label_classes(label_encoder, save_dir):
    path = os.path.join(save_dir, 'label_classes.txt')
    with open(path, 'w', encoding='utf-8') as f:
        for label in label_encoder.classes_:
            f.write(f'{label}\n')

def calculate_competition_score(all_preds, all_labels, m_weights, num_classes):
    """
    完全遵循赛题文档的最终分数计算规则:
    Score_final = sum(m_weights * m_score) / sum(m_weights)
    m_score = m_correct / m_total
    """
    m_correct = np.zeros(num_classes)
    m_total = np.zeros(num_classes)
    
    for p, l in zip(all_preds, all_labels):
        m_total[l] += 1
        if p == l:
            m_correct[l] += 1
            
    m_scores = np.zeros(num_classes)
    # 只有在验证集中出现的类别才参与计算，避免分母为0
    present_mask = (m_total > 0)
    
    for i in range(num_classes):
        if m_total[i] > 0:
            m_scores[i] = m_correct[i] / m_total[i]
            
    # 分母只计算验证集中出现的类别的权重和，以保证验证得分的公平性
    numerator = (m_weights[present_mask] * m_scores[present_mask]).sum()
    denominator = m_weights[present_mask].sum()
    
    score_final = numerator / denominator if denominator > 0 else 0.0
    return score_final, m_scores, m_total

def get_optimizer_params(model, lr, weight_decay, layerwise_lr_decay):
    no_decay = ["bias", "LayerNorm.weight"]
    # For BERT-like models
    optimizer_grouped_parameters = []
    
    # Check if it's DeBERTa or BERT
    if hasattr(model.encoder, 'embeddings'):
        layers = model.encoder.encoder.layer if hasattr(model.encoder.encoder, 'layer') else model.encoder.transformer.layer
        
        # Initial LR for embeddings
        current_lr = lr * (layerwise_lr_decay ** (len(layers) + 1))
        
        optimizer_grouped_parameters.append({
            "params": [p for n, p in model.encoder.embeddings.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            "lr": current_lr,
        })
        optimizer_grouped_parameters.append({
            "params": [p for n, p in model.encoder.embeddings.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": current_lr,
        })
        
        # LLRD for layers
        for i, layer in enumerate(layers):
            current_lr = lr * (layerwise_lr_decay ** (len(layers) - i))
            optimizer_grouped_parameters.append({
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": current_lr,
            })
            optimizer_grouped_parameters.append({
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": current_lr,
            })
            
        # Task head (classifier) gets the base LR
        optimizer_grouped_parameters.append({
            "params": [p for n, p in model.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            "lr": lr,
        })
        optimizer_grouped_parameters.append({
            "params": [p for n, p in model.classifier.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": lr,
        })
    else:
        # Fallback to simple grouping
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        
    return optimizer_grouped_parameters

# train
def run_training(args):
    save_dir_base = args.output_dir
    os.makedirs(save_dir_base, exist_ok=True)
    setup_logging(save_dir_base)
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
    save_label_classes(label_encoder, save_dir_base)

    # 3. K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.random_seed)
    
    # Filter out labels with only 1 sample for StratifiedKFold
    counts = raw_train_df['label'].value_counts()
    valid_labels = counts[counts >= args.n_folds].index
    df_fold = raw_train_df[raw_train_df['label'].isin(valid_labels)].reset_index(drop=True)
    df_single = raw_train_df[~raw_train_df['label'].isin(valid_labels)].reset_index(drop=True)
    
    logging.info(f"Starting {args.n_folds}-fold cross validation...")
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_fold, df_fold['label'])):
        logging.info(f"\n{'='*20} Fold {fold + 1}/{args.n_folds} {'='*20}")
        
        save_dir = os.path.join(save_dir_base, f"fold_{fold+1}")
        os.makedirs(save_dir, exist_ok=True)
        
        train_df = pd.concat([df_fold.iloc[train_idx], df_single]).sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
        val_df = df_fold.iloc[val_idx].reset_index(drop=True)
        
        # Calculate competition weights based on FULL train set distribution
        counts_full = raw_train_df['label'].value_counts()
        counts_max = counts_full.max()
        counts_min = counts_full.min()
        
        m_weights_tensor = torch.zeros(num_classes).to(device)
        for label, count in counts_full.items():
            weight = (counts_max - count + counts_min * 0.1) / (counts_max + counts_min * 0.1)
            class_idx = label_encoder.transform([label])[0]
            m_weights_tensor[class_idx] = weight
            
        # Logit Adj Offsets
        priors = np.zeros(num_classes)
        for label, count in counts_full.items():
            class_idx = label_encoder.transform([label])[0]
            priors[class_idx] = count
        priors = priors / priors.sum()
        logit_adj_offsets = torch.tensor(np.log(priors + 1e-9)).to(device, dtype=torch.float32)

        # 4. Tokenizer & DataLoader
        tokenizer = AutoTokenizer.from_pretrained(args.shortcut_name)
        train_dataset = RelationDataset(train_df, tokenizer, label_encoder, args.max_length, augment=args.use_augmentation, use_type_hint=args.use_type_hint)
        
        sampler = None
        if args.use_balanced_sampler:
            class_sample_counts = train_df['label'].value_counts()
            weights = 1.0 / (class_sample_counts ** 0.5)
            sample_weights = train_df['label'].map(weights).values
            sampler = WeightedRandomSampler(torch.from_numpy(sample_weights), len(sample_weights))
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler, collate_fn=dynamic_collate_fn, num_workers=args.num_workers)
        val_loader = DataLoader(RelationDataset(val_df, tokenizer, label_encoder, args.max_length, use_type_hint=args.use_type_hint), batch_size=args.batch_size, shuffle=False, collate_fn=dynamic_collate_fn, num_workers=args.num_workers)

        # 5. model init
        model = CPAModel(args.shortcut_name, num_classes)
        model.to(device)
        
        total_steps = max(1, len(train_loader) * args.epoch)
        optimizer = torch.optim.AdamW(get_optimizer_params(model, args.lr, args.weight_decay, args.layerwise_lr_decay))
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio), num_training_steps=total_steps)
        
        if args.use_focal_loss:
            loss_fn = FocalLoss(weight=m_weights_tensor if args.use_weighted_loss else None, gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
        else:
            loss_fn = nn.CrossEntropyLoss(weight=m_weights_tensor if args.use_weighted_loss else None, label_smoothing=args.label_smoothing)

        ema = EMA(model, args.ema_decay) if args.use_ema else None
        if ema: ema.register()
        fgm = FGM(model) if args.use_fgm else None
        use_amp = args.use_amp and device.type != 'cpu'
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # 6. training loop
        best_score = 0.0
        patience_counter = 0
        
        for epoch in range(args.epoch):
            model.train()
            tr_loss, train_steps = 0.0, 0
            pbar = tqdm(train_loader, desc=f'Fold {fold+1} Epoch {epoch+1}')
            
            for batch in pbar:
                if batch is None: continue
                input_ids = torch.tensor(batch['data'], dtype=torch.long, device=device)
                mask = torch.tensor(batch['cls_mask'], dtype=torch.long, device=device)
                label_ids = torch.tensor(batch['label'], dtype=torch.long, device=device)
                
                optimizer.zero_grad()
                
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    if args.use_rdrop:
                        input_ids_double = torch.cat([input_ids, input_ids], dim=0)
                        mask_double = torch.cat([mask, mask], dim=0)
                        logits_all = model(input_ids_double, mask_double)
                        if args.use_logit_adj: logits_all = logits_all + args.logit_adj_tau * logit_adj_offsets
                        logits, logits2 = torch.split(logits_all, input_ids.size(0))
                        ce_loss = 0.5 * (loss_fn(logits, label_ids) + loss_fn(logits2, label_ids))
                        loss = ce_loss + args.rdrop_alpha * compute_kl_loss(logits, logits2)
                    else:
                        logits = model(input_ids, mask)
                        if args.use_logit_adj: logits = logits + args.logit_adj_tau * logit_adj_offsets
                        loss = loss_fn(logits, label_ids)
                    
                    loss = loss / args.grad_accum_steps
                
                scaler.scale(loss).backward()
                
                if (train_steps + 1) % args.grad_accum_steps == 0:
                    if fgm:
                        fgm.attack()
                        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                            logits_adv = model(input_ids, mask)
                            if args.use_logit_adj: logits_adv = logits_adv + args.logit_adj_tau * logit_adj_offsets
                            loss_adv = loss_fn(logits_adv, label_ids) / args.grad_accum_steps
                        scaler.scale(loss_adv).backward()
                        fgm.restore()
                        
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if ema: ema.update()
                    lr_scheduler.step()
                
                tr_loss += loss.item() * args.grad_accum_steps
                train_steps += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # val stage
            model.eval()
            if ema: ema.apply_shadow()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    if batch is None: continue
                    input_ids = torch.tensor(batch['data'], dtype=torch.long, device=device)
                    mask = torch.tensor(batch['cls_mask'], dtype=torch.long, device=device)
                    logits = model(input_ids, mask)
                    all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                    all_labels.extend(batch['label'])

            score_final, m_scores, m_total_val = calculate_competition_score(np.array(all_preds), np.array(all_labels), m_weights_tensor.cpu().numpy(), num_classes)
            logging.info(f'Fold {fold+1} Epoch {epoch+1} | Score: {score_final:.4f}')

            if score_final > best_score:
                best_score = score_final
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                logging.info(f'New best score: {best_score:.4f}')
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logging.info("Early stopping...")
                    break
            if ema: ema.restore()
            
        fold_scores.append(best_score)
        
    logging.info(f'\nCV Finish! Avg Score: {np.mean(fold_scores):.4f} | Scores: {fold_scores}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default="./dataset/Train_Set")
    parser.add_argument('--output_dir', type=str, default='./model')
    parser.add_argument('--shortcut_name', type=str, default='bert-base-uncased')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_weighted_loss', action='store_true', help='Use weighted loss based on competition importance')
    parser.add_argument('--oversample_rare', action='store_true', help='Oversample rare classes in training set')
    parser.add_argument('--use_augmentation', action='store_true', help='Use simple text augmentation')
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--use_balanced_sampler', action='store_true', help='Use WeightedRandomSampler for class balancing')
    parser.add_argument('--use_focal_loss', action='store_true', help='Use FocalLoss for class imbalance')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma for FocalLoss')
    parser.add_argument('--use_rdrop', action='store_true', help='Use R-Drop regularization')
    parser.add_argument('--rdrop_alpha', type=float, default=4.0, help='Weight for KL loss in R-Drop')
    parser.add_argument('--use_supcon', action='store_true', help='Use Supervised Contrastive Learning')
    parser.add_argument('--supcon_temp', type=float, default=0.07, help='Temperature for SupCon')
    parser.add_argument('--supcon_weight', type=float, default=0.1, help='Weight for SupCon loss')
    parser.add_argument('--use_mixup', action='store_true', help='Use Manifold Mixup')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Alpha for Beta distribution in Mixup')
    parser.add_argument('--use_fgm', action='store_true', help='Use FGM adversarial training')
    parser.add_argument('--use_logit_adj', action='store_true', help='Use Logit Adjustment for long-tail learning')
    parser.add_argument('--logit_adj_tau', type=float, default=1.0, help='Tau for Logit Adjustment')
    parser.add_argument('--use_type_hint', action='store_true', help='Use data type hinting in features')
    parser.add_argument('--use_ema', action='store_true', help='Use Exponential Moving Average')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='Decay for EMA')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for AdamW')
    parser.add_argument('--layerwise_lr_decay', type=float, default=0.95, help='Layer-wise learning rate decay factor')
    parser.add_argument('--grad_accum_steps', type=int, default=2, help='Steps for gradient accumulation to save VRAM')
    parser.add_argument('--n_folds', type=int, default=3, help='Number of folds for cross validation')
    parser.add_argument('--use_gradient_checkpointing', action='store_true', default=True, help='Use gradient checkpointing to save VRAM')
    args = parser.parse_args()
    run_training(args)
