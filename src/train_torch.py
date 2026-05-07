import argparse
import os
import random
import logging
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

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
def encode_pair(tokenizer, text_a, text_b, max_length):
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
    def __init__(self, dataframe, tokenizer, label_encoder, max_length=64, augment=False):
        self.augment = augment
        self.max_length = max_length
        
        # Pre-process Subject and Object columns
        subject_col, object_col = None, None
        for col in dataframe.columns:
            if col.lower() == 'subject': subject_col = col
            elif col.lower() == 'object': object_col = col
        
        subjects = dataframe[subject_col].astype(str).tolist()
        objects = dataframe[object_col].astype(str).tolist()
        self.labels = label_encoder.transform(dataframe['label']).astype('int64').tolist()

        logging.info(f"Pre-tokenizing {len(subjects)} samples...")
        # Use batch encoding for massive speedup
        encodings = tokenizer(
            text=subjects,
            text_pair=objects,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors=None
        )
        self.input_ids = encodings['input_ids']
        self.attention_masks = encodings['attention_mask']
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.labels)

    def _apply_token_augmentation(self, input_ids):
        # Faster augmentation directly on token IDs
        input_ids = list(input_ids)
        length = len(input_ids)
        
        # Randomly delete non-special tokens
        if random.random() < 0.1:
            # Keep [CLS] and [SEP], delete one random token
            idx = random.randint(1, length - 2)
            if input_ids[idx] not in [self.cls_token_id, self.sep_token_id, self.pad_token_id]:
                input_ids.pop(idx)
                input_ids.append(self.pad_token_id)
        
        return np.array(input_ids, dtype='int64')

    def __getitem__(self, idx):
        input_ids = np.array(self.input_ids[idx], dtype='int64')
        attention_mask = np.array(self.attention_masks[idx], dtype='int64')
        
        if self.augment:
            input_ids = self._apply_token_augmentation(input_ids)
            
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
    def __init__(self, shortcut_name, num_classes):
        super(CPAModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(shortcut_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token pooling to align with baseline
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

    # 3. split dataset
    counts = raw_train_df['label'].value_counts()
    
    # Calculate competition weights
    counts_max = counts.max()
    counts_min = counts.min()
    m_weights_dict = {}
    for label, count in counts.items():
        weight = (counts_max - count + counts_min * 0.1) / (counts_max + counts_min * 0.1)
        m_weights_dict[label] = weight
    
    # Create weight tensor for loss function
    class_weights = torch.zeros(num_classes)
    for label, weight in m_weights_dict.items():
        class_idx = label_encoder.transform([label])[0]
        class_weights[class_idx] = weight
    class_weights = class_weights.to(device)
    logging.info("Calculated few-shot importance weights for loss function.")

    rare_labels = counts[counts < 2].index
    df_rare = raw_train_df[raw_train_df['label'].isin(rare_labels)]
    df_common = raw_train_df[~raw_train_df['label'].isin(rare_labels)]

    if len(df_common) == 0:
        raise ValueError("data num < 2, can't split dataset")

    train_c, val_c = train_test_split(
        df_common,
        test_size=args.val_ratio,
        stratify=df_common['label'],
        random_state=args.random_seed,
    )
    train_df = pd.concat([train_c, df_rare]).sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
    
    # Oversampling rare classes if specified
    if args.oversample_rare:
        logging.info("Oversampling rare classes...")
        # Define 'rare' as classes with fewer than a certain threshold, e.g., median count or a fixed number
        threshold = counts.median()
        rare_to_oversample = counts[counts < threshold].index
        
        oversampled_parts = []
        for label in rare_to_oversample:
            label_df = train_df[train_df['label'] == label]
            if len(label_df) == 0: continue
            # Duplicate up to the threshold (or at least 5 times if very rare)
            num_repeats = int(max(2, threshold // len(label_df)))
            oversampled_parts.append(pd.concat([label_df] * num_repeats))
        
        if oversampled_parts:
            train_df = pd.concat([train_df] + oversampled_parts).sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
            logging.info(f"Oversampling complete. New train size: {len(train_df)}")

    val_df = val_c.reset_index(drop=True)
    logging.info(f'split success: train={len(train_df)}, val={len(val_df)}')

    # 4. Tokenizer & DataLoader
    tokenizer = AutoTokenizer.from_pretrained(args.shortcut_name)
    
    train_dataset = RelationDataset(train_df, tokenizer, label_encoder, args.max_length, augment=args.use_augmentation)
    
    sampler = None
    if args.use_balanced_sampler:
        logging.info("Using WeightedRandomSampler for class balancing...")
        class_sample_counts = train_df['label'].value_counts()
        # Weight for each class is 1/count
        weights = 1.0 / class_sample_counts
        # Weight for each sample is the weight of its class
        sample_weights = train_df['label'].map(weights).values
        sample_weights = torch.from_numpy(sample_weights)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=dynamic_collate_fn,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        RelationDataset(val_df, tokenizer, label_encoder, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dynamic_collate_fn,
        num_workers=args.num_workers,
    )

    # 5. model init
    model = CPAModel(args.shortcut_name, num_classes)
    model.to(device)
    
    total_steps = max(1, len(train_loader) * args.epoch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps
    )
    
    # Use specialized loss if specified
    if args.use_focal_loss:
        logging.info("Using FocalLoss to handle class imbalance.")
        loss_fn = FocalLoss(weight=class_weights if args.use_weighted_loss else None, 
                           gamma=args.focal_gamma, 
                           label_smoothing=args.label_smoothing)
    elif args.use_weighted_loss:
        logging.info("Using weighted CrossEntropyLoss based on few-shot importance.")
        loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    else:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    use_amp = args.use_amp and device.type != 'cpu'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # 6. training
    best_acc = 0.0
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

            input_ids = torch.tensor(batch['data'], dtype=torch.long, device=device)
            mask = torch.tensor(batch['cls_mask'], dtype=torch.long, device=device)
            label_ids = torch.tensor(batch['label'], dtype=torch.long, device=device)

            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    if args.use_rdrop:
                        # Concatenate inputs for a single forward pass (much faster)
                        input_ids_double = torch.cat([input_ids, input_ids], dim=0)
                        mask_double = torch.cat([mask, mask], dim=0)
                        logits_all = model(input_ids_double, mask_double)
                        logits, logits2 = torch.split(logits_all, input_ids.size(0))
                        
                        ce_loss = 0.5 * (loss_fn(logits, label_ids) + loss_fn(logits2, label_ids))
                        kl_loss = compute_kl_loss(logits, logits2)
                        loss = ce_loss + args.rdrop_alpha * kl_loss
                    else:
                        logits = model(input_ids, mask)
                        loss = loss_fn(logits, label_ids)
                        
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.use_rdrop:
                    input_ids_double = torch.cat([input_ids, input_ids], dim=0)
                    mask_double = torch.cat([mask, mask], dim=0)
                    logits_all = model(input_ids_double, mask_double)
                    logits, logits2 = torch.split(logits_all, input_ids.size(0))
                    
                    ce_loss = 0.5 * (loss_fn(logits, label_ids) + loss_fn(logits2, label_ids))
                    kl_loss = compute_kl_loss(logits, logits2)
                    loss = ce_loss + args.rdrop_alpha * kl_loss
                else:
                    logits = model(input_ids, mask)
                    loss = loss_fn(logits, label_ids)
                    
                loss.backward()
                optimizer.step()

            lr_scheduler.step()
            loss_value = loss.item()
            tr_loss += loss_value
            train_steps += 1
            pbar.set_postfix({'loss': f'{loss_value:.4f}'})

        # val stage
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                input_ids = torch.tensor(batch['data'], dtype=torch.long, device=device)
                mask = torch.tensor(batch['cls_mask'], dtype=torch.long, device=device)
                label_ids = torch.tensor(batch['label'], dtype=torch.long, device=device)
                
                logits = model(input_ids, mask)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == label_ids).sum().item()
                val_total += label_ids.size(0)

        avg_train_loss = tr_loss / max(1, train_steps)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        logging.info(f'Epoch {epoch + 1} | Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            try:
                tokenizer.save_pretrained(save_dir)
            except Exception:
                pass
            logging.info(f'best model! (Acc: {best_acc:.4f})')
        else:
            patience_counter += 1
            logging.info(f'early stop count: {patience_counter}/{patience_limit}')
            if patience_counter == patience_limit:
                logging.info(f'{patience_limit} epoch not up, early stop!!!')
                break

    logging.info(f'train finish, best acc: {best_acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default="./dataset/Train_Set")
    parser.add_argument('--output_dir', type=str, default='./cpa_output')
    parser.add_argument('--shortcut_name', type=str, default='bert-base-uncased')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--use_amp', action='store_true')
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
    args = parser.parse_args()
    run_training(args)
