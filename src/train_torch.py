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
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


def load_data_from_directory(dir_path):
    all_data = []
    if not os.path.exists(dir_path):
        raise ValueError(f"can't find: {dir_path}")

    try:
        csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    except PermissionError:
        logging.error(f"Permission denied: {dir_path}")
        return []
    logging.info(f"load data from {dir_path} ...")

    for filename in tqdm(csv_files, desc=f"loading {os.path.basename(dir_path)}"):
        file_path = os.path.join(dir_path, filename)
        label_name = filename[:-4]
        try:
            df = pd.read_csv(file_path, low_memory=False, encoding='utf-8-sig')
            if df.empty:
                continue
            df.columns = [str(col).strip() for col in df.columns]
            
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


def encode_pair(tokenizer, text_a, text_b, max_length):
    encoding = tokenizer(
        text=text_a,
        text_pair=text_b,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='np'
    )
    
    input_ids = encoding['input_ids']
    attention_mask = encoding.get('attention_mask', None)

    if attention_mask is None:
        seq_len = len(input_ids)
        seq_len = min(seq_len, max_length)
        attention_mask = [1] * seq_len + [0] * (max_length - seq_len)

    return np.array(input_ids, dtype='int64'), np.array(attention_mask, dtype='int64')


class RelationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, label_encoder, max_length=128):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.le = label_encoder
        self.max_length = max_length

        self.subject_col = None
        self.object_col = None
        for col in self.data.columns:
            if col.lower() == 'subject':
                self.subject_col = col
            elif col.lower() == 'object':
                self.object_col = col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        subject_text = str(row[self.subject_col])
        object_text = str(row[self.object_col])
        input_ids, attention_mask = encode_pair(self.tokenizer, subject_text, object_text, self.max_length)
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


class CPAModel(nn.Module):
    def __init__(self, shortcut_name, num_classes, hidden_size=None):
        super(CPAModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(shortcut_name)
        if hidden_size is None:
            self.hidden_size = self.encoder.config.hidden_size
        else:
            self.hidden_size = hidden_size
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq_output = outputs.last_hidden_state
        mask = attention_mask.float().unsqueeze(-1)
        sum_embeddings = torch.sum(seq_output * mask, dim=1)
        sum_mask = torch.sum(mask, dim=1)
        mean_pooled = sum_embeddings / sum_mask
        
        logits = self.classifier(self.dropout(mean_pooled))
        return logits


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                if param.grad is not None:
                    norm = torch.norm(param.grad)
                    if norm > 0 and not torch.isnan(norm):
                        r_at = epsilon * param.grad / norm
                        param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data.copy_(self.backup[name])
        self.backup = {}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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


def save_label_classes(label_encoder, save_dir):
    path = os.path.join(save_dir, 'label_classes.txt')
    with open(path, 'w', encoding='utf-8') as f:
        for label in label_encoder.classes_:
            f.write(f'{label}\n')


def calculate_few_shot_weights(counts_dict, label_encoder, device):
    counts = np.array([counts_dict.get(label, 1) for label in label_encoder.classes_], dtype=np.float32)
    c_max = np.max(counts)
    c_min = np.min(counts)
    
    weights = (c_max - counts + c_min * 0.1) / (c_max + c_min * 0.1)
    
    weights = weights * 0.8 + 0.2 * np.ones_like(weights)
    
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_training(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.output_dir, f'cpa_{timestamp}')
    setup_logging(save_dir)
    set_seed(args.random_seed)
    device = resolve_device(args.device)

    logging.info(f'device: {device}')

    raw_train_df = load_data_from_directory(args.train_dir)

    label_encoder = LabelEncoder()
    label_encoder.fit(raw_train_df['label'].unique())
    num_classes = len(label_encoder.classes_)
    logging.info(f'label_num: {num_classes}')
    save_label_classes(label_encoder, save_dir)
    
    label_counts = raw_train_df['label'].value_counts().to_dict()
    class_weights = calculate_few_shot_weights(label_counts, label_encoder, device)
    logging.info("few-shot weights calculated and applied to Loss function.")

    counts = raw_train_df['label'].value_counts()
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
    val_df = val_c.reset_index(drop=True)
    logging.info(f'split success: train={len(train_df)}, val={len(val_df)}')

    tokenizer = AutoTokenizer.from_pretrained(args.shortcut_name)
    
    train_dataset = RelationDataset(train_df, tokenizer, label_encoder, args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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

    model = CPAModel(args.shortcut_name, num_classes)
    model.to(device)
    
    total_steps = max(1, len(train_loader) * args.epoch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps
    )
    
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    fgm = FGM(model)

    use_amp = args.use_amp and device.type != 'cpu'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

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

            input_ids = torch.tensor(batch['data'], dtype=torch.long, device=device)
            attention_mask = torch.tensor(batch['cls_mask'], dtype=torch.long, device=device)
            label_ids = torch.tensor(batch['label'], dtype=torch.long, device=device)

            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(input_ids, attention_mask)
                    loss = loss_fn(logits, label_ids)
                scaler.scale(loss).backward()
                
                fgm.attack(epsilon=args.fgm_epsilon)
                with torch.cuda.amp.autocast():
                    logits_adv = model(input_ids, attention_mask)
                    loss_adv = loss_fn(logits_adv, label_ids)
                scaler.scale(loss_adv).backward()
                fgm.restore()
                
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits, label_ids)
                loss.backward()
                
                fgm.attack(epsilon=args.fgm_epsilon)
                logits_adv = model(input_ids, attention_mask)
                loss_adv = loss_fn(logits_adv, label_ids)
                loss_adv.backward()
                fgm.restore()
                
                optimizer.step()

            lr_scheduler.step()
            optimizer.zero_grad()
            loss_value = loss.item()
            tr_loss += loss_value
            train_steps += 1
            pbar.set_postfix({'loss': f'{loss_value:.4f}'})

        m_weights = class_weights.cpu().numpy()
        m_correct = np.zeros(num_classes)
        m_total = np.zeros(num_classes)
        
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                input_ids = torch.tensor(batch['data'], dtype=torch.long, device=device)
                attention_mask = torch.tensor(batch['cls_mask'], dtype=torch.long, device=device)
                label_ids = batch['label']
                
                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels = label_ids
                
                for p, l in zip(preds, labels):
                    m_total[l] += 1
                    if p == l:
                        m_correct[l] += 1
        
        m_scores = np.divide(m_correct, m_total, out=np.zeros_like(m_correct), where=m_total!=0)
        valid_mask = m_total > 0
        final_score = np.sum(m_weights[valid_mask] * m_scores[valid_mask]) / np.sum(m_weights[valid_mask])
        
        val_acc = np.sum(m_correct) / np.sum(m_total) if np.sum(m_total) > 0 else 0.0
        logging.info(f"Epoch {epoch+1} | Loss: {tr_loss/max(1, train_steps):.4f} | Val Acc: {val_acc:.4f} | Score_final: {final_score:.4f}")

        if final_score > best_final_score:
            best_final_score = final_score
            patience_counter = 0
            logging.info(f"best model! (Score_final: {final_score:.4f})")
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
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

    logging.info(f'train finish, best score: {best_final_score:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default="./dataset/Train_Set")
    parser.add_argument('--output_dir', type=str, default='./cpa_output')
    parser.add_argument('--shortcut_name', type=str, default='bert-base-chinese')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--use_flash_attention', action='store_true')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--fgm_epsilon', type=float, default=1.0)
    args = parser.parse_args()
    run_training(args)
