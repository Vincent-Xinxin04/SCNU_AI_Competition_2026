import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel


# ==================== Model Definition ====================
class CPAModel(nn.Module):
    def __init__(self, model_name, num_labels, dropout_rate=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        
        if hasattr(self.encoder.config, 'hidden_size'):
            hidden_size = self.encoder.config.hidden_size
        elif hasattr(self.encoder.config, 'dim'):
            hidden_size = self.encoder.config.dim
        else:
            hidden_size = 768
        
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


# ==================== Inference Dataset ====================
class InferenceDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.original_rows = []
        
        df = pd.read_csv(csv_path, low_memory=False, encoding='utf-8-sig')
        df.columns = [str(col).strip() for col in df.columns]
        
        # 查找Subject和Object列
        subject_col = None
        object_col = None
        for col in df.columns:
            if col.lower() == 'subject':
                subject_col = col
            elif col.lower() == 'object':
                object_col = col
        
        if subject_col is None or object_col is None:
            raise ValueError("CSV file must contain 'Subject' and 'Object' columns")
        
        temp_df = df[[subject_col, object_col]].dropna()
        for idx, row in temp_df.iterrows():
            self.samples.append((str(row[subject_col]), str(row[object_col])))
            self.original_rows.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subject_text, object_text = self.samples[idx]
        text_input = f"{subject_text} [SEP] {object_text}"
        
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
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'orig_idx': torch.tensor(idx, dtype=torch.long)
        }


def collate_fn(samples):
    return {
        'input_ids': torch.stack([s['input_ids'] for s in samples]),
        'attention_mask': torch.stack([s['attention_mask'] for s in samples]),
        'orig_idx': torch.tensor([s['orig_idx'] for s in samples], dtype=torch.long)
    }


# ==================== Inference Pipeline ====================
def run_inference(args):
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')
    print(f'Device: {device}')
    
    # 加载标签映射
    with open(args.labels_path, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]
    id2label = {idx: label for idx, label in enumerate(classes)}
    num_labels = len(classes)
    
    # 初始化tokenizer和模型
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    print(f"Loading model: {args.model_path}")
    model = CPAModel(args.model_name, num_labels)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 加载数据集
    print(f"Loading dataset: {args.input_csv}")
    dataset = InferenceDataset(args.input_csv, tokenizer, args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    print(f"Starting inference. Total rows: {len(dataset)}")
    predictions = [None] * len(dataset)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Running inference'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            orig_indices = batch['orig_idx'].tolist()
            
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            
            for idx_in_batch, pred_idx in enumerate(preds):
                original_position = orig_indices[idx_in_batch]
                predictions[original_position] = id2label[pred_idx]
    
    # 保存结果
    original_df = pd.read_csv(args.input_csv, low_memory=False, encoding='utf-8-sig')
    original_df.columns = [str(col).strip() for col in original_df.columns]
    
    subject_col = None
    object_col = None
    for col in original_df.columns:
        if col.lower() == 'subject':
            subject_col = col
        elif col.lower() == 'object':
            object_col = col
    
    valid_mask = original_df[subject_col].notna() & original_df[object_col].notna()
    valid_indices = original_df[valid_mask].index.tolist()
    
    original_df['Label'] = None
    for row_idx, pred_label in zip(valid_indices, predictions):
        original_df.loc[row_idx, 'Label'] = pred_label
    
    original_df.to_csv(args.output_file, index=False, encoding='utf-8-sig')
    print(f"Inference completed. Results saved to: {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Inference for CPA Task')
    
    parser.add_argument('--input_csv', type=str, default="./dataset/test.csv",
                        help='Path to input CSV file')
    parser.add_argument('--labels_path', type=str, required=True,
                        help='Path to label classes file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--output_file', type=str, default='./submission.csv',
                        help='Output CSV file path')
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-base',
                        help='Pre-trained model name')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='gpu')
    
    args = parser.parse_args()
    run_inference(args)