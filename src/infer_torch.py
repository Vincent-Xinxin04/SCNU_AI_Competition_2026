import argparse
import os
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


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


class SingleTableInferenceDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.original_rows = []

        df = pd.read_csv(csv_path, low_memory=False, encoding='utf-8-sig')

        df.columns = [str(col).strip() for col in df.columns]

        subject_col = None
        object_col = None
        for col in df.columns:
            if col.lower() == 'subject':
                subject_col = col
            elif col.lower() == 'object':
                object_col = col

        if subject_col is None or object_col is None:
            raise ValueError("The CSV file must contain 'Subject' and 'Object' columns (case-insensitive).")

        temp_df = df[[subject_col, object_col]].dropna()
        for idx, row in temp_df.iterrows():
            self.samples.append((str(row[subject_col]), str(row[object_col])))
            self.original_rows.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subject_text, object_text = self.samples[idx]
        input_ids, attention_mask = encode_pair(
            self.tokenizer,
            subject_text,
            object_text,
            self.max_length,
        )
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'orig_idx': np.int64(idx),
        }


def collate_fn(samples):
    return {
        'input_ids': np.stack([s['input_ids'] for s in samples]).astype('int64'),
        'attention_mask': np.stack([s['attention_mask'] for s in samples]).astype('int64'),
        'orig_idx': np.array([s['orig_idx'] for s in samples], dtype='int64'),
    }


def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    with open(args.labels_path, 'r', encoding='utf-8-sig') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    id2label = {idx: label for idx, label in enumerate(classes)}

    tokenizer = AutoTokenizer.from_pretrained(args.shortcut_name)
    model = CPAModel(args.shortcut_name, len(classes))

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f'Model file not found: {args.model_path}')

    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    dataset = SingleTableInferenceDataset(args.input_csv, tokenizer, args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    print(f'Starting inference. Total valid rows: {len(dataset)}')
    predictions = [None] * len(dataset)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Running inference'):
            ids = torch.tensor(batch['input_ids'], dtype=torch.long, device=device)
            mask = torch.tensor(batch['attention_mask'], dtype=torch.long, device=device)
            orig_indices = batch['orig_idx'].tolist()

            logits = model(ids, mask)

            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            for idx_in_batch, pred_idx in enumerate(preds):
                original_position = orig_indices[idx_in_batch]
                predictions[original_position] = id2label[pred_idx]

    original_df = pd.read_csv(args.input_csv, low_memory=False, encoding='utf-8-sig')
    original_df.columns = [str(col).strip() for col in original_df.columns]

    subject_col = None
    object_col = None
    for col in original_df.columns:
        if col.lower() == 'subject':
            subject_col = col
        elif col.lower() == 'object':
            object_col = col

    if subject_col is None or object_col is None:
        raise ValueError("The CSV file must contain 'Subject' and 'Object' columns (case-insensitive).")

    valid_mask = original_df[subject_col].notna() & original_df[object_col].notna()
    valid_indices = original_df[valid_mask].index.tolist()

    original_df['Label'] = None
    for row_idx, pred_label in zip(valid_indices, predictions):
        original_df.loc[row_idx, 'Label'] = pred_label

    original_df.to_csv(args.output_file, index=False, encoding='utf-8-sig')
    print(f'Inference completed. Results saved to: {args.output_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, default="./dataset/test.csv")
    parser.add_argument('--labels_path', type=str, default="./dataset/labels.txt")
    parser.add_argument('--model_path', type=str, default="./cpa_output/cpa_20260430_112547/best_model.pth")
    parser.add_argument('--output_file', type=str, default='./submission.csv')
    parser.add_argument('--shortcut_name', type=str, default='bert-base-uncased')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_amp', action='store_true')
    args = parser.parse_args()
    run_inference(args)
