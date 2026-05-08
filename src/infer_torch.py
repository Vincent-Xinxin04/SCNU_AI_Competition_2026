import argparse
import os
import pandas as pd
import warnings
import re

# Set Hugging Face Mirror (Allow override via environment variable)
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# ==========================================
# 1. Model definition (must match training)
# ==========================================
class CPAModel(nn.Module):
    def __init__(self, shortcut_name, num_classes):
        super(CPAModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(shortcut_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token pooling
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_embedding))
        return logits


# ==========================================
# 2. Tokenization helper
# ==========================================
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


# ==========================================
# 3. Single-table inference dataset
# ==========================================
class SingleTableInferenceDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=128, use_type_hint=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_type_hint = use_type_hint
        self.samples = []
        self.original_rows = []

        # Read the CSV file.
        df = pd.read_csv(csv_path, low_memory=False, encoding='utf-8-sig')

        # Normalize column names by trimming whitespace.
        df.columns = [str(col).strip() for col in df.columns]

        # Locate Subject and Object columns in a case-insensitive way.
        subject_col = None
        object_col = None
        for col in df.columns:
            if col.lower() == 'subject':
                subject_col = col
            elif col.lower() == 'object':
                object_col = col

        if subject_col is None or object_col is None:
            raise ValueError("The CSV file must contain 'Subject' and 'Object' columns (case-insensitive).")

        # Drop rows with missing Subject/Object values.
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
            use_type_hint=self.use_type_hint
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


# ==========================================
# 4. Inference pipeline
# ==========================================
def run_inference(args):
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    print(f'Using device: {device}')

    # Load label mapping.
    with open(args.labels_path, 'r', encoding='utf-8-sig') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    id2label = {idx: label for idx, label in enumerate(classes)}

    # Initialize tokenizer and model.
    tokenizer = AutoTokenizer.from_pretrained(args.shortcut_name)
    model = CPAModel(args.shortcut_name, len(classes))

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f'Model file not found: {args.model_path}')

    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load the dataset.
    dataset = SingleTableInferenceDataset(args.input_csv, tokenizer, args.max_length, use_type_hint=args.use_type_hint)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    print(f'Starting inference. Total valid rows: {len(dataset)}')
    predictions = [None] * len(dataset)
    use_amp = args.use_amp and device.type != 'cpu'

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Running inference'):
            ids = torch.tensor(batch['input_ids'], dtype=torch.long, device=device)
            mask = torch.tensor(batch['attention_mask'], dtype=torch.long, device=device)
            orig_indices = batch['orig_idx'].tolist()

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(ids, mask)
            else:
                logits = model(ids, mask)

            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            for idx_in_batch, pred_idx in enumerate(preds):
                original_position = orig_indices[idx_in_batch]
                predictions[original_position] = id2label[pred_idx]

    # Reload the original CSV and attach predictions.
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

    # Only rows with valid Subject/Object pairs receive predictions.
    valid_mask = original_df[subject_col].notna() & original_df[object_col].notna()
    valid_indices = original_df[valid_mask].index.tolist()

    original_df['Label'] = None
    for row_idx, pred_label in zip(valid_indices, predictions):
        original_df.loc[row_idx, 'Label'] = pred_label

    # Save the result without modifying the source file.
    original_df.to_csv(args.output_file, index=False, encoding='utf-8-sig')
    print(f'Inference completed. Results saved to: {args.output_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, default="./dataset/test.csv")
    parser.add_argument('--labels_path', type=str, default="./model/label_classes.txt")
    parser.add_argument('--model_path', type=str, default="./model/best_model.pth")
    parser.add_argument('--output_file', type=str, default='./submission.csv')
    parser.add_argument('--shortcut_name', type=str, default='bert-base-uncased')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--use_type_hint', action='store_true', help='Use data type hinting in features')
    args = parser.parse_args()
    run_inference(args)
