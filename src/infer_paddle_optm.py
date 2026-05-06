import argparse
import os
import warnings
import numpy as np
import pandas as pd
import paddle
import paddle.nn as nn
from paddlenlp.transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==========================================
# 1. Model Definition (Must match training)
# ==========================================
class CPAModel(nn.Layer):
    def __init__(self, shortcut_name, num_classes, hidden_size=None):
        super(CPAModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(shortcut_name)
        if hidden_size is None:
            self.hidden_size = self.encoder.config['hidden_size']
        else:
            self.hidden_size = hidden_size
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        seq_output = outputs[0]
        mask = paddle.cast(attention_mask, dtype='float32').unsqueeze(-1)
        sum_embeddings = paddle.sum(seq_output * mask, axis=1)
        sum_mask = paddle.sum(mask, axis=1)
        mean_pooled = sum_embeddings / sum_mask
        
        logits = self.classifier(self.dropout(mean_pooled))
        return logits

def run_inference(args):
    paddle.set_device(args.device)
    
    # Load labels
    if not os.path.exists(args.labels_path):
        raise ValueError(f"Label path not found: {args.labels_path}")
    with open(args.labels_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    num_classes = len(labels)
    print(f"Loaded {num_classes} labels.")

    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(args.shortcut_name)
    model = CPAModel(args.shortcut_name, num_classes)
    
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model weights not found: {args.model_path}")
    state_dict = paddle.load(args.model_path)
    model.set_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {args.model_path}")

    # Load Test Data
    test_df = pd.read_csv(args.input_csv)
    print(f"Running inference on {len(test_df)} samples...")
    results = []

    with paddle.no_grad():
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inference"):
            text_input = f"{str(row['Subject'])} [SEP] {str(row['Object'])}"
            
            encoded = tokenizer(
                text_input,
                max_seq_len=args.max_length,
                pad_to_max_seq_len=True,
                truncation_strategy='longest_first',
                return_attention_mask=True
            )
            
            ids = paddle.to_tensor([encoded['input_ids']], dtype='int64')
            mask = paddle.to_tensor([encoded['attention_mask']], dtype='int64')
            
            logits = model(ids, mask)
            pred_id = paddle.argmax(logits, axis=1).item()
            results.append(labels[pred_id])

    # Save Results
    test_df['Label'] = results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    test_df.to_csv(args.output_file, index=False, encoding='utf-8-sig')
    print(f"Inference completed. Results saved to: {args.output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, default="./dataset/test.csv")
    parser.add_argument('--labels_path', type=str, required=True, help="Path to label_classes.txt")
    parser.add_argument('--model_path', type=str, required=True, help="Path to best_model.pdparams")
    parser.add_argument('--output_file', type=str, default='./result/submission.csv')
    parser.add_argument('--shortcut_name', type=str, default='bert-base-uncased')
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--device', type=str, default='gpu')
    run_inference(parser.parse_args())
