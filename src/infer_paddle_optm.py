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
        attn_weights = paddle.softmax(attn_weights, axis=1) * mask
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

def run_inference(args):
    paddle.set_device(args.device)
    
    # Load labels
    labels_path = os.path.join(args.model_dir, 'label_classes.txt')
    if not os.path.exists(labels_path):
        raise ValueError(f"Label path not found: {labels_path}")
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    num_classes = len(labels)
    print(f"Loaded {num_classes} labels.")

    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = CPAModel(args.shortcut_name, num_classes, use_contrastive=False)
    
    model_path = os.path.join(args.model_dir, 'best_model.pdparams')
    if not os.path.exists(model_path):
        raise ValueError(f"Model weights not found: {model_path}")
    state_dict = paddle.load(model_path)
    model.set_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {model_path}")

    # Load Test Data
    test_df = pd.read_csv(args.input_csv)
    print(f"Running inference on {len(test_df)} samples...")
    results = []

    with paddle.no_grad():
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inference"):
            text_input = f"{str(row['Subject'])} [SEP] {str(row['Object'])}"
            
            try:
                encoded = tokenizer(
                    text_input,
                    max_length=args.max_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True
                )
            except TypeError:
                encoded = tokenizer(
                    text=text_input,
                    max_seq_len=args.max_length,
                    pad_to_max_seq_len=True,
                    truncation=True,
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
    parser.add_argument('--model_dir', type=str, required=True, help="Path to model directory containing best_model.pdparams and tokenizer")
    parser.add_argument('--output_file', type=str, default='./result/submission.csv')
    parser.add_argument('--shortcut_name', type=str, default='bert-base-chinese')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--device', type=str, default='gpu')
    run_inference(parser.parse_args())