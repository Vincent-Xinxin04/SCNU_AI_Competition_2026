import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ==========================================
# Model Definition (same as training)
# ==========================================
class SemanticRelationModel(nn.Module):
    def __init__(self, model_name, label_names, tokenizer, device):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.device = device

        # Attention Pooling for sample
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )

        # Move encoder to device before pre-encoding
        self.encoder.to(device)

        # Pre-encode label names to create "Prototypes"
        print("Encoding label semantics...")
        self.tokenizer = tokenizer
        self.label_names = label_names
        self.label_prototypes = self._encode_labels(label_names).to(device)

        # Learnable Prototype Adjustment
        self.prototype_adjust = nn.Linear(self.hidden_size, self.hidden_size)

        # Projection layer to align sample space with label semantic space
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        # Learnable Temperature for Cosine Similarity
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def _encode_labels(self, label_names):
        self.encoder.eval()
        prototypes = []
        with torch.no_grad():
            for name in label_names:
                inputs = self.tokenizer(name, return_tensors='pt', padding=True, truncation=True).to(self.device)
                outputs = self.encoder(**inputs)
                mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size())
                proto = torch.sum(outputs.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(1), min=1e-9)
                prototypes.append(proto.squeeze(0))
        return torch.stack(prototypes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # Attention Pooling
        weights = self.attention(sequence_output)
        weights = F.softmax(weights, dim=1)
        sample_vec = torch.sum(weights * sequence_output, dim=1)

        # Project to semantic space
        sample_vec = self.projector(sample_vec)

        # Get adjusted prototypes
        label_protos = self.prototype_adjust(self.label_prototypes)

        # Normalize for cosine similarity
        sample_vec_norm = sample_vec / sample_vec.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        label_protos_norm = label_protos / label_protos.norm(dim=-1, keepdim=True).clamp(min=1e-9)

        # Cosine Similarity with Temperature
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits = logit_scale * sample_vec_norm @ label_protos_norm.t()

        return logits

# ==========================================
# Test Dataset
# ==========================================
class TestDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = f"{row['Subject']} [SEP] {row['Object']}"
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

# ==========================================
# Inference Function
# ==========================================
def infer():
    # Configuration
    MODEL_DIR = "/home/SCNU_AI_Competition/model/v4_semantic"
    TEST_FILE = "/home/SCNU_AI_Competition/dataset/test.csv"
    OUTPUT_FILE = "/home/SCNU_AI_Competition/submission/semantic_submission.csv"
    BATCH_SIZE = 64
    MAX_LENGTH = 128

    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load label names
    with open(os.path.join(MODEL_DIR, "labels.txt"), "r") as f:
        label_names = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(label_names)} labels")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    print("Loaded tokenizer")

    # Load model
    model = SemanticRelationModel("xlm-roberta-base", label_names, tokenizer, device).to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_semantic.pt"), map_location=device, weights_only=True))
    model.eval()
    print("Loaded model")

    # Load test data
    test_df = pd.read_csv(TEST_FILE, encoding='utf-8-sig')
    print(f"Loaded test data: {len(test_df)} samples")
    print(f"Test columns: {test_df.columns.tolist()}")

    # Create dataloader
    test_dataset = TestDataset(test_df, tokenizer, MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Predict
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            logits = model(ids, mask)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(preds)

    # Convert predictions to label names
    test_df['label'] = [label_names[p] for p in predictions]

    # Save submission
    test_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"Predictions saved to {OUTPUT_FILE}")

    # Print some statistics
    label_counts = test_df['label'].value_counts().head(10)
    print("\nTop 10 predicted labels:")
    print(label_counts)

if __name__ == "__main__":
    infer()