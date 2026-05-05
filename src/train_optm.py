import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ==========================================
# 1. Config & Seeding
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# ==========================================
# 2. Focal Loss for Few-Shot
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, 
                                  label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce_loss.clamp(max=10))
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

# ==========================================
# 3. Adversarial Training (FGM)
# ==========================================
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

# ==========================================
# 4. Enhanced Semantic Prototypical Model
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
                # Use MEAN pooling for better semantic representation
                mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size())
                proto = torch.sum(outputs.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(1), min=1e-9)
                prototypes.append(proto.squeeze(0))
        return torch.stack(prototypes)

    def forward(self, input_ids, attention_mask, return_features=False):
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
        
        if return_features:
            return logits, sample_vec_norm
        
        return logits

# ==========================================
# 5. Dataset
# ==========================================
class RelationDataset(Dataset):
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
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(row['label_id'], dtype=torch.long)
        }

# ==========================================
# 6. Training Function
# ==========================================
def load_all_data(train_dir, weights_path, pseudo_path=None):
    with open(weights_path, 'r', encoding='utf-8') as f:
        weights_data = json.load(f)
    all_rows = []
    for label in weights_data.keys():
        file_path = os.path.join(train_dir, f"{label}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, low_memory=False, encoding='utf-8-sig')
            df['label'] = label
            all_rows.append(df[['Subject', 'Object', 'label']])
    full_df = pd.concat(all_rows, ignore_index=True)
    if pseudo_path and os.path.exists(pseudo_path):
        pseudo_df = pd.read_csv(pseudo_path, low_memory=False, encoding='utf-8-sig')
        pseudo_df.columns = ['Subject', 'Object', 'label']
        pseudo_df = pseudo_df[pseudo_df['label'].isin(weights_data.keys())]
        full_df = pd.concat([full_df, pseudo_df], ignore_index=True)
    return full_df, weights_data

def train():
    MODEL_NAME = "FacebookAI/xlm-roberta-large"
    TRAIN_DIR = "/home/SCNU_AI_Competition/dataset/Train_Set"
    WEIGHTS_PATH = "/home/SCNU_AI_Competition/relation_weights.json"
    PSEUDO_PATH = "/home/SCNU_AI_Competition/dataset/Pseudo_Set/pseudo_labels.csv"
    OUTPUT_DIR = "/home/SCNU_AI_Competition/model/v5_xlmr_large_focal"
    BATCH_SIZE = 16
    EPOCHS = 15
    LR = 1e-5
    MAX_LENGTH = 192
    RDROP_ALPHA = 0.8
    FGM_EPSILON = 1.0
    LABEL_SMOOTHING = 0.05
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    full_df, weights_dict = load_all_data(TRAIN_DIR, WEIGHTS_PATH, PSEUDO_PATH)
    le = LabelEncoder()
    full_df['label_id'] = le.fit_transform(full_df['label'])
    num_labels = len(le.classes_)
    label_names = le.classes_.tolist()
    
    # Stratified Split - Protect few-shot classes
    counts = full_df['label'].value_counts()
    rare_labels = counts[counts < 5].index
    df_rare = full_df[full_df['label'].isin(rare_labels)]
    df_common = full_df[~full_df['label'].isin(rare_labels)]
    train_c, val_c = train_test_split(df_common, test_size=0.03, stratify=df_common['label'], random_state=42)
    train_df = pd.concat([train_c, df_rare]).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = val_c.reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = SemanticRelationModel(MODEL_NAME, label_names, tokenizer, device).to(device)
    
    # Enhanced Class Weights
    class_weights = torch.zeros(num_labels).to(device)
    for idx, label in enumerate(le.classes_):
        count = counts.get(label, 1)
        weight = (counts.max() / count) ** 0.5
        class_weights[idx] = weight
    
    # Use Focal Loss with class weights and label smoothing
    criterion = FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=LABEL_SMOOTHING)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    train_loader = DataLoader(RelationDataset(train_df, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=True)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05*total_steps), num_training_steps=total_steps)
    
    fgm = FGM(model)
    best_score = 0
    scaler = torch.cuda.amp.GradScaler()
    
    # Early Stopping
    patience = 5
    no_improve_epoch = 0

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        total_loss = 0
        for batch_idx, batch in enumerate(pbar):
            ids, mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                # R-Drop Forward
                logits1 = model(ids, mask)
                logits2 = model(ids, mask)
                
                # Classification loss
                ce_loss = (criterion(logits1, labels) + criterion(logits2, labels)) / 2
                
                # R-Drop KL loss
                kl_loss = (F.kl_div(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1), reduction='batchmean') + 
                           F.kl_div(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1), reduction='batchmean')) / 2
                
                # Combined loss
                loss = ce_loss + RDROP_ALPHA * kl_loss
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"NaN detected at epoch {epoch+1}, batch {batch_idx}")
                loss = ce_loss  # Fallback to CE loss
            
            total_loss += loss.item()
            scaler.scale(loss).backward()
            
            # FGM Attack with increased epsilon
            fgm.attack(epsilon=FGM_EPSILON, emb_name='embeddings.word_embeddings')
            with torch.cuda.amp.autocast():
                logits_adv = model(ids, mask)
                loss_adv = criterion(logits_adv, labels)
            scaler.scale(loss_adv * 0.2).backward()
            fgm.restore(emb_name='embeddings.word_embeddings')
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            pbar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

        # Eval
        model.eval()
        m_scores = {label: [0, 0] for label in label_names}
        with torch.no_grad():
            for batch in DataLoader(RelationDataset(val_df, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE):
                logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                for p, l in zip(preds, batch['label'].numpy()):
                    name = label_names[l]
                    m_scores[name][1] += 1
                    if p == l: m_scores[name][0] += 1
        
        f_sum, w_sum = 0, 0
        for name, (c, t) in m_scores.items():
            if t > 0:
                f_sum += (c/t) * weights_dict[name]['weight']
                w_sum += weights_dict[name]['weight']
        final_score = f_sum / w_sum if w_sum > 0 else 0
        print(f"Score: {final_score:.4f}")
        
        if final_score > best_score:
            best_score = final_score
            no_improve_epoch = 0  # Reset patience counter
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_semantic.pt"))
            tokenizer.save_pretrained(OUTPUT_DIR)
            with open(os.path.join(OUTPUT_DIR, "labels.txt"), "w") as f:
                for l in label_names: f.write(f"{l}\n")
            print(f"Saved! Best Score: {best_score:.4f}")
        else:
            no_improve_epoch += 1
            if no_improve_epoch >= patience:
                print(f"Early stopping at epoch {epoch+1} - no improvement for {patience} epochs")
                print(f"Best Score achieved: {best_score:.4f}")
                break

if __name__ == "__main__":
    train()